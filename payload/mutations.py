"""payload.mutations (context-aware, token-level)

This module provides:
- A lightweight mutation registry (string id -> function)
- Token-level mutations that operate on a token list + token index
- Context-aware filtering: only offer mutations that make sense for a token type

Important constraints
- Mutations work at payload/chunk level (not character-by-character generation).
- No SQL grammar is required; we use a lightweight tokenizer.

Public functions used by scanner.py
- get_available_mutations(payload) -> dict[token_idx] = [mutation_id,...]
- apply_mutation(payload, mutation_id, token_idx, rng=...) -> new_payload|None

The scanner orchestrator should:
- build valid actions (token_idx, mutation_id) for the current payload
- let the RL agent pick among those actions
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Dict, List, Optional

from payload.context import ContextAwareMutations, MutationContext
from payload.tokenizer import Token, TokenType, default_tokenizer


TokenMutationFn = Callable[[List[Token], int, random.Random], Optional[List[Token]]]


@dataclass(frozen=True)
class Mutation:
    id: str
    description: str
    fn: TokenMutationFn


class MutationRegistry:
    def __init__(self) -> None:
        self._muts: Dict[str, Mutation] = {}

    def register(self, mutation_id: str, description: str) -> Callable[[TokenMutationFn], TokenMutationFn]:
        def deco(fn: TokenMutationFn) -> TokenMutationFn:
            self._muts[mutation_id] = Mutation(id=mutation_id, description=description, fn=fn)
            return fn

        return deco

    def get(self, mutation_id: str) -> Mutation:
        if mutation_id not in self._muts:
            raise KeyError(f"Unknown mutation: {mutation_id}")
        return self._muts[mutation_id]

    def ids(self) -> List[str]:
        return list(self._muts.keys())


registry = MutationRegistry()


# -------------------------
# Token-level mutations
# -------------------------


@registry.register("randomize_case", "Randomize case of SQL keywords")
def m_randomize_case(tokens: List[Token], token_idx: int, rng: random.Random) -> Optional[List[Token]]:
    tok = tokens[token_idx]
    if tok.type != TokenType.KEYWORD:
        return None

    text = tok.text
    if len(text) <= 1:
        return None

    new_text = "".join(ch.upper() if rng.random() < 0.5 else ch.lower() for ch in text)
    if new_text == text:
        return None

    out = list(tokens)
    out[token_idx] = Token(text=new_text, type=tok.type, start=tok.start)
    return out


@registry.register("change_to_hex", "Convert string literal to hex (0x...) or number to hex")
def m_change_to_hex(tokens: List[Token], token_idx: int, rng: random.Random) -> Optional[List[Token]]:
    _ = rng
    tok = tokens[token_idx]

    if tok.type == TokenType.LITERAL_STRING:
        # Strip quotes and convert to hex bytes
        if len(tok.text) < 2:
            return None
        content = tok.text[1:-1]
        hex_str = "0x" + content.encode("utf-8", errors="ignore").hex()
        out = list(tokens)
        out[token_idx] = Token(text=hex_str, type=TokenType.LITERAL_NUMBER, start=tok.start)
        return out

    if tok.type == TokenType.LITERAL_NUMBER:
        s = tok.text.strip().lower()
        if s.startswith("0x"):
            return None
        try:
            hex_str = hex(int(float(s)))
        except ValueError:
            return None
        out = list(tokens)
        out[token_idx] = Token(text=hex_str, type=TokenType.LITERAL_NUMBER, start=tok.start)
        return out

    return None


@registry.register("add_comment_after", "Insert /**/ right after this token")
def m_add_comment_after(tokens: List[Token], token_idx: int, rng: random.Random) -> Optional[List[Token]]:
    _ = rng
    out = list(tokens)
    out.insert(token_idx + 1, Token(text="/**/", type=TokenType.COMMENT, start=out[token_idx].end))
    return out


@registry.register("wrap_with_comment", "Wrap token with /**/token/**/")
def m_wrap_with_comment(tokens: List[Token], token_idx: int, rng: random.Random) -> Optional[List[Token]]:
    _ = rng
    tok = tokens[token_idx]
    out = list(tokens)
    out[token_idx: token_idx + 1] = [
        Token(text="/**/", type=TokenType.COMMENT, start=tok.start),
        tok,
        Token(text="/**/", type=TokenType.COMMENT, start=tok.end),
    ]
    return out


@registry.register("add_whitespace", "Add whitespace around token")
def m_add_whitespace(tokens: List[Token], token_idx: int, rng: random.Random) -> Optional[List[Token]]:
    ws = rng.choice([" ", "%0a", "%09"])
    tok = tokens[token_idx]
    out = list(tokens)
    out[token_idx: token_idx + 1] = [
        Token(text=ws, type=TokenType.WHITESPACE, start=tok.start),
        tok,
        Token(text=ws, type=TokenType.WHITESPACE, start=tok.end),
    ]
    return out


@registry.register("add_sleep", "Replace OR/AND <cond> with OR/AND SLEEP(n)")
def m_add_sleep(tokens: List[Token], token_idx: int, rng: random.Random) -> Optional[List[Token]]:
    tok = tokens[token_idx]
    if tok.type != TokenType.KEYWORD:
        return None
    if tok.text.upper() not in {"OR", "AND"}:
        return None

    n = rng.choice([3, 5, 7])
    sleep = Token(text=f"SLEEP({n})", type=TokenType.KEYWORD, start=tok.end)

    # Insert SLEEP(n) after OR/AND
    out = list(tokens)
    out.insert(token_idx + 1, Token(text=" ", type=TokenType.WHITESPACE, start=tok.end))
    out.insert(token_idx + 2, sleep)
    return out


# -------------------------
# Public helpers
# -------------------------


def get_available_mutations(payload: str) -> Dict[int, List[str]]:
    """Return mapping token_idx -> list of valid mutation IDs for that token."""

    tokens = default_tokenizer.tokenize(payload)
    out: Dict[int, List[str]] = {}

    for i, tok in enumerate(tokens):
        ctx = MutationContext(
            current_token=tok,
            prev_token=tokens[i - 1] if i > 0 else None,
            next_token=tokens[i + 1] if i + 1 < len(tokens) else None,
            token_index=i,
            all_tokens=tokens,
        )
        valid = ContextAwareMutations.get_valid_mutations(ctx)
        # Only keep mutations that actually exist in registry
        valid = [m for m in valid if m in registry.ids()]
        if valid:
            out[i] = sorted(valid)

    return out


def apply_mutation(
    payload: str,
    mutation_id: str,
    token_idx: int,
    rng: Optional[random.Random] = None,
) -> Optional[str]:
    """Apply token-level mutation. Returns new payload or None if not applicable."""

    rng = rng or random.Random()

    tokens = default_tokenizer.tokenize(payload)
    if token_idx < 0 or token_idx >= len(tokens):
        return None

    mut = registry.get(mutation_id)
    new_tokens = mut.fn(tokens, token_idx, rng)
    if not new_tokens:
        return None

    # Reconstruct payload from tokens
    return "".join(t.text for t in new_tokens)
