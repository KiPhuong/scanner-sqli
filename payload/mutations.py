"""payload.mutations

Payload-level mutations for SQLi fuzzing.

Each mutation:
- takes a full payload string
- returns a modified payload string

Non-goals:
- character-by-character generation

Mutation categories implemented:
- Keyword obfuscation (case randomization)
- Comment injection (/**/)
- Whitespace substitution (%0a, %09)
- Operator substitution (AND→&&, OR→||)
- Wrapper injection (quotes, parentheses)
- Time-based transformation (e.g., OR 1=1 → OR SLEEP(5))

Includes:
- a mutation registry
- apply_mutation_by_id
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import re
from typing import Callable, Dict, List, Optional


MutationFn = Callable[[str, random.Random], str]


@dataclass(frozen=True)
class Mutation:
    id: str
    name: str
    description: str
    fn: MutationFn


# -------------------------
# Helpers
# -------------------------

_SQL_KEYWORDS = [
    "SELECT",
    "UNION",
    "WHERE",
    "FROM",
    "AND",
    "OR",
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "SLEEP",
    "BENCHMARK",
    "WAITFOR",
    "DELAY",
    "ORDER",
    "GROUP",
    "BY",
    "HAVING",
]


def _randomize_case(s: str, rng: random.Random) -> str:
    out = []
    for ch in s:
        if ch.isalpha():
            out.append(ch.upper() if rng.random() < 0.5 else ch.lower())
        else:
            out.append(ch)
    return "".join(out)


def _replace_word_boundary(payload: str, word: str, repl: str) -> str:
    # (?i) for case-insensitive; \b for word boundary
    pattern = re.compile(rf"(?i)\b{re.escape(word)}\b")
    return pattern.sub(repl, payload)


def _choose_ws(rng: random.Random) -> str:
    return rng.choice(["%0a", "%09"])  # newline / tab


def _inject_comment_in_spaces(payload: str, rng: random.Random) -> str:
    # Replace some whitespace runs with /**/ (optionally combined with ws encodings)
    def repl(_match: re.Match) -> str:
        # Keep it readable but obfuscated
        choices = ["/**/", f"/**/{_choose_ws(rng)}", f"{_choose_ws(rng)}/**/"]
        return rng.choice(choices)

    return re.sub(r"\s+", repl, payload)


# -------------------------
# Mutations
# -------------------------


def m_keyword_case_randomization(payload: str, rng: random.Random) -> str:
    """Randomize case of common SQL keywords only (not the whole payload)."""

    out = payload
    for kw in _SQL_KEYWORDS:
        # Replace keyword occurrences with randomized-case variant.
        def _kw_repl(m: re.Match) -> str:
            return _randomize_case(m.group(0), rng)

        out = re.sub(rf"(?i)\b{re.escape(kw)}\b", _kw_repl, out)
    return out


def m_comment_injection(payload: str, rng: random.Random) -> str:
    """Inject /**/ into whitespace positions."""
    return _inject_comment_in_spaces(payload, rng)


def m_whitespace_substitution(payload: str, rng: random.Random) -> str:
    """Replace some spaces with %0a or %09."""

    # Replace spaces between non-space characters; keep leading/trailing as-is.
    def repl(_match: re.Match) -> str:
        return _choose_ws(rng)

    # only replace literal spaces (not all whitespace) to avoid exploding transformations
    return re.sub(r" +", repl, payload)


def m_operator_substitution(payload: str, rng: random.Random) -> str:
    """Substitute AND/OR operators."""

    out = payload

    # Replace AND with && sometimes
    if re.search(r"(?i)\bAND\b", out) and rng.random() < 0.9:
        out = _replace_word_boundary(out, "AND", "&&")

    # Replace OR with || sometimes
    if re.search(r"(?i)\bOR\b", out) and rng.random() < 0.9:
        out = _replace_word_boundary(out, "OR", "||")

    return out


def m_wrapper_injection(payload: str, rng: random.Random) -> str:
    """Wrap the entire payload with quotes/parentheses."""

    wrappers = [
        ("(", ")"),
        ("'", "'"),
        ('"', '"'),
        ("')(", ")('") if rng.random() < 0.5 else ("'(", ")'"),
    ]

    left, right = rng.choice(wrappers)
    return f"{left}{payload}{right}"


def m_time_based_transformation(payload: str, rng: random.Random) -> str:
    """Try to convert a boolean OR/AND condition into a time-based one.

    Examples:
    - OR 1=1  -> OR SLEEP(5)
    - OR '1'='1' -> OR SLEEP(5)

    If no simple pattern is found, append a time-based clause.
    """

    sleep_seconds = rng.choice([3, 5, 7])
    sleep_expr = f"SLEEP({sleep_seconds})"

    # Replace common always-true patterns when preceded by OR/AND
    patterns = [
        r"(?i)(\bOR\b)\s+1\s*=\s*1\b",
        r"(?i)(\bOR\b)\s+'1'\s*=\s*'1'\b",
        r"(?i)(\bOR\b)\s+\"1\"\s*=\s*\"1\"\b",
        r"(?i)(\bAND\b)\s+1\s*=\s*1\b",
        r"(?i)(\bAND\b)\s+'1'\s*=\s*'1'\b",
        r"(?i)(\bAND\b)\s+\"1\"\s*=\s*\"1\"\b",
    ]

    for pat in patterns:
        if re.search(pat, payload):
            return re.sub(pat, lambda m: f"{m.group(1)} {sleep_expr}", payload)

    # If no match: add a time-based clause in a conservative way
    if re.search(r"(?i)\bOR\b", payload):
        return re.sub(r"(?i)\bOR\b", lambda m: f"{m.group(0)} {sleep_expr} OR", payload, count=1).rstrip(" OR")
    if re.search(r"(?i)\bAND\b", payload):
        return re.sub(r"(?i)\bAND\b", lambda m: f"{m.group(0)} {sleep_expr} AND", payload, count=1).rstrip(" AND")

    # Fallback append
    joiner = " OR " if rng.random() < 0.5 else " AND "
    return f"{payload}{joiner}{sleep_expr}"


# -------------------------
# Registry
# -------------------------


MUTATIONS: List[Mutation] = [
    Mutation(
        id="kw_case",
        name="Keyword case randomization",
        description="Randomize case of common SQL keywords.",
        fn=m_keyword_case_randomization,
    ),
    Mutation(
        id="comment_inject",
        name="Comment injection",
        description="Replace whitespace with /**/ (and variants).",
        fn=m_comment_injection,
    ),
    Mutation(
        id="ws_sub",
        name="Whitespace substitution",
        description="Replace spaces with %0a or %09.",
        fn=m_whitespace_substitution,
    ),
    Mutation(
        id="op_sub",
        name="Operator substitution",
        description="Replace AND/OR with &&/||.",
        fn=m_operator_substitution,
    ),
    Mutation(
        id="wrap",
        name="Wrapper injection",
        description="Wrap payload with quotes/parentheses.",
        fn=m_wrapper_injection,
    ),
    Mutation(
        id="time",
        name="Time-based transformation",
        description="Transform boolean conditions into SLEEP(n) time-based clauses.",
        fn=m_time_based_transformation,
    ),
]

_MUTATION_BY_ID: Dict[str, Mutation] = {m.id: m for m in MUTATIONS}


def mutation_ids() -> List[str]:
    return [m.id for m in MUTATIONS]


def get_mutation(mutation_id: str) -> Mutation:
    try:
        return _MUTATION_BY_ID[mutation_id]
    except KeyError as e:
        raise KeyError(f"Unknown mutation id: {mutation_id}. Known: {mutation_ids()}") from e


def apply_mutation_by_id(
    payload: str,
    mutation_id: str,
    seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> str:
    """Apply a mutation by ID.

    Provide either seed or rng.
    """

    if rng is None:
        rng = random.Random(seed)

    mut = get_mutation(mutation_id)
    return mut.fn(payload, rng)

