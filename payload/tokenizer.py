"""payload.tokenizer

A lightweight, regex-based tokenizer for SQLi payloads.

Its purpose is not to be a full SQL parser, but to provide local context
(e.g., "this is a keyword", "this is a string literal") to enable
context-aware mutations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import re
from typing import List


class TokenType(Enum):
    KEYWORD = auto()
    LITERAL_STRING = auto()
    LITERAL_NUMBER = auto()
    OPERATOR = auto()
    WHITESPACE = auto()
    COMMENT = auto()
    PUNCTUATION = auto()
    IDENTIFIER = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class Token:
    text: str
    type: TokenType
    start: int

    @property
    def end(self) -> int:
        return self.start + len(self.text)


# Regex patterns for tokenization. Order matters.
_PATTERNS = [
    (r"\b(SELECT|UNION|AND|OR|SLEEP|BENCHMARK|WAITFOR|IF|CASE|WHEN|THEN|ELSE|END|LIKE|IN|BETWEEN|NOT|NULL|TRUE|FALSE|XOR|DIV|MOD|RLIKE|REGEXP|FROM|WHERE|ORDER|BY|GROUP|HAVING|LIMIT|INSERT|UPDATE|DELETE|SET)\b", TokenType.KEYWORD),
    (r"'[^'\\]*(?:\\.[^'\\]*)*'", TokenType.LITERAL_STRING),
    (r'"[^"\\]*(?:\\.[^"\\]*)*"', TokenType.LITERAL_STRING),
    (r"--[^\\n]*", TokenType.COMMENT),
    (r"#.*", TokenType.COMMENT),
    (r"/\*.*?\*/", TokenType.COMMENT),
    (r"0x[0-9a-fA-F]+", TokenType.LITERAL_NUMBER),
    (r"\b\d+\.?\d*\b", TokenType.LITERAL_NUMBER),
    (r"[=<>!~]+|\|\||&&", TokenType.OPERATOR),
    (r"[+\-*/%&|^]", TokenType.OPERATOR),
    (r"[a-zA-Z_][a-zA-Z0-9_]*", TokenType.IDENTIFIER),
    (r"[(),;]", TokenType.PUNCTUATION),
    (r"\s+", TokenType.WHITESPACE),
]

_COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), t) for p, t in _PATTERNS]


class SQLiTokenizer:
    def tokenize(self, payload: str) -> List[Token]:
        tokens: List[Token] = []
        pos = 0
        n = len(payload)

        while pos < n:
            matched = False
            for regex, token_type in _COMPILED_PATTERNS:
                match = regex.match(payload, pos)
                if match:
                    text = match.group(0)
                    tokens.append(Token(text, token_type, pos))
                    pos = match.end()
                    matched = True
                    break

            if not matched:
                # No pattern matched, treat as unknown single character
                tokens.append(Token(payload[pos], TokenType.UNKNOWN, pos))
                pos += 1

        return tokens


# Singleton instance for convenience
default_tokenizer = SQLiTokenizer()
