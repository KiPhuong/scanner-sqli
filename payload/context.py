"""payload.context

Context-aware mutation filtering.

This module maps token types (from payload.tokenizer) to mutation IDs that are
reasonable to apply.

Note: payload.tokenizer.TokenType does NOT have LOGIC_OP. Logical operators like
AND/OR are represented as TokenType.KEYWORD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from payload.tokenizer import Token, TokenType


@dataclass
class MutationContext:
    current_token: Token
    prev_token: Optional[Token] = None
    next_token: Optional[Token] = None
    token_index: int = -1
    all_tokens: Optional[List[Token]] = None

    def is_in_string(self) -> bool:
        return self.current_token.type == TokenType.LITERAL_STRING

    def is_in_comment(self) -> bool:
        return self.current_token.type == TokenType.COMMENT

    def is_keyword(self, keyword: Optional[str] = None) -> bool:
        if self.current_token.type != TokenType.KEYWORD:
            return False
        if keyword is None:
            return True
        return self.current_token.text.upper() == keyword.upper()


class ContextAwareMutations:
    """Return valid mutation IDs for a given token context."""

    TOKEN_TYPE_TO_MUTATIONS: Dict[TokenType, Set[str]] = {
        TokenType.KEYWORD: {
            "randomize_case",
            "change_to_hex",
            "add_comment_after",
            "wrap_with_comment",
            "add_whitespace",
            "add_sleep",  # will be filtered further below for OR/AND
        },
        TokenType.LITERAL_STRING: {
            "change_to_hex",
            "add_comment_after",
            "wrap_with_comment",
            "add_whitespace",
        },
        TokenType.LITERAL_NUMBER: {
            "change_to_hex",
            "add_comment_after",
            "wrap_with_comment",
            "add_whitespace",
        },
        TokenType.OPERATOR: {
            "add_comment_after",
            "wrap_with_comment",
            "add_whitespace",
        },
        TokenType.PUNCTUATION: {
            "add_comment_after",
            "add_whitespace",
        },
        TokenType.IDENTIFIER: {
            "add_comment_after",
            "wrap_with_comment",
            "add_whitespace",
        },
        TokenType.UNKNOWN: {
            "add_comment_after",
            "add_whitespace",
        },
        TokenType.COMMENT: {
            "add_whitespace",
        },
        TokenType.WHITESPACE: {
            # allow changing whitespace around (we implement add_whitespace as an insertion)
            "add_whitespace",
        },
    }

    @classmethod
    def get_valid_mutations(cls, context: MutationContext) -> Set[str]:
        token_type = context.current_token.type
        muts = set(cls.TOKEN_TYPE_TO_MUTATIONS.get(token_type, set()))

        # Extra safety: if we're in a comment token, keep only very safe muts
        if context.is_in_comment():
            return {"add_whitespace"}

        # Only allow add_sleep when token is OR/AND keyword
        if "add_sleep" in muts:
            if not (context.is_keyword("OR") or context.is_keyword("AND")):
                muts.discard("add_sleep")

        return muts
