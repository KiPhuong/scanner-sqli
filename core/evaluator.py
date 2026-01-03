"""core.evaluator

Evaluator compares a fuzzed response against a baseline and produces:
- a set of signals (deltas, similarity, error/block flags)
- a scalar reward suitable for RL

No database access is assumed; detection is heuristic/regex-based.

False-positive mitigation
- Semantic deltas are a *weak* signal. We only reward strong semantic deviation
  (similarity below `semantic_reward_threshold`).
- Time-based and SQL error signals remain the primary positive signals.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Optional

from utils.similarity import cosine_similarity


_SQL_ERROR_PATTERNS = [
    r"you have an error in your sql syntax",
    r"warning:\s*mysql",
    r"mysql_fetch",
    r"mysqli?_(query|fetch|num_rows|result)",
    r"unclosed quotation mark after the character string",
    r"quoted string not properly terminated",
    r"pg_.*error",
    r"postgresql.*error",
    r"psql:\s*fatal",
    r"syntax error at or near",
    r"sqlstate\[",
    r"odbc.*driver",
    r"microsoft ole db provider for sql server",
    r"sql server.*error",
    r"ora-\d{4,5}",
    r"sqlite\/jdbcdriver",
    r"sqlite error",
    r"near \".*\": syntax error",
]

_BLOCK_KEYWORDS = [
    "captcha",
    "verify you are human",
    "access denied",
    "request blocked",
    "unusual traffic",
    "bot detection",
    "security check",
    "cloudflare",
]


@dataclass(frozen=True)
class EvaluationResult:
    time_delta: float
    length_delta: int
    semantic_similarity: float

    sql_error: bool
    blocked: bool
    block_reason: Optional[str]

    reward: float
    signals: Dict[str, float]


class Evaluator:
    def __init__(
        self,
        sql_error_regex: Optional[re.Pattern[str]] = None,
        block_keyword_list: Optional[list[str]] = None,
        time_trigger_threshold: float = 2.0,
        semantic_reward_threshold: float = 0.70,
        # reward weights
        w_time: float = 2.0,
        w_semantic: float = 0.5,
        w_sql_error: float = 1.0,
        penalty_blocked: float = 3.0,
        penalty_no_effect: float = 0.25,
    ):
        self.sql_error_regex = sql_error_regex or re.compile(
            "|".join(f"(?:{p})" for p in _SQL_ERROR_PATTERNS), re.IGNORECASE
        )
        self.block_keywords = [s.lower() for s in (block_keyword_list or _BLOCK_KEYWORDS)]

        self.time_trigger_threshold = time_trigger_threshold
        self.semantic_reward_threshold = semantic_reward_threshold

        self.w_time = w_time
        self.w_semantic = w_semantic
        self.w_sql_error = w_sql_error
        self.penalty_blocked = penalty_blocked
        self.penalty_no_effect = penalty_no_effect

    def evaluate(self, baseline, response) -> EvaluationResult:
        time_delta = float(response.response_time - baseline.elapsed)
        length_delta = int(response.response_length - baseline.length)
        semantic_similarity = self._semantic_similarity(baseline, response)

        sql_error = self._has_sql_error(response.response_text)
        blocked, reason = self._is_blocked(response)

        reward = self._compute_reward(
            time_delta=time_delta,
            semantic_similarity=semantic_similarity,
            sql_error=sql_error,
            blocked=blocked,
        )

        signals: Dict[str, float] = {
            "time_delta": time_delta,
            "length_delta": float(length_delta),
            "semantic_similarity": semantic_similarity,
            "sql_error": 1.0 if sql_error else 0.0,
            "blocked": 1.0 if blocked else 0.0,
        }

        return EvaluationResult(
            time_delta=time_delta,
            length_delta=length_delta,
            semantic_similarity=semantic_similarity,
            sql_error=sql_error,
            blocked=blocked,
            block_reason=reason,
            reward=reward,
            signals=signals,
        )

    def _has_sql_error(self, text: str) -> bool:
        if not text:
            return False
        return self.sql_error_regex.search(text) is not None

    def _is_blocked(self, response) -> tuple[bool, Optional[str]]:
        if getattr(response, "status_code", None) in {401, 403, 429}:
            return True, f"status_{response.status_code}"

        txt = (response.response_text or "").lower()
        for kw in self.block_keywords:
            if kw in txt:
                return True, f"keyword:{kw}"
        return False, None

    def _semantic_similarity(self, baseline, response) -> float:
        base_emb = getattr(baseline, "embedding", None)
        if not base_emb:
            return 1.0

        embed_fn = getattr(baseline.__class__, "_simple_embedding", None)
        if callable(embed_fn):
            resp_emb = embed_fn(response.response_text)  # type: ignore[misc]
            return float(cosine_similarity(base_emb, resp_emb))

        if baseline.length == 0 and response.response_length == 0:
            return 1.0
        denom = max(baseline.length, response.response_length, 1)
        diff = abs(response.response_length - baseline.length) / denom
        return float(max(0.0, 1.0 - diff))

    def _compute_reward(
        self,
        time_delta: float,
        semantic_similarity: float,
        sql_error: bool,
        blocked: bool,
    ) -> float:
        if blocked:
            return -float(self.penalty_blocked)

        reward = 0.0

        # Strong signal 1: time-based
        if time_delta >= self.time_trigger_threshold:
            reward += self.w_time * min(1.0, time_delta / max(self.time_trigger_threshold, 1e-6))

        # Strong-ish signal 2: SQL error
        if sql_error:
            reward += self.w_sql_error

        # Weak signal: semantic deviation (reward only if deviation is strong)
        if semantic_similarity < self.semantic_reward_threshold:
            reward += self.w_semantic * (self.semantic_reward_threshold - semantic_similarity)

        if reward == 0.0:
            reward -= self.penalty_no_effect

        return float(reward)
