"""core.evaluator

Evaluator compares a fuzzed response against a baseline and produces:
- a set of signals (deltas, similarity, error/block flags)
- a scalar reward suitable for RL

No database access is assumed; detection is heuristic/regex-based.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Optional

from utils.similarity import cosine_similarity


# Common SQL error patterns across engines (MySQL, PostgreSQL, MSSQL, Oracle, SQLite)
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
    """Compute deltas/similarity/error/blocking and derive a reward."""

    def __init__(
        self,
        sql_error_regex: Optional[re.Pattern[str]] = None,
        block_keyword_list: Optional[list[str]] = None,
        time_trigger_threshold: float = 2.0,
        semantic_change_threshold: float = 0.90,
        # reward weights
        w_time: float = 2.0,
        w_semantic: float = 1.0,
        w_sql_error: float = 0.75,
        penalty_blocked: float = 3.0,
        penalty_no_effect: float = 0.25,
    ):
        self.sql_error_regex = sql_error_regex or re.compile(
            "|".join(f"(?:{p})" for p in _SQL_ERROR_PATTERNS), re.IGNORECASE
        )
        self.block_keywords = [s.lower() for s in (block_keyword_list or _BLOCK_KEYWORDS)]

        self.time_trigger_threshold = time_trigger_threshold
        self.semantic_change_threshold = semantic_change_threshold

        self.w_time = w_time
        self.w_semantic = w_semantic
        self.w_sql_error = w_sql_error
        self.penalty_blocked = penalty_blocked
        self.penalty_no_effect = penalty_no_effect

    def evaluate(self, baseline, response) -> EvaluationResult:
        """Evaluate a fuzz response against baseline.

        baseline is expected to be core.baseline.BaselineResponse-like.
        response is expected to be core.requester.FuzzResponse-like.
        """

        time_delta = float(response.response_time - baseline.elapsed)
        length_delta = int(response.response_length - baseline.length)

        # Semantic similarity: cosine between baseline embedding and an embedding
        # for response body. If baseline embedding is unavailable, fall back.
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

    # -------------------------
    # Signals
    # -------------------------

    def _has_sql_error(self, text: str) -> bool:
        if not text:
            return False
        return self.sql_error_regex.search(text) is not None

    def _is_blocked(self, response) -> tuple[bool, Optional[str]]:
        # Hard block by status
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
            # Without baseline embedding, similarity can't be computed reliably.
            return 1.0

        # Reuse the same lightweight embedding function from baseline if present
        # (duck-typing). Otherwise fall back to a trivial heuristic.
        embed_fn = getattr(baseline.__class__, "_simple_embedding", None)
        if callable(embed_fn):
            resp_emb = embed_fn(response.response_text)  # type: ignore[misc]
            return float(cosine_similarity(base_emb, resp_emb))

        # Fallback: if we cannot embed, use length-based proxy (clamped)
        if baseline.length == 0 and response.response_length == 0:
            return 1.0
        denom = max(baseline.length, response.response_length, 1)
        diff = abs(response.response_length - baseline.length) / denom
        return float(max(0.0, 1.0 - diff))

    # -------------------------
    # Reward
    # -------------------------

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

        # Time-based trigger: big positive if response got significantly slower
        if time_delta >= self.time_trigger_threshold:
            reward += self.w_time * min(1.0, time_delta / max(self.time_trigger_threshold, 1e-6))

        # Semantic change: reward when similarity drops below threshold
        # e.g., baseline 0.98 -> fuzzed 0.70 indicates content changes
        if semantic_similarity < self.semantic_change_threshold:
            reward += self.w_semantic * (self.semantic_change_threshold - semantic_similarity)

        # SQL error presence is a useful weak signal
        if sql_error:
            reward += self.w_sql_error

        # Penalize no effect: no time trigger, no semantic change, no sql error
        if reward == 0.0:
            reward -= self.penalty_no_effect

        return float(reward)

