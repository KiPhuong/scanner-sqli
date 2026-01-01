"""core.detector

Aggregates evaluation evidence over multiple payload attempts to decide if an
injection point is vulnerable.

High-level algorithm (simple, explainable):
1. For every payload attempt we receive an EvaluationResult (from
   core.evaluator.Evaluator).
2. We classify each attempt into evidence types:
   - time_based  : time_delta >= min_time_delay
   - semantic_dev: semantic_similarity < semantic_threshold
   - sql_error   : sql_error flag True
   - blocked     : blocked flag True (not evidence, but useful for stats)
3. A verdict of **vulnerable** is reached when any of the following holds:
   a) ≥ `consistent_evidence` attempts show time_based (not blocked)
   b) ≥ `consistent_evidence` attempts show semantic_dev (not blocked)
   c) at least 1 attempt shows sql_error (not blocked)

`successful filter bypass` roughly means: at least one evidence-bearing attempt
was *not* blocked (e.g., status 200, no captcha), indicating the payload
reached the backend.

Outputs:
- vulnerable (bool)
- trigger_payloads (list[str]) – payloads that contributed to detection
- request_count (int)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from core.evaluator import EvaluationResult


@dataclass
class Attempt:
    payload: str
    eval: EvaluationResult


@dataclass
class DetectionResult:
    vulnerable: bool
    trigger_payloads: List[str]
    request_count: int
    evidence_summary: dict[str, int] = field(default_factory=dict)


class SQLiDetector:
    def __init__(
        self,
        consistent_evidence: int = 2,
        min_time_delay: float = 2.0,
        semantic_threshold: float = 0.90,
    ) -> None:
        self.consistent_evidence = consistent_evidence
        self.min_time_delay = min_time_delay
        self.semantic_threshold = semantic_threshold

        self._attempts: List[Attempt] = []

    # -------------------------
    # Public API
    # -------------------------

    def record_attempt(self, payload: str, evaluation: EvaluationResult) -> None:
        """Store evaluation for a payload attempt."""
        self._attempts.append(Attempt(payload=payload, eval=evaluation))

    def verdict(self) -> DetectionResult:
        """Compute detection verdict based on accumulated evidence."""

        time_hits: List[Attempt] = []
        semantic_hits: List[Attempt] = []
        sql_error_hits: List[Attempt] = []

        for att in self._attempts:
            ev = att.eval
            if ev.blocked:
                # evidence only valid if not blocked
                continue

            if ev.time_delta >= self.min_time_delay:
                time_hits.append(att)
            if ev.semantic_similarity < self.semantic_threshold:
                semantic_hits.append(att)
            if ev.sql_error:
                sql_error_hits.append(att)

        evidence_counts = {
            "time_based": len(time_hits),
            "semantic_dev": len(semantic_hits),
            "sql_error": len(sql_error_hits),
        }

        vulnerable = False
        trigger_payloads: List[str] = []

        # Rule a) time-based
        if len(time_hits) >= self.consistent_evidence:
            vulnerable = True
            trigger_payloads.extend(a.payload for a in time_hits)

        # Rule b) semantic deviation
        if not vulnerable and len(semantic_hits) >= self.consistent_evidence:
            vulnerable = True
            trigger_payloads.extend(a.payload for a in semantic_hits)

        # Rule c) sql_error (single suffices)
        if not vulnerable and sql_error_hits:
            vulnerable = True
            trigger_payloads.extend(a.payload for a in sql_error_hits)

        # Deduplicate payload list while preserving order
        seen = set()
        trigger_payloads = [p for p in trigger_payloads if not (p in seen or seen.add(p))]

        return DetectionResult(
            vulnerable=vulnerable,
            trigger_payloads=trigger_payloads,
            request_count=len(self._attempts),
            evidence_summary=evidence_counts,
        )

    # Convenience – resets detector for new injection point
    def reset(self) -> None:
        self._attempts.clear()

