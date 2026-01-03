"""core.detector

Aggregates evaluation evidence over multiple payload attempts to decide if an
injection point is vulnerable.

To reduce false positives, the detection logic is conservative:
- SQL Error: 1 hit is enough (strong signal).
- Time-based: Requires `consistent_evidence` hits (e.g., 3) to confirm.
- Semantic Deviation: Requires `consistent_evidence` hits where the similarity
  drops below a *strict* threshold (`semantic_trigger_threshold`).
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
        consistent_evidence: int = 3,  # Increased from 2 to 3
        min_time_delay: float = 2.0,
        semantic_trigger_threshold: float = 0.70,  # Stricter threshold for semantic hits
    ) -> None:
        self.consistent_evidence = consistent_evidence
        self.min_time_delay = min_time_delay
        self.semantic_trigger_threshold = semantic_trigger_threshold
        self._attempts: List[Attempt] = []

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
                continue

            if ev.time_delta >= self.min_time_delay:
                time_hits.append(att)
            if ev.semantic_similarity < self.semantic_trigger_threshold:
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

        # Priority of evidence: SQL Error > Time-based > Semantic
        # Rule 1: SQL error (strongest signal, 1 hit is enough)
        if sql_error_hits:
            vulnerable = True
            trigger_payloads.extend(a.payload for a in sql_error_hits)

        # Rule 2: Time-based (strong signal, requires consistency)
        if not vulnerable and len(time_hits) >= self.consistent_evidence:
            vulnerable = True
            trigger_payloads.extend(a.payload for a in time_hits)

        # Rule 3: Semantic deviation (weaker signal, requires consistency)
        if not vulnerable and len(semantic_hits) >= self.consistent_evidence:
            vulnerable = True
            trigger_payloads.extend(a.payload for a in semantic_hits)

        # Deduplicate payload list while preserving order
        seen = set()
        trigger_payloads = [p for p in trigger_payloads if not (p in seen or seen.add(p))]

        return DetectionResult(
            vulnerable=vulnerable,
            trigger_payloads=trigger_payloads,
            request_count=len(self._attempts),
            evidence_summary=evidence_counts,
        )

    def reset(self) -> None:
        self._attempts.clear()
