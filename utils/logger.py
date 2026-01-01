"""utils.logger

Logging & payload analysis utilities.

Goals
- Record all payloads used (including mutated payloads)
- Record successful payload-mutation pairs
- Compare used payloads with original dataset
- Provide analysis for:
  - payload diversity
  - mutation effectiveness
  - novel payload emergence

This is designed to be simple, dependency-free, and friendly for pentesting
workflows (CSV/JSON outputs).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class PayloadEvent:
    ts: float
    target_url: str
    injection_url: str
    method: str
    param: str
    payload_id: int
    mutation_id: str
    base_payload: str
    mutated_payload: str
    reward: float
    time_delta: float
    length_delta: float
    semantic_similarity: float
    sql_error: bool
    blocked: bool
    vulnerable: bool


class PayloadLogger:
    """Collects payload events in-memory and can flush to JSONL."""

    def __init__(self, out_path: str = "logs/payload_events.jsonl") -> None:
        self.out_path = out_path
        self._events: List[PayloadEvent] = []

        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    def log(self, event: PayloadEvent) -> None:
        self._events.append(event)

    def events(self) -> List[PayloadEvent]:
        return list(self._events)

    def flush(self) -> None:
        """Append events to jsonl file."""
        if not self._events:
            return
        with open(self.out_path, "a", encoding="utf-8") as f:
            for ev in self._events:
                f.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")
        self._events.clear()


class PayloadAnalyzer:
    """Analysis helpers for payload runs."""

    def __init__(self, original_payloads: List[str]):
        self.original_set: Set[str] = set(original_payloads)

    # ---------- Recording comparisons ----------

    def compare_to_dataset(self, used_payloads: List[str]) -> Dict[str, Any]:
        used_set = set(used_payloads)
        novel = sorted(list(used_set - self.original_set))
        reused = sorted(list(used_set & self.original_set))
        return {
            "used_count": len(used_payloads),
            "unique_used": len(used_set),
            "reused_from_dataset": len(reused),
            "novel_count": len(novel),
            "novel_payloads": novel,
        }

    # ---------- Diversity ----------

    def payload_diversity(self, used_payloads: List[str]) -> Dict[str, Any]:
        """Simple diversity metrics: uniqueness ratio + length stats."""
        if not used_payloads:
            return {"unique": 0, "total": 0, "uniq_ratio": 0.0, "avg_len": 0.0, "min_len": 0, "max_len": 0}

        total = len(used_payloads)
        unique = len(set(used_payloads))
        lens = [len(p) for p in used_payloads]
        return {
            "total": total,
            "unique": unique,
            "uniq_ratio": unique / total,
            "avg_len": sum(lens) / total,
            "min_len": min(lens),
            "max_len": max(lens),
        }

    # ---------- Mutation effectiveness ----------

    def mutation_effectiveness(self, events: List[PayloadEvent]) -> Dict[str, Any]:
        """Summarize mutation effectiveness based on reward & success flags."""
        by_mut: Dict[str, Dict[str, Any]] = {}
        for ev in events:
            m = by_mut.setdefault(
                ev.mutation_id,
                {"attempts": 0, "successes": 0, "avg_reward": 0.0, "blocked": 0},
            )
            m["attempts"] += 1
            if ev.vulnerable or ev.reward > 0:
                m["successes"] += 1
            if ev.blocked:
                m["blocked"] += 1

            # online average
            m["avg_reward"] += (ev.reward - m["avg_reward"]) / m["attempts"]

        return by_mut

    # ---------- Successful pairs ----------

    def successful_pairs(self, events: List[PayloadEvent]) -> List[Dict[str, Any]]:
        pairs = {}
        for ev in events:
            if not (ev.vulnerable or ev.reward > 0):
                continue
            key = (ev.payload_id, ev.mutation_id)
            cur = pairs.get(key)
            if cur is None or ev.reward > cur["best_reward"]:
                pairs[key] = {
                    "payload_id": ev.payload_id,
                    "mutation_id": ev.mutation_id,
                    "best_reward": ev.reward,
                    "example_payload": ev.mutated_payload,
                }
        return sorted(pairs.values(), key=lambda x: x["best_reward"], reverse=True)


def now_ts() -> float:
    return time.time()

