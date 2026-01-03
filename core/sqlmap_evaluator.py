"""core.sqlmap_evaluator

Reward shaping for sqlmap-driven RL.

Objective: find injectable parameter with as few HTTP requests as possible.
We don't have exact request counts, so we use runtime as a proxy cost.

Terminal:
- injectable => big positive reward
- episode end (max steps) => no terminal bonus

Penalties:
- blocked/WAF
- timeout
- runtime cost per step
"""

from __future__ import annotations

from dataclasses import dataclass

from core.sqlmap_parser import SqlmapObservation


@dataclass(frozen=True)
class RewardConfig:
    success_bonus: float = 100.0
    blocked_penalty: float = -10.0
    timeout_penalty: float = -20.0
    step_penalty: float = -1.0
    runtime_cost: float = -0.2  # per second


def compute_reward(obs: SqlmapObservation, *, duration_sec: float, cfg: RewardConfig) -> float:
    r = 0.0

    # base step cost to encourage fewer steps
    r += cfg.step_penalty

    # runtime proxy (less time ~= fewer requests)
    r += cfg.runtime_cost * float(duration_sec)

    if obs.blocked:
        r += cfg.blocked_penalty
    if obs.timeout:
        r += cfg.timeout_penalty

    if obs.injectable:
        r += cfg.success_bonus

    return float(r)

