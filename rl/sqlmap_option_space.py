"""rl.sqlmap_option_space

Atomic (single-option) actions for sequential sqlmap configuration.

One RL step chooses ONE option action, which updates a persistent SqlmapConfig.
Then the orchestrator runs sqlmap once using the updated config.

Keep this action space relatively small to make learning feasible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class OptionAction:
    """Atomic action applied to config.

    kind:
      - technique
      - level
      - risk
      - delay
      - time_sec
      - add_tamper
      - clear_tamper
      - random_agent
      - timeout
      - retries
      - reset

    Notes:
      - Some options (e.g. timeout/retries) are also enforced by the subprocess runner.
        Here they represent *sqlmap-side* knobs.
    """

    kind: str
    value: Optional[str] = None


def default_option_actions(tampers: Sequence[str]) -> List[OptionAction]:
    actions: List[OptionAction] = []

    # Technique combos sqlmap accepts (subset)
    for t in ["B", "E", "T", "U", "BE", "BT", "ET", "BET"]:
        actions.append(OptionAction("technique", t))

    for lvl in ["1", "3", "5"]:
        actions.append(OptionAction("level", lvl))

    for risk in ["1", "2", "3"]:
        actions.append(OptionAction("risk", risk))

    for d in ["0", "0.5", "1"]:
        actions.append(OptionAction("delay", d))

    for ts in ["3", "5", "8", "10"]:
        actions.append(OptionAction("time_sec", ts))

    # New: random-agent toggle
    actions.append(OptionAction("random_agent", "on"))
    actions.append(OptionAction("random_agent", "off"))

    # New: sqlmap-side timeout/retries knobs
    for t in ["10", "20", "30", "60"]:
        actions.append(OptionAction("timeout", t))

    for r in ["0", "1", "2", "3"]:
        actions.append(OptionAction("retries", r))

    # Tampers as atomic add operations
    for t in tampers:
        actions.append(OptionAction("add_tamper", t))

    actions.append(OptionAction("clear_tamper", None))
    actions.append(OptionAction("reset", None))

    return actions
