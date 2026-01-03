"""rl.sqlmap_config

Mutable sqlmap configuration that is updated by atomic OptionAction(s).

The config is then rendered into argv options for a sqlmap run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from rl.sqlmap_option_space import OptionAction


@dataclass
class SqlmapConfig:
    technique: Optional[str] = None
    level: int = 1
    risk: int = 1
    delay: Optional[float] = None
    time_sec: Optional[int] = None

    # New knobs
    random_agent: bool = False
    timeout: Optional[int] = None
    retries: Optional[int] = None

    tampers: List[str] = field(default_factory=list)

    def reset(self) -> None:
        self.technique = None
        self.level = 1
        self.risk = 1
        self.delay = None
        self.time_sec = None
        self.random_agent = False
        self.timeout = None
        self.retries = None
        self.tampers = []


def apply_option(cfg: SqlmapConfig, act: OptionAction, *, max_tampers: int = 15) -> None:
    if act.kind == "reset":
        cfg.reset()
        return

    if act.kind == "technique":
        cfg.technique = act.value
        return

    if act.kind == "level":
        cfg.level = int(act.value or "1")
        return

    if act.kind == "risk":
        cfg.risk = int(act.value or "1")
        return

    if act.kind == "delay":
        cfg.delay = float(act.value) if act.value is not None else None
        return

    if act.kind == "time_sec":
        cfg.time_sec = int(act.value) if act.value is not None else None
        return

    if act.kind == "random_agent":
        cfg.random_agent = (act.value == "on")
        return

    if act.kind == "timeout":
        cfg.timeout = int(act.value) if act.value is not None else None
        return

    if act.kind == "retries":
        cfg.retries = int(act.value) if act.value is not None else None
        return

    if act.kind == "add_tamper":
        if act.value is None:
            return
        if act.value in cfg.tampers:
            return
        if len(cfg.tampers) >= int(max_tampers):
            return
        cfg.tampers.append(act.value)
        return

    if act.kind == "clear_tamper":
        cfg.tampers = []
        return


def to_sqlmap_args(cfg: SqlmapConfig) -> List[str]:
    args: List[str] = []

    if cfg.random_agent:
        args.append("--random-agent")

    if cfg.timeout is not None:
        args.extend(["--timeout", str(cfg.timeout)])

    if cfg.retries is not None:
        args.extend(["--retries", str(cfg.retries)])

    if cfg.technique:
        args.extend(["--technique", cfg.technique])

    args.extend(["--level", str(cfg.level)])
    args.extend(["--risk", str(cfg.risk)])

    if cfg.delay is not None:
        args.extend(["--delay", str(cfg.delay)])

    # If technique includes time-based, ensure time-sec exists
    if cfg.time_sec is not None:
        args.extend(["--time-sec", str(cfg.time_sec)])
    else:
        if cfg.technique and "T" in cfg.technique:
            args.extend(["--time-sec", "5"])  # safe default

    if cfg.tampers:
        args.extend(["--tamper", ",".join(cfg.tampers)])

    return args
