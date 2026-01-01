"""utils.timing

Tiny timing helpers.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class TimerResult:
    elapsed: float


@contextmanager
def timer() -> Iterator[TimerResult]:
    start = time.perf_counter()
    res = TimerResult(elapsed=0.0)
    try:
        yield res
    finally:
        object.__setattr__(res, "elapsed", time.perf_counter() - start)

