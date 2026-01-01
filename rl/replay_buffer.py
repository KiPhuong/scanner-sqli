"""rl.replay_buffer

A minimal experience replay buffer.

Stores transitions:
(s, a, r, s', done)

Where:
- s and s' are flat float vectors
- a is a tuple (payload_id, mutation_id)

This buffer is framework-agnostic; the Agent can implement learning however it
wants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import random


Action = Tuple[int, str]


@dataclass(frozen=True)
class Transition:
    state: List[float]
    action: Action
    reward: float
    next_state: List[float]
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000, seed: Optional[int] = None) -> None:
        self.capacity = int(capacity)
        self._rng = random.Random(seed)
        self._buf: List[Transition] = []
        self._pos = 0

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, transition: Transition) -> None:
        if self.capacity <= 0:
            return
        if len(self._buf) < self.capacity:
            self._buf.append(transition)
        else:
            self._buf[self._pos] = transition
            self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        if batch_size <= 0:
            return []
        n = min(batch_size, len(self._buf))
        return self._rng.sample(self._buf, n)

