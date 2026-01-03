"""rl.policy

A minimal epsilon-greedy policy over a discrete action space.

Actions are discrete ids (ints) from a fixed action space.
"""

from __future__ import annotations

import random
from typing import List, Optional

Action = int


class EpsilonGreedyPolicy:
    def __init__(self, epsilon: float = 0.2, seed: Optional[int] = None) -> None:
        self.epsilon = float(epsilon)
        self._rng = random.Random(seed)

    def select(self, q_values: List[float]) -> int:
        """Select an action index given Q-values."""
        if not q_values:
            raise ValueError("q_values is empty")
        if self._rng.random() < self.epsilon:
            return self._rng.randrange(0, len(q_values))
        # greedy
        best_i = 0
        best_v = q_values[0]
        for i, v in enumerate(q_values):
            if v > best_v:
                best_v = v
                best_i = i
        return best_i

