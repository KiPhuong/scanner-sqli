"""rl.rnd

Random Network Distillation (RND) for exploration in RL.

This module implements a lightweight RND (Random Network Distillation) to compute
intrinsic rewards based on state novelty. The reward is normalized using
exponentially weighted moving statistics (mean/std) for stable training.

Key features:
- Fixed random target network (no training)
- Trainable predictor network (SGD-updated)
- Normalized intrinsic rewards (z-score with online stats)
- No external dependencies (pure Python)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import List, Optional, Tuple


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _l2(v: List[float]) -> float:
    return sum(x * x for x in v) ** 0.5


def _matvec(mat: List[List[float]], vec: List[float]) -> List[float]:
    return [_dot(row, vec) for row in mat]


def _sub(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]


def _update_stats(
    current: Tuple[float, float, float],  # (count, mean, M2)
    new_value: float,
) -> Tuple[float, float, float]:
    """Update running mean and variance using Welford's online algorithm."""
    count, mean, M2 = current
    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    M2 += delta * delta2
    return (count, mean, M2)


def _get_mean_std(
    stats: Tuple[float, float, float],
    ddof: int = 1,
    eps: float = 1e-8,
) -> Tuple[float, float]:
    """Get mean and standard deviation from running stats."""
    count, mean, M2 = stats
    if count < 2:
        return mean, 1.0  # Avoid division by zero
    variance = M2 / (count - ddof)
    return mean, math.sqrt(max(0.0, variance) + eps)


@dataclass
class RND:
    input_dim: int
    output_dim: int = 32
    lr: float = 1e-3
    reward_clip: Optional[float] = 5.0  # Clip normalized reward to [-clip, clip]
    eps: float = 1e-8  # Small constant for numerical stability
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        # Initialize networks
        self.target = self._rand_matrix(self.output_dim, self.input_dim)
        self.predictor = self._rand_matrix(self.output_dim, self.input_dim)
        # Online stats: (count, mean, M2) for Welford's algorithm
        self._stats: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def _rand_matrix(self, rows: int, cols: int) -> List[List[float]]:
        # Small random values to avoid instability
        return [
            [self._rng.uniform(-0.1, 0.1) for _ in range(cols)]
            for _ in range(rows)
        ]

    def _compute_mse(self, state: List[float]) -> float:
        """Compute mean squared error between target and predictor outputs."""
        t = _matvec(self.target, state)
        p = _matvec(self.predictor, state)
        err = _sub(p, t)
        # Mean squared error
        return sum(e * e for e in err) / max(1, len(err))

    def intrinsic_reward(self, state: List[float]) -> float:
        """Compute normalized intrinsic reward for the given state.

        The reward is the MSE between target and predictor outputs,
        normalized using running statistics and clipped for stability.
        """
        mse = self._compute_mse(state)
        mean, std = _get_mean_std(self._stats, eps=self.eps)
        
        # Avoid division by zero in early stages
        if std < self.eps:
            return 0.0
            
        # Z-score normalization
        norm_reward = (mse - mean) / std
        
        # Clip to avoid extreme values
        if self.reward_clip is not None:
            norm_reward = max(-self.reward_clip, min(norm_reward, self.reward_clip))
            
        # Shift to be non-negative (optional, but common in practice)
        return max(0.0, norm_reward)

    def update(self, state: List[float]) -> float:
        """Update predictor with a single SGD step and return normalized reward.

        Args:
            state: Input state vector

        Returns:
            Normalized intrinsic reward (before the update)
        """
        # Compute prediction error
        t = _matvec(self.target, state)
        p = _matvec(self.predictor, state)
        err = _sub(p, t)
        mse = sum(e * e for e in err) / max(1, len(err))

        # Update running statistics (before predictor update)
        self._stats = _update_stats(self._stats, mse)
        
        # Compute normalized reward using current stats
        mean, std = _get_mean_std(self._stats, eps=self.eps)
        norm_reward = (mse - mean) / std if std > self.eps else 0.0
        if self.reward_clip is not None:
            norm_reward = max(-self.reward_clip, min(norm_reward, self.reward_clip))
        norm_reward = max(0.0, norm_reward)  # Ensure non-negative

        # Update predictor using SGD
        x = state
        for i in range(self.output_dim):
            ei = err[i]
            row = self.predictor[i]
            for j in range(self.input_dim):
                row[j] -= self.lr * (2.0 * ei * x[j])

        return float(norm_reward)