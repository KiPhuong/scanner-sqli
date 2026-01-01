"""payload.encoder

PayloadEncoder

Encodes payload strings into fixed-size numeric vectors for the RL agent.

Constraints:
- Payloads are treated as atomic strings (no token-level generation).
- Must be deterministic and dependency-free.

Implementation:
- Bag-of-words over whitespace-delimited tokens
- Deterministic FNV-1a hashing into a fixed-size vector
- L2 normalization

This is intentionally simple; you can swap it later with a real embedding model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class PayloadEncoder:
    dim: int = 64

    def encode(self, payload: str) -> List[float]:
        vec = [0.0] * self.dim
        for tok in payload.split():
            h = 2166136261
            for ch in tok:
                h ^= ord(ch)
                h = (h * 16777619) & 0xFFFFFFFF
            vec[h % self.dim] += 1.0

        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

