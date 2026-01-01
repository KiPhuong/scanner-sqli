"""utils.similarity

Vector similarity helpers.
"""

from __future__ import annotations


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns value between -1 and 1. 1 is identical, 0 is orthogonal.
    Returns 0.0 if either vector has a zero norm.
    """

    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must have the same dimension")

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))

    norm_a = sum(v * v for v in vec_a) ** 0.5
    norm_b = sum(v * v for v in vec_b) ** 0.5

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)

