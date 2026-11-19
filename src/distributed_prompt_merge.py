"""A deterministic, read-only merge of distributed prompt-search observations.

This demonstrates an elementary join operation for a FIXED deterministic score.
It is not a distributed optimizer, network protocol, robotic system or LLM run.
"""
from __future__ import annotations

import math
from typing import TypeAlias

Observation: TypeAlias = tuple[float, tuple[str, ...]]


def _validate(item: Observation | None) -> None:
    if item is None:
        return
    value, prompt = item
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("a reported score must be finite")
    if not isinstance(prompt, tuple) or not prompt or any(
        not isinstance(token, str) for token in prompt
    ):
        raise ValueError("a prompt must be a nonempty tuple of strings")


def merge_best(left: Observation | None, right: Observation | None) -> Observation | None:
    """Take the maximum (score, lexicographic prompt), with None as bottom.

    Fixed score, matching candidate representation and deterministic evaluation
    are preconditions. Duplicate/reordered updates cannot change the result.
    """
    _validate(left)
    _validate(right)
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def merge_all(updates: list[Observation | None]) -> Observation | None:
    best = None
    for item in updates:
        best = merge_best(best, item)
    return best
