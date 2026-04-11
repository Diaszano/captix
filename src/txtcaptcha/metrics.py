"""Evaluation metrics."""
from __future__ import annotations

from typing import Sequence


def sequence_accuracy(preds: Sequence[str], targets: Sequence[str]) -> float:
    """Fraction of sequences where the full predicted string equals the target."""
    if len(preds) != len(targets):
        raise ValueError("preds and targets must have the same length")
    if not preds:
        return 0.0
    correct = sum(1 for p, t in zip(preds, targets) if p == t)
    return correct / len(preds)
