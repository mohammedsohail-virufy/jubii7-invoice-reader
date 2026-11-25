"""Precision, recall and F1 from counts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scores:
    precision: float
    recall: float
    f1: float


def scores(tp: int, fp: int, fn: int) -> Scores:
    """TP=90, FP=30, FN=10 -> precision 0.75, recall 0.90, F1 0.818."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Scores(precision, recall, f1)
