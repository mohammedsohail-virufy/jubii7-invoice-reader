"""Bounding-box intersection over union (IoU)."""

from __future__ import annotations

Box = tuple[float, float, float, float]  # (x1, y1, x2, y2), x2 > x1, y2 > y1


def area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two axis-aligned boxes. 0 when they don't overlap."""
    intersection = area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))
    union = area(a) + area(b) - intersection
    return intersection / union if union else 0.0
