from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from jubii7.puzzle.solver import solve
from jubii7.track_c.geometry import iou
from jubii7.track_c.metrics import scores


def test_puzzle_has_one_solution() -> None:
    solutions = solve()
    assert len(solutions) == 1
    people = {p.name: p for p in solutions[0]}
    assert people["Chen"].job == "chef"
    assert people["Dev"].emirate == "Abu Dhabi"


def test_iou_worked_example() -> None:
    assert iou((10, 10, 50, 50), (30, 30, 70, 70)) == pytest.approx(1 / 7)


def test_iou_edge_cases() -> None:
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0  # disjoint
    assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0  # touching edge
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0  # identical
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0  # degenerate


coords = st.lists(st.integers(min_value=0, max_value=1000), min_size=8, max_size=8)


@given(coords)
def test_iou_is_symmetric_and_bounded(c: list[int]) -> None:
    a = (min(c[0], c[2]), min(c[1], c[3]), max(c[0], c[2]), max(c[1], c[3]))
    b = (min(c[4], c[6]), min(c[5], c[7]), max(c[4], c[6]), max(c[5], c[7]))
    assert 0.0 <= iou(a, b) <= 1.0
    assert iou(a, b) == pytest.approx(iou(b, a))


def test_metrics_worked_example() -> None:
    s = scores(tp=90, fp=30, fn=10)
    assert (s.precision, s.recall) == (0.75, 0.9)
    assert s.f1 == pytest.approx(0.818, abs=1e-3)


def test_metrics_zero_division() -> None:
    assert scores(0, 0, 0).f1 == 0.0
