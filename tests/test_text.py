from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from jubii7.text import has_arabic_digits, preprocess, to_number, to_western

ARABIC = "٠١٢٣٤٥٦٧٨٩"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("٢٤ رول", "24 رول"),
        ("NK/PO/٢٠٢٦/٣٨٩", "NK/PO/2026/389"),
        ("۱۲۳", "123"),
        ("١٢٫٥", "12.5"),
        ("abc", "abc"),
    ],
)
def test_to_western(raw: str, expected: str) -> None:
    assert to_western(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1,650.00", 1650.0), ("(155.20)", -155.2), ("١,٥٠٠", 1500.0), ("٢٤", 24.0), (3, 3.0), (" 7 ", 7.0)],
)
def test_to_number(raw: str | float, expected: float) -> None:
    assert to_number(raw) == expected


def test_to_number_rejects_text() -> None:
    with pytest.raises(ValueError, match="could not convert"):
        to_number("twelve")


def test_preprocess_lowercases_and_collapses_spaces() -> None:
    assert preprocess("  Rebar   ١٦ MM ") == "rebar 16 mm"


# ---- property-based tests: hold for any input, not just the examples above ----


@given(st.text())
def test_to_western_is_idempotent(text: str) -> None:
    once = to_western(text)
    assert to_western(once) == once


@given(st.text())
def test_to_western_removes_all_arabic_digits(text: str) -> None:
    assert not has_arabic_digits(to_western(text))


@given(st.integers(min_value=0, max_value=10**9))
def test_arabic_and_western_numbers_parse_the_same(n: int) -> None:
    arabic = str(n).translate(str.maketrans("0123456789", ARABIC))
    assert to_number(arabic) == to_number(str(n)) == float(n)


@given(st.decimals(min_value=0, max_value=10**7, places=2, allow_nan=False))
def test_brackets_mean_negative(value: object) -> None:
    text = f"{value:,.2f}"
    assert to_number(f"({text})") == -to_number(text)
