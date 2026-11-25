"""Normalisation for mixed Arabic / Western text and numbers.

Pure functions with no I/O, shared by the checker, the classifier and the API.
"""

from __future__ import annotations

import re

# Arabic-Indic (U+0660-0669) and Eastern Arabic / Persian (U+06F0-06F9) digits.
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
# Arabic decimal and thousands separators.
_SEPARATORS = str.maketrans({"٫": ".", "٬": ","})
_ARABIC_DIGIT = re.compile(r"[٠-٩۰-۹]")


def to_western(text: str) -> str:
    """Convert Arabic digits and separators to Western ones: '٢٤ رول' -> '24 رول'."""
    return text.translate(_DIGITS).translate(_SEPARATORS)


def has_arabic_digits(text: str) -> bool:
    """True if the text contains any Arabic-Indic or Persian digit."""
    return bool(_ARABIC_DIGIT.search(text))


def to_number(value: str | float) -> float:
    """Parse an invoice amount: '1,650.00', '٢٤' or '(155.20)'. Brackets mean negative.

    Raises:
        ValueError: if the value is not a number.
    """
    if isinstance(value, int | float):
        return float(value)
    cleaned = to_western(value).strip().replace(",", "")
    negative = cleaned.startswith("(") and cleaned.endswith(")")
    number = float(cleaned.strip("()"))
    return -number if negative else number


def preprocess(text: str) -> str:
    """Model preprocessor: Western digits, lower case, single spaces."""
    return " ".join(to_western(text).lower().split())
