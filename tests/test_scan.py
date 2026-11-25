from __future__ import annotations

import pytest

from jubii7.classifier.cascade import Cascade, RuleClassifier
from jubii7.scan import analyse, check_line_maths

# What OCR returns for the challenge invoice (Arabic digits kept as the scanner reads them).
OCR_TEXT = """AL WAHA BUILDING MATERIALS TRADING LLC   TRN 100000000000003
Tax Invoice No. WB-2026-11-0457   Date 14/11/2026
1 Portland cement 50 kg   120   120   18.50   2,220.00
2 Rebar 12 mm   4   4   2,750.00   11,000.00
3 Washed sand  رمل مغسول   ٢٤   ٢٤   65.00   1,650.00
5 Waterproofing membrane   20   18   145.00   2,610.00
Subtotal Site A 14,870.00
Taxable amount 22,474.80
VAT 5% 1,131.50
D3 delivery 31/11/2026
"""


@pytest.fixture
def report() -> dict:  # type: ignore[type-arg]
    return analyse(OCR_TEXT, Cascade([RuleClassifier()])).to_dict()


def test_finds_line_items_and_skips_totals(report: dict) -> None:  # type: ignore[type-arg]
    texts = [line["description"] for line in report["lines"]]
    assert len(texts) == 4
    assert not any("Subtotal" in t or "VAT" in t for t in texts)


def test_line_maths_catches_the_swapped_digits(report: dict) -> None:  # type: ignore[type-arg]
    maths = {line["description"].split()[0]: line["maths"] for line in report["lines"]}
    assert maths == {"Portland": "ok", "Rebar": "ok", "Washed": "mismatch", "Waterproofing": "ok"}


def test_arabic_digits_are_read(report: dict) -> None:  # type: ignore[type-arg]
    sand = next(line for line in report["lines"] if "sand" in line["description"])
    assert 24.0 in sand["numbers"]


def test_lines_are_classified(report: dict) -> None:  # type: ignore[type-arg]
    categories = {line["description"].split()[0]: line["category"] for line in report["lines"]}
    assert categories["Rebar"] == "steel"
    assert categories["Washed"] == "aggregates"


def test_dates_trn_and_vat(report: dict) -> None:  # type: ignore[type-arg]
    assert {"date": "31/11/2026", "valid": False} in report["dates"]
    assert report["trns"] == ["100000000000003"]
    checks = {issue["check"] for issue in report["issues"]}
    assert {"scan_line_maths", "scan_dates", "scan_vat"} <= checks


@pytest.mark.parametrize(
    ("numbers", "expected"),
    [
        ([120, 120, 18.5, 2220], "ok"),
        ([24, 24, 65, 1650], "mismatch"),
        ([1, 2], "n/a"),
        ([3, 24, 65, 1560], "ok"),
    ],
)
def test_check_line_maths(numbers: list[float], expected: str) -> None:
    assert check_line_maths(numbers)[0] == expected


def test_empty_and_noise_text() -> None:
    assert analyse("").to_dict()["summary"]["line_items"] == 0
    assert analyse("|||  ...  ###").to_dict()["lines"] == []
