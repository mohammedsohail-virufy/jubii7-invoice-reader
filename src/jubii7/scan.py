"""Analyse raw OCR text from a scanned or photographed invoice.

OCR gives lines of text, not a structured invoice. This module does what can be done
reliably from text alone, and says so when it can't:

- finds line items (a description plus numbers) and checks qty x unit price = amount
- classifies each line with any LineClassifier (rules -> model cascade)
- validates dates (31/11 is not a date) and 15-digit UAE TRNs
- checks VAT = 5% of the taxable amount when both appear

Mapping every field into the full invoice JSON needs layout and table detection
(see docs/ARCHITECTURE.md); the structured checker then takes over.
Pure Python, so it runs on the server and in the browser (Pyodide).
"""

from __future__ import annotations

import itertools
import re
from dataclasses import asdict, dataclass, field
from datetime import date

from jubii7.classifier.cascade import LineClassifier
from jubii7.invoice.models import Issue, Severity
from jubii7.text import to_number, to_western

_NUMBER = re.compile(r"\(?\d[\d,]*(?:\.\d+)?\)?")
_DATE = re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b")
_TRN = re.compile(r"\b\d{15}\b")
_LETTERS = re.compile(r"[A-Za-z؀-ۿ]{3,}")
_TAXABLE = re.compile(r"taxable|خاضع", re.IGNORECASE)
_VAT = re.compile(r"\bvat\b|ضريبة", re.IGNORECASE)
_SKIP = re.compile(
    r"total|subtotal|vat|tax|discount|trn|invoice|date|المجموع|الإجمالي|ضريبة|خصم|فاتورة", re.IGNORECASE
)
TOLERANCE = 0.01
VAT_RATE = 0.05


@dataclass
class ScannedLine:
    text: str
    description: str
    numbers: list[float]
    maths: str  # "ok", "mismatch" or "n/a"
    maths_detail: str
    category: str = ""
    confidence: float = 0.0
    needs_review: bool = True
    source: str = ""


@dataclass
class ScanReport:
    lines: list[ScannedLine] = field(default_factory=list)
    dates: list[dict[str, object]] = field(default_factory=list)
    trns: list[str] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        errors = sum(1 for i in self.issues if i.severity is Severity.ERROR)
        return {
            "summary": {
                "line_items": len(self.lines),
                "maths_checked": sum(1 for line in self.lines if line.maths != "n/a"),
                "errors": errors,
                "needs_review": sum(1 for line in self.lines if line.needs_review),
            },
            "lines": [asdict(line) for line in self.lines],
            "dates": self.dates,
            "trns": self.trns,
            "issues": [i.to_dict() for i in self.issues],
        }


def _numbers(text: str) -> list[float]:
    values = []
    for raw in _NUMBER.findall(text):
        try:
            values.append(to_number(raw))
        except ValueError:
            continue
    return values


def check_line_maths(numbers: list[float]) -> tuple[str, str]:
    """Is there a qty x price = amount among the numbers (in reading order)? Amount = the largest."""
    if len(numbers) < 3:
        return "n/a", "fewer than 3 numbers"
    amount = max(numbers)
    rest = [n for n in numbers if n != amount] or numbers[:-1]
    for qty, price in itertools.combinations(rest, 2):
        if qty and price and abs(qty * price - amount) <= TOLERANCE:
            return "ok", f"{qty:g} x {price:,.2f} = {amount:,.2f}"
    return "mismatch", f"no quantity x price among {[f'{n:g}' for n in rest]} gives {amount:,.2f}"


def analyse(text: str, classifier: LineClassifier | None = None) -> ScanReport:
    report = ScanReport()
    for raw_line in to_western(text).splitlines():
        line = " ".join(raw_line.split())
        if not line:
            continue
        _collect_dates_and_trns(line, report)
        if _SKIP.search(line) or _DATE.search(line) or not _LETTERS.search(line):
            continue
        numbers = _numbers(line)
        if len(numbers) < 2:
            continue
        report.lines.append(_item(line, numbers, classifier, report))
    _check_vat(text, report)
    return report


def _item(
    line: str, numbers: list[float], classifier: LineClassifier | None, report: ScanReport
) -> ScannedLine:
    maths, detail = check_line_maths(numbers)
    description = " ".join(_NUMBER.sub(" ", line).split())
    item = ScannedLine(text=line, description=description, numbers=numbers, maths=maths, maths_detail=detail)
    if maths == "mismatch":
        report.issues.append(Issue(Severity.ERROR, line[:40], f"line maths: {detail}", "scan_line_maths"))
    if classifier and description:
        prediction = classifier.predict(description)
        item.category, item.confidence = prediction.category, prediction.confidence
        item.needs_review, item.source = prediction.needs_review, prediction.source
    return item


def _collect_dates_and_trns(line: str, report: ScanReport) -> None:
    for day, month, year in _DATE.findall(line):
        text = f"{day}/{month}/{year}"
        try:
            date(int(year), int(month), int(day))
            report.dates.append({"date": text, "valid": True})
        except ValueError:
            report.dates.append({"date": text, "valid": False})
            report.issues.append(Issue(Severity.ERROR, "date", f"{text} is not a real date", "scan_dates"))
    report.trns.extend(_TRN.findall(line))


def _check_vat(text: str, report: ScanReport) -> None:
    taxable = vat = None
    for line in to_western(text).splitlines():
        numbers = _numbers(line)
        if not numbers:
            continue
        if _TAXABLE.search(line):
            taxable = numbers[-1]
        elif _VAT.search(line) and not re.search(r"trn|incl", line, re.IGNORECASE):
            vat = numbers[-1]
    if taxable and vat and abs(round(taxable * VAT_RATE, 2) - vat) > TOLERANCE:
        report.issues.append(
            Issue(
                Severity.ERROR,
                "vat",
                f"5% of taxable {taxable:,.2f} = {taxable * VAT_RATE:,.2f}, scanned VAT {vat:,.2f}",
                "scan_vat",
            )
        )
