"""Typed domain model of an invoice.

Raw values are kept exactly as printed (strings, possibly with Arabic digits), so a
check can compare what the document says against what it should say. Parsed numeric
views are exposed as properties, so no check has to repeat parsing logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from jubii7.text import to_number


class Severity(StrEnum):
    """How serious an issue is. ERROR blocks posting; WARN and INFO need a look."""

    ERROR = "ERROR"
    WARN = "WARN"
    INFO = "INFO"


@dataclass(frozen=True, slots=True)
class Issue:
    """One finding produced by a check."""

    severity: Severity
    field: str
    message: str
    check: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity.value,
            "field": self.field,
            "issue": self.message,
            "check": self.check,
        }


@dataclass(frozen=True, slots=True)
class LineItem:
    no: int
    desc_en: str
    desc_ar: str
    unit: str
    ordered_raw: str
    delivered_raw: str
    unit_price_raw: str
    amount_raw: str

    @property
    def ordered(self) -> float:
        return to_number(self.ordered_raw)

    @property
    def delivered(self) -> float:
        return to_number(self.delivered_raw)

    @property
    def unit_price(self) -> float:
        return to_number(self.unit_price_raw)

    @property
    def printed_amount(self) -> float:
        return to_number(self.amount_raw)

    @property
    def expected_amount(self) -> float:
        """Billed on delivered quantity."""
        return round(self.delivered * self.unit_price, 2)


@dataclass(frozen=True, slots=True)
class LineGroup:
    name: str
    lines: tuple[LineItem, ...]
    subtotal_raw: str

    @property
    def printed_subtotal(self) -> float:
        return to_number(self.subtotal_raw)

    @property
    def expected_subtotal(self) -> float:
        return round(sum(line.expected_amount for line in self.lines), 2)


@dataclass(frozen=True, slots=True)
class Totals:
    total_before_discount_raw: str
    discount_raw: str
    discount_rate: float
    discount_group_index: int
    taxable_amount_raw: str
    vat_rate: float
    vat_raw: str
    grand_total_raw: str


@dataclass(frozen=True, slots=True)
class Delivery:
    id: str
    items: tuple[int, ...]
    planned_date: str
    received_by: str | None
    handwritten_note: str | None = None


@dataclass(frozen=True, slots=True)
class Invoice:
    invoice_no: str
    date: str
    supplier_trn: str
    customer_trn: str
    payment: dict[str, bool]
    groups: tuple[LineGroup, ...]
    totals: Totals
    amount_in_words_value: str
    deliveries: tuple[Delivery, ...]
    terms_en: str
    terms_days_en: str
    terms_days_ar: str
    delivery_to_ar: str
    customer_signature: str | None
    raw: dict[str, object] = field(default_factory=dict, repr=False, compare=False)

    @property
    def lines(self) -> tuple[LineItem, ...]:
        return tuple(line for group in self.groups for line in group.lines)
