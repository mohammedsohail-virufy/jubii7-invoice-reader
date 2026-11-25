"""Recompute the totals chain from the line items (the single source of truth)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from jubii7.invoice.models import Invoice


@dataclass(frozen=True, slots=True)
class CorrectedTotals:
    subtotals: tuple[float, ...]
    total_before_discount: float
    discount: float
    taxable_amount: float
    vat: float
    grand_total: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def recompute(invoice: Invoice) -> CorrectedTotals:
    """lines -> subtotals -> total -> discount -> taxable -> VAT on taxable -> grand total."""
    subtotals = tuple(group.expected_subtotal for group in invoice.groups)
    total = round(sum(subtotals), 2)

    t = invoice.totals
    discount_base = subtotals[t.discount_group_index] if t.discount_group_index >= 0 else 0.0
    discount = round(discount_base * t.discount_rate, 2)

    taxable = round(total - discount, 2)
    vat = round(taxable * t.vat_rate, 2)
    return CorrectedTotals(
        subtotals=subtotals,
        total_before_discount=total,
        discount=discount,
        taxable_amount=taxable,
        vat=vat,
        grand_total=round(taxable + vat, 2),
    )
