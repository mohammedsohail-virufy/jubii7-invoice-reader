"""Load the extracted invoice JSON into the typed domain model.

Validation happens here, at the boundary, so checks can trust their input.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jubii7.invoice.models import Delivery, Invoice, LineGroup, LineItem, Totals

_DISCOUNT_LABEL = re.compile(r"(?P<rate>\d+(?:\.\d+)?)\s*%\s*on\s+(?P<group>.+)", re.IGNORECASE)


class InvoiceFormatError(ValueError):
    """The JSON does not have the fields the checker needs."""


def load_invoice(path: str | Path) -> Invoice:
    """Read an invoice JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_invoice(data)


def parse_invoice(data: dict[str, Any]) -> Invoice:
    """Build an Invoice from a dict. Raises InvoiceFormatError on missing fields."""
    try:
        groups = tuple(_parse_group(g) for g in data["groups"])
        return Invoice(
            invoice_no=data["invoice_no"],
            date=data["date"],
            supplier_trn=data["supplier"]["trn"],
            customer_trn=data["customer_trn"],
            payment=dict(data["payment"]),
            groups=groups,
            totals=_parse_totals(data["totals"], groups),
            amount_in_words_value=data["amount_in_words_value"],
            deliveries=tuple(_parse_delivery(d) for d in data["deliveries"]),
            terms_en=data["terms_en"],
            terms_days_en=data["terms_days_en"],
            terms_days_ar=data["terms_days_ar"],
            delivery_to_ar=data.get("delivery_to_ar", ""),
            customer_signature=data.get("customer_signature"),
            raw=data,
        )
    except KeyError as missing:
        raise InvoiceFormatError(f"invoice JSON is missing field {missing}") from missing


def _parse_group(g: dict[str, Any]) -> LineGroup:
    lines = tuple(
        LineItem(
            no=int(ln["no"]),
            desc_en=ln["desc_en"],
            desc_ar=ln["desc_ar"],
            unit=ln["unit"],
            ordered_raw=str(ln["ordered"]),
            delivered_raw=str(ln["delivered"]),
            unit_price_raw=str(ln["unit_price"]),
            amount_raw=str(ln["amount"]),
        )
        for ln in g["lines"]
    )
    return LineGroup(name=g["name"], lines=lines, subtotal_raw=str(g["subtotal"]))


def _parse_totals(t: dict[str, Any], groups: tuple[LineGroup, ...]) -> Totals:
    rate, group_index = _parse_discount_label(t.get("discount_label", ""), groups)
    return Totals(
        total_before_discount_raw=str(t["total_before_discount"]),
        discount_raw=str(t["discount"]),
        discount_rate=rate,
        discount_group_index=group_index,
        taxable_amount_raw=str(t["taxable_amount"]),
        vat_rate=float(t["vat_rate"]),
        vat_raw=str(t["vat"]),
        grand_total_raw=str(t["grand_total"]),
    )


def _parse_discount_label(label: str, groups: tuple[LineGroup, ...]) -> tuple[float, int]:
    """'Discount 2% on Site B' -> (0.02, index of the group whose name starts with 'Site B')."""
    match = _DISCOUNT_LABEL.search(label)
    if not match:
        return 0.0, -1
    target = match["group"].strip().lower()
    for index, group in enumerate(groups):
        if group.name.lower().startswith(target):
            return float(match["rate"]) / 100, index
    raise InvoiceFormatError(f"discount label refers to unknown group {match['group']!r}")


def _parse_delivery(d: dict[str, Any]) -> Delivery:
    return Delivery(
        id=d["id"],
        items=tuple(int(i) for i in d.get("items", [])),
        planned_date=d["planned_date"],
        received_by=d.get("received_by"),
        handwritten_note=d.get("handwritten_note"),
    )
