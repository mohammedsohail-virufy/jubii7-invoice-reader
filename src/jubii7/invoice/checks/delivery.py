"""Delivery and sign-off rules."""

from __future__ import annotations

from datetime import date

from jubii7.invoice.checks.base import Check
from jubii7.invoice.models import Invoice, Issue, Severity
from jubii7.text import to_western


def parse_date(text: str) -> date:
    """'31/11/2026' -> date. Raises ValueError for impossible dates."""
    day, month, year = (int(part) for part in to_western(text).split("/"))
    return date(year, month, day)


class DeliveryDateCheck(Check):
    """Every delivery date must be a real calendar date."""

    name = "delivery_date"

    def run(self, invoice: Invoice) -> list[Issue]:
        issues = []
        for delivery in invoice.deliveries:
            try:
                parse_date(delivery.planned_date)
            except ValueError:
                issues.append(
                    self.issue(
                        Severity.ERROR, f"delivery {delivery.id}", f"invalid date {delivery.planned_date!r}"
                    )
                )
        return issues


class SignatureCheck(Check):
    """Each delivery needs a receiver, and the invoice needs the customer's signature."""

    name = "signatures"

    def run(self, invoice: Invoice) -> list[Issue]:
        issues = [
            self.issue(
                Severity.WARN,
                f"delivery {d.id}",
                f"no receiver signature; note: {d.handwritten_note!r}",
            )
            for d in invoice.deliveries
            if not d.received_by
        ]
        if not invoice.customer_signature:
            issues.append(self.issue(Severity.WARN, "customer_signature", "missing: no proof of acceptance"))
        return issues


class DeliveryChargeContextCheck(Check):
    """Context, not an error: delivery outside Sharjah justifies a delivery charge."""

    name = "delivery_charge_context"

    def run(self, invoice: Invoice) -> list[Issue]:
        if "عجمان" in invoice.delivery_to_ar and "sharjah only" in invoice.terms_en.lower():
            return [
                self.issue(
                    Severity.INFO,
                    "delivery_to",
                    "delivery is to Ajman; prices include delivery inside Sharjah only, "
                    "so the delivery charge is justified",
                )
            ]
        return []
