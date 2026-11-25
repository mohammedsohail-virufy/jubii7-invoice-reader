"""Arithmetic rules: line maths, subtotals and the totals chain."""

from __future__ import annotations

from jubii7.invoice.checks.base import Check, differs
from jubii7.invoice.models import Invoice, Issue, Severity
from jubii7.invoice.totals import recompute
from jubii7.text import to_number


class LineAmountCheck(Check):
    """delivered x unit price must equal the printed amount."""

    name = "line_amount"

    def run(self, invoice: Invoice) -> list[Issue]:
        return [
            self.issue(
                Severity.ERROR,
                f"line {line.no}",
                f"{line.delivered:g} x {line.unit_price:,.2f} = {line.expected_amount:,.2f}, "
                f"printed {line.printed_amount:,.2f}",
            )
            for line in invoice.lines
            if differs(line.expected_amount, line.printed_amount)
        ]


class ShortDeliveryCheck(Check):
    """Ordered vs delivered: billing on delivered is right, but the open balance must be tracked."""

    name = "short_delivery"

    def run(self, invoice: Invoice) -> list[Issue]:
        return [
            self.issue(
                Severity.WARN,
                f"line {line.no}",
                f"ordered {line.ordered:g}, delivered {line.delivered:g}: billed on delivered, "
                f"{line.ordered - line.delivered:g} pending",
            )
            for line in invoice.lines
            if line.ordered != line.delivered
        ]


class SubtotalCheck(Check):
    """Each group's printed subtotal must equal the sum of its corrected lines."""

    name = "subtotal"

    def run(self, invoice: Invoice) -> list[Issue]:
        return [
            self.issue(
                Severity.ERROR,
                f"subtotal {group.name}",
                f"lines add to {group.expected_subtotal:,.2f}, printed {group.printed_subtotal:,.2f}",
            )
            for group in invoice.groups
            if differs(group.expected_subtotal, group.printed_subtotal)
        ]


class TotalsChainCheck(Check):
    """Total, discount, taxable amount and grand total, recomputed from the lines."""

    name = "totals_chain"

    def run(self, invoice: Invoice) -> list[Issue]:
        expected = recompute(invoice)
        t = invoice.totals
        pairs = {
            "total_before_discount": (expected.total_before_discount, to_number(t.total_before_discount_raw)),
            "discount": (expected.discount, abs(to_number(t.discount_raw))),
            "taxable_amount": (expected.taxable_amount, to_number(t.taxable_amount_raw)),
            "grand_total": (expected.grand_total, to_number(t.grand_total_raw)),
        }
        return [
            self.issue(Severity.ERROR, name, f"should be {want:,.2f}, printed {got:,.2f}")
            for name, (want, got) in pairs.items()
            if differs(want, got)
        ]
