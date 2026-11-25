"""VAT rules (UAE FTA: 5% on the taxable amount, i.e. after discount)."""

from __future__ import annotations

from jubii7.invoice.checks.base import Check, differs
from jubii7.invoice.models import Invoice, Issue, Severity
from jubii7.text import to_number


class VatBaseCheck(Check):
    """Printed VAT must be the VAT rate times the *printed* taxable amount.

    Comparing against the printed taxable amount (not the recomputed one) isolates
    the root cause: a wrong VAT base, rather than a knock-on from a line error.
    """

    name = "vat_base"

    def run(self, invoice: Invoice) -> list[Issue]:
        t = invoice.totals
        printed_vat = to_number(t.vat_raw)
        taxable = to_number(t.taxable_amount_raw)
        expected = round(taxable * t.vat_rate, 2)
        if not differs(expected, printed_vat):
            return []

        pre_discount_vat = round(to_number(t.total_before_discount_raw) * t.vat_rate, 2)
        cause = "" if differs(pre_discount_vat, printed_vat) else " (VAT charged on the pre-discount total)"
        return [
            self.issue(
                Severity.ERROR,
                "vat",
                f"{t.vat_rate:.0%} of printed taxable {taxable:,.2f} = {expected:,.2f}, "
                f"printed {printed_vat:,.2f}{cause}",
            )
        ]
