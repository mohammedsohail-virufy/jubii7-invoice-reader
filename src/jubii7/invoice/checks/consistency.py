"""Form-field rules: words vs figures, payment method, TRNs, English vs Arabic terms."""

from __future__ import annotations

from jubii7.invoice.checks.base import Check, differs
from jubii7.invoice.models import Invoice, Issue, Severity
from jubii7.text import to_number, to_western

TRN_LENGTH = 15  # UAE Tax Registration Number


class AmountInWordsCheck(Check):
    """The amount written in words must match the grand total figure."""

    name = "amount_in_words"

    def run(self, invoice: Invoice) -> list[Issue]:
        words = to_number(invoice.amount_in_words_value)
        figure = to_number(invoice.totals.grand_total_raw)
        if not differs(words, figure):
            return []
        return [
            self.issue(
                Severity.ERROR, "amount_in_words", f"words say {words:,.2f}, figure says {figure:,.2f}"
            )
        ]


class PaymentMethodCheck(Check):
    """Exactly one payment method must be ticked, or the entry can't be posted."""

    name = "payment_method"

    def run(self, invoice: Invoice) -> list[Issue]:
        ticked = [method for method, is_ticked in invoice.payment.items() if is_ticked]
        if len(ticked) == 1:
            return []
        return [self.issue(Severity.ERROR, "payment", f"exactly one method expected, ticked: {ticked}")]


class TrnFormatCheck(Check):
    """Supplier and customer TRNs must be 15 digits."""

    name = "trn_format"

    def run(self, invoice: Invoice) -> list[Issue]:
        issues = []
        for who, trn in (("supplier", invoice.supplier_trn), ("customer", invoice.customer_trn)):
            digits = to_western(trn)
            if not (digits.isdigit() and len(digits) == TRN_LENGTH):
                issues.append(
                    self.issue(Severity.ERROR, f"{who}_trn", f"UAE TRN must be 15 digits, got {trn!r}")
                )
        return issues


class BilingualTermsCheck(Check):
    """Payment terms in English and Arabic must agree."""

    name = "bilingual_terms"

    def run(self, invoice: Invoice) -> list[Issue]:
        english, arabic = to_number(invoice.terms_days_en), to_number(invoice.terms_days_ar)
        if english == arabic:
            return []
        return [
            self.issue(Severity.ERROR, "terms", f"English says {english:g} days, Arabic says {arabic:g} days")
        ]
