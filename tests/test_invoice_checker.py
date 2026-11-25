from __future__ import annotations

from typing import Any

import pytest

from jubii7.invoice import InvoiceChecker, Severity, default_checker
from jubii7.invoice.checks import ALL_CHECKS, Check
from jubii7.invoice.loader import InvoiceFormatError, parse_invoice
from jubii7.invoice.models import Invoice, Issue
from jubii7.invoice.totals import recompute

# The answer key for the challenge invoice: (severity, field) pairs the checker must report.
EXPECTED = {
    (Severity.ERROR, "line 3"),
    (Severity.ERROR, "subtotal Site A - Foundation"),
    (Severity.ERROR, "total_before_discount"),
    (Severity.ERROR, "taxable_amount"),
    (Severity.ERROR, "vat"),
    (Severity.ERROR, "grand_total"),
    (Severity.ERROR, "amount_in_words"),
    (Severity.ERROR, "payment"),
    (Severity.ERROR, "terms"),
    (Severity.ERROR, "delivery D3"),
    (Severity.WARN, "line 5"),
    (Severity.WARN, "delivery D3"),
    (Severity.WARN, "customer_signature"),
    (Severity.INFO, "delivery_to"),
}


def test_recomputed_totals(invoice: Invoice) -> None:
    totals = recompute(invoice)
    assert totals.subtotals == (14_780.00, 7_760.00)
    assert totals.discount == 155.20
    assert totals.taxable_amount == 22_384.80
    assert totals.vat == 1_119.24
    assert totals.grand_total == 23_504.04


def test_challenge_invoice_matches_answer_key(invoice: Invoice) -> None:
    report = default_checker().run(invoice)
    found = {(issue.severity, issue.field) for issue in report.issues}
    assert found == EXPECTED
    assert report.status.startswith("REJECT")


def test_vat_check_names_the_root_cause(invoice: Invoice) -> None:
    vat = next(i for i in default_checker().run(invoice).issues if i.field == "vat")
    assert "pre-discount" in vat.message


def test_issues_are_sorted_errors_first(invoice: Invoice) -> None:
    severities = [i.severity for i in default_checker().run(invoice).issues]
    order = [Severity.ERROR, Severity.WARN, Severity.INFO]
    assert severities == sorted(severities, key=order.index)


def test_clean_invoice_passes(clean_invoice: Invoice) -> None:
    report = default_checker().run(clean_invoice)
    assert not report.has_errors
    assert report.count(Severity.WARN) == 0
    assert report.status == "OK"


@pytest.mark.parametrize("check_cls", ALL_CHECKS, ids=lambda c: c.name)
def test_every_check_honours_the_contract(
    check_cls: type[Check], invoice: Invoice, clean_invoice: Invoice
) -> None:
    """Liskov substitution: any Check can stand in for another in the pipeline."""
    check = check_cls()
    for inv in (invoice, clean_invoice):
        before = repr(inv)
        issues = check.run(inv)
        assert isinstance(issues, list)
        assert all(isinstance(i, Issue) and i.check == check.name for i in issues)
        assert repr(inv) == before  # never mutates its input


def test_checker_is_open_for_extension(invoice: Invoice) -> None:
    class AlwaysWarn(Check):
        name = "always_warn"

        def run(self, invoice: Invoice) -> list[Issue]:
            return [self.issue(Severity.WARN, "custom", "added without touching the checker")]

    report = InvoiceChecker([AlwaysWarn()]).run(invoice)
    assert [i.check for i in report.issues] == ["always_warn"]


def test_a_crashing_check_is_reported_not_raised(invoice: Invoice) -> None:
    class Broken(Check):
        name = "broken"

        def run(self, invoice: Invoice) -> list[Issue]:
            raise ValueError("bad data")

    report = InvoiceChecker([Broken()]).run(invoice)
    assert report.issues[0].field == "unreadable"


def test_loader_reports_missing_fields(raw_invoice: dict[str, Any]) -> None:
    del raw_invoice["totals"]
    with pytest.raises(InvoiceFormatError, match="totals"):
        parse_invoice(raw_invoice)


def test_discount_label_is_parsed(invoice: Invoice) -> None:
    assert invoice.totals.discount_rate == 0.02
    assert invoice.groups[invoice.totals.discount_group_index].name.startswith("Site B")


def test_profile_finds_arabic_digits_and_flags(invoice: Invoice) -> None:
    profile = default_checker().run(invoice).profile
    assert any("po_no" in f for f in profile.fields_with_arabic_digits)
    assert "customer_trn" in profile.flagged_for_verification
    assert "customer_signature" in profile.missing_values
