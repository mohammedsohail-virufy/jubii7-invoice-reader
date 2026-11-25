"""Run a set of checks over an invoice and produce one report.

The checker depends only on the Check abstraction (dependency inversion), so rules can
be added, removed or swapped per client without touching this file (open/closed).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from jubii7.invoice.checks import ALL_CHECKS, Check
from jubii7.invoice.models import Invoice, Issue, Severity
from jubii7.invoice.profiling import Profile, profile
from jubii7.invoice.totals import CorrectedTotals, recompute

logger = logging.getLogger(__name__)

_SEVERITY_ORDER = {Severity.ERROR: 0, Severity.WARN: 1, Severity.INFO: 2}


@dataclass(frozen=True)
class CheckReport:
    invoice_no: str
    profile: Profile
    issues: tuple[Issue, ...]
    corrected_totals: CorrectedTotals

    @property
    def has_errors(self) -> bool:
        return any(issue.severity is Severity.ERROR for issue in self.issues)

    @property
    def status(self) -> str:
        return "REJECT: needs human review" if self.has_errors else "OK"

    def count(self, severity: Severity) -> int:
        return sum(1 for issue in self.issues if issue.severity is severity)

    def to_dict(self) -> dict[str, object]:
        return {
            "invoice_no": self.invoice_no,
            "status": self.status,
            "summary": {s.value: self.count(s) for s in Severity},
            "issues": [issue.to_dict() for issue in self.issues],
            "corrected_totals": self.corrected_totals.to_dict(),
            "profile": self.profile.to_dict(),
        }


class InvoiceChecker:
    """Applies checks in order. A failing check is logged and reported, never crashes the run."""

    def __init__(self, checks: Iterable[Check]) -> None:
        self._checks: Sequence[Check] = tuple(checks)

    @property
    def checks(self) -> Sequence[Check]:
        return self._checks

    def run(self, invoice: Invoice) -> CheckReport:
        issues: list[Issue] = []
        for check in self._checks:
            issues.extend(self._run_one(check, invoice))
        issues.sort(key=lambda issue: _SEVERITY_ORDER[issue.severity])
        return CheckReport(
            invoice_no=invoice.invoice_no,
            profile=profile(invoice.raw),
            issues=tuple(issues),
            corrected_totals=recompute(invoice),
        )

    @staticmethod
    def _run_one(check: Check, invoice: Invoice) -> list[Issue]:
        try:
            return check.run(invoice)
        except (ValueError, KeyError, IndexError, TypeError) as error:
            logger.exception("check %s failed", check.name)
            return [check.issue(Severity.ERROR, "unreadable", f"check could not run: {error}")]


def default_checker() -> InvoiceChecker:
    """Checker with every built-in rule."""
    return InvoiceChecker(check() for check in ALL_CHECKS)
