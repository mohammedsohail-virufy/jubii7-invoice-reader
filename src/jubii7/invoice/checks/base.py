"""The contract every invoice check follows.

Design (SOLID):
- Single responsibility: each check class verifies one business rule.
- Open/closed: add a rule by adding a class; the checker never changes.
- Liskov substitution: every Check takes an Invoice and returns a list of Issues,
  never raises for bad data and never mutates the invoice. Any check can replace any
  other in the pipeline (tests/test_checks.py::test_every_check_honours_the_contract).
- Interface segregation: one small method, `run`.
- Dependency inversion: InvoiceChecker depends on this abstraction, not on concrete rules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from jubii7.invoice.models import Invoice, Issue, Severity

TOLERANCE = 0.01  # 1 fils


class Check(ABC):
    """One business rule. Subclasses set `name` and implement `run`."""

    name: ClassVar[str] = "check"

    @abstractmethod
    def run(self, invoice: Invoice) -> list[Issue]:
        """Return the issues found. An empty list means the rule passed."""

    def issue(self, severity: Severity, field: str, message: str) -> Issue:
        """Helper so every issue is tagged with the check that raised it."""
        return Issue(severity=severity, field=field, message=message, check=self.name)


def differs(a: float, b: float) -> bool:
    """Money comparison with a 1-fils tolerance."""
    return abs(a - b) > TOLERANCE
