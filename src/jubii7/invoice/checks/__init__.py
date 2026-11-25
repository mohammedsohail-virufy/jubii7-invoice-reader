"""Business rules for an invoice. Each rule is one small, swappable Check class."""

from jubii7.invoice.checks.arithmetic import (
    LineAmountCheck,
    ShortDeliveryCheck,
    SubtotalCheck,
    TotalsChainCheck,
)
from jubii7.invoice.checks.base import Check
from jubii7.invoice.checks.consistency import (
    AmountInWordsCheck,
    BilingualTermsCheck,
    PaymentMethodCheck,
    TrnFormatCheck,
)
from jubii7.invoice.checks.delivery import DeliveryChargeContextCheck, DeliveryDateCheck, SignatureCheck
from jubii7.invoice.checks.tax import VatBaseCheck

ALL_CHECKS: tuple[type[Check], ...] = (
    LineAmountCheck,
    ShortDeliveryCheck,
    SubtotalCheck,
    TotalsChainCheck,
    VatBaseCheck,
    AmountInWordsCheck,
    PaymentMethodCheck,
    TrnFormatCheck,
    BilingualTermsCheck,
    DeliveryDateCheck,
    SignatureCheck,
    DeliveryChargeContextCheck,
)

__all__ = ["ALL_CHECKS", "Check"]
