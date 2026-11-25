from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from jubii7.invoice.loader import parse_invoice
from jubii7.invoice.models import Invoice

ROOT = Path(__file__).resolve().parents[1]
INVOICE_PATH = ROOT / "data" / "invoice_WB-2026-11-0457.json"
# The same invoice with every problem fixed: the checker must find no ERROR or WARN.
CORRECTED_PATH = ROOT / "data" / "invoice_WB-2026-11-0457_corrected.json"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def raw_invoice() -> dict[str, Any]:
    return _read(INVOICE_PATH)


@pytest.fixture
def invoice(raw_invoice: dict[str, Any]) -> Invoice:
    return parse_invoice(raw_invoice)


@pytest.fixture
def clean_raw_invoice() -> dict[str, Any]:
    return _read(CORRECTED_PATH)


@pytest.fixture
def clean_invoice(clean_raw_invoice: dict[str, Any]) -> Invoice:
    return parse_invoice(clean_raw_invoice)
