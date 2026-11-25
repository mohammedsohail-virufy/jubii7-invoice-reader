from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jubii7.api.app import create_app
from jubii7.web.dashboard import CodeBrowser

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(root=ROOT))


def test_index_page_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Invoice Reader" in response.text


def test_dashboard_payload_has_every_section(client: TestClient) -> None:
    body = client.get("/api/dashboard").json()
    assert {"model_card", "evaluation", "figures", "quality", "challenge_rows"} <= body.keys()


@pytest.mark.parametrize("variant", ["original", "corrected"])
def test_sample_invoices(client: TestClient, variant: str) -> None:
    assert client.get(f"/api/sample-invoice?variant={variant}").json()["invoice_no"] == "WB-2026-11-0457"


def test_corrected_sample_passes_the_checker(client: TestClient) -> None:
    invoice = client.get("/api/sample-invoice?variant=corrected").json()
    assert client.post("/invoice/check", json=invoice).json()["status"] == "OK"


def test_code_browser_lists_source_and_tests(client: TestClient) -> None:
    files = client.get("/api/files").json()
    assert "src/jubii7/invoice/checks/base.py" in files
    assert "tests/test_dashboard.py" in files
    assert not any(".venv" in f or f.endswith(".joblib") for f in files)


@pytest.mark.parametrize(
    "path",
    [
        "../../Windows/win.ini",
        "..\\..\\secret.txt",
        "/etc/passwd",
        "models/model.joblib",
        ".venv/pyvenv.cfg",
        "mlflow.db",
    ],
)
def test_code_browser_blocks_path_traversal_and_non_whitelisted_files(client: TestClient, path: str) -> None:
    assert client.get("/api/file", params={"path": path}).status_code == 404


def test_code_browser_reads_a_whitelisted_file() -> None:
    assert "class Check" in CodeBrowser(ROOT).read("src/jubii7/invoice/checks/base.py")
