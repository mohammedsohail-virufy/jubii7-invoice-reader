from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from jubii7.api.app import create_app
from jubii7.classifier import LABELS
from jubii7.classifier.cascade import Cascade, ModelClassifier, RuleClassifier
from jubii7.classifier.predictor import Predictor


class FakeModel:
    """Stands in for the trained pipeline (Liskov): always 90% steel."""

    classes_ = np.array(LABELS)

    def predict_proba(self, texts: list[str]) -> Any:
        row = [0.025, 0.9, 0.025, 0.025, 0.025]
        return np.array([row] * len(texts))


@pytest.fixture
def client() -> TestClient:
    cascade = Cascade([RuleClassifier(), ModelClassifier(Predictor(FakeModel()))])
    return TestClient(create_app(classifier=cascade, with_dashboard=False))


def test_health(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"


def test_predict_uses_rules_first(client: TestClient) -> None:
    body = client.post("/predict", json={"text": "Delivery of cement bags"}).json()
    assert (body["category"], body["source"]) == ("services", "rules")
    assert "delivery" in body["reason"]


def test_predict_falls_back_to_model(client: TestClient) -> None:
    body = client.post("/predict", json={"text": "item 47 grade 60"}).json()
    assert (body["category"], body["source"]) == ("steel", "model")
    assert body["needs_review"] is False


def test_predict_normalises_arabic_digits(client: TestClient) -> None:
    body = client.post("/predict", json={"text": "حديد تسليح ١٦ مم"}).json()
    assert body["category"] == "steel"
    assert body["normalised"] == "حديد تسليح 16 مم"


def test_predict_batch(client: TestClient) -> None:
    body = client.post("/predict/batch", json={"lines": [{"text": "a"}, {"text": "b"}]}).json()
    assert len(body) == 2


@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": "x" * 501}])
def test_predict_validates_input(client: TestClient, payload: dict[str, str]) -> None:
    assert client.post("/predict", json=payload).status_code == 422


def test_invoice_check_endpoint(client: TestClient, raw_invoice: dict[str, Any]) -> None:
    body = client.post("/invoice/check", json=raw_invoice).json()
    assert body["corrected_totals"]["grand_total"] == 23_504.04
    assert body["summary"]["ERROR"] == 10


def test_invoice_check_rejects_bad_json(client: TestClient) -> None:
    assert client.post("/invoice/check", json={"invoice_no": "x"}).status_code == 422
