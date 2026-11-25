"""FastAPI service: classify invoice lines, check a whole extracted invoice, and serve the dashboard.

Run:  jubii7 serve            then open http://127.0.0.1:8000  (API docs at /docs)

`create_app` receives its dependencies (dependency injection), so tests pass a fake
classifier and no model file is needed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from jubii7 import __version__
from jubii7.classifier.cascade import Cascade, LineClassifier, ModelClassifier, RuleClassifier
from jubii7.classifier.predictor import Predictor
from jubii7.invoice.checker import InvoiceChecker, default_checker
from jubii7.invoice.loader import InvoiceFormatError, parse_invoice

ROOT = Path(os.environ.get("JUBII7_ROOT", Path.cwd()))


class LineIn(BaseModel):
    text: str = Field(min_length=1, max_length=500, examples=["حديد تسليح ١٦ مم"])


class BatchIn(BaseModel):
    lines: list[LineIn] = Field(min_length=1, max_length=200)


def default_classifier(root: Path = ROOT) -> LineClassifier:
    """Rules first, then the trained model; unsure lines come back with needs_review=true."""
    predictor = Predictor.from_files(root / "models/model.joblib", root / "models/model_card.json")
    return Cascade([RuleClassifier(), ModelClassifier(predictor)])


class LazyClassifier:
    """Loads the model on first use, so the API starts even before training."""

    def __init__(self, classifier: LineClassifier | None, root: Path) -> None:
        self._classifier = classifier
        self._root = root

    def get(self) -> LineClassifier:
        if self._classifier is None:
            try:
                self._classifier = default_classifier(self._root)
            except FileNotFoundError as error:
                raise HTTPException(503, "model not trained yet: run `jubii7 train`") from error
        return self._classifier


def create_app(
    classifier: LineClassifier | None = None,
    checker: InvoiceChecker | None = None,
    root: Path = ROOT,
    with_dashboard: bool = True,
) -> FastAPI:
    app = FastAPI(title="jubii.7 invoice reader", version=__version__)
    lazy = LazyClassifier(classifier, root)
    invoice_checker = checker or default_checker()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/predict")
    def predict(line: LineIn) -> dict[str, object]:
        return lazy.get().predict(line.text).to_dict()

    @app.post("/predict/batch")
    def predict_batch(batch: BatchIn) -> list[dict[str, object]]:
        model = lazy.get()
        return [model.predict(line.text).to_dict() for line in batch.lines]

    @app.post("/invoice/check")
    def check_invoice(invoice: dict[str, Any]) -> dict[str, object]:
        try:
            parsed = parse_invoice(invoice)
        except InvoiceFormatError as error:
            raise HTTPException(422, str(error)) from error
        return invoice_checker.run(parsed).to_dict()

    if with_dashboard:
        from jubii7.web.dashboard import mount_dashboard

        mount_dashboard(app, root)
    return app


app = create_app()
