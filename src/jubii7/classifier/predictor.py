"""Load the trained model safely and turn its output into a business decision."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from jubii7.text import preprocess

REVIEW_BELOW = 0.5  # send low-confidence lines to a person


class ModelIntegrityError(RuntimeError):
    """The model file does not match the checksum in its model card."""


class ProbabilisticModel(Protocol):
    classes_: Any

    def predict_proba(self, texts: list[str]) -> Any: ...


@dataclass(frozen=True)
class Prediction:
    text: str
    normalised: str
    category: str
    confidence: float
    needs_review: bool
    top3: list[dict[str, float | str]]
    source: str = "model"
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class Predictor:
    """Wraps any model with predict_proba (Liskov: a fake model works in tests)."""

    def __init__(self, model: ProbabilisticModel, review_below: float = REVIEW_BELOW) -> None:
        self.model = model
        self.review_below = review_below

    @classmethod
    def from_files(cls, model_path: Path, card_path: Path) -> Predictor:
        """Verify the SHA-256 from the model card before unpickling (joblib can run code)."""
        expected = json.loads(card_path.read_text(encoding="utf-8"))["sha256"]
        actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if actual != expected:
            raise ModelIntegrityError(f"{model_path.name} checksum mismatch")
        import joblib  # lazy: the browser build (Pyodide) uses PortableModel and has no joblib

        # Safe: SHA-256 verified above and the file is produced by our own training run.
        return cls(joblib.load(model_path))  # nosec B301

    def predict(self, text: str) -> Prediction:
        proba = self.model.predict_proba([text])[0]
        ranked = sorted(zip(self.model.classes_, proba, strict=True), key=lambda pair: -pair[1])
        label, confidence = ranked[0]
        unsure = confidence < self.review_below
        return Prediction(
            text=text,
            normalised=preprocess(text),
            category=str(label),
            confidence=round(float(confidence), 3),
            needs_review=bool(unsure),
            top3=[{"category": str(c), "p": round(float(p), 3)} for c, p in ranked[:3]],
            source="model",
            reason=f"confidence below {self.review_below}: human review" if unsure else "model confident",
        )
