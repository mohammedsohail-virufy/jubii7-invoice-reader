from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jubii7.classifier.cascade import Cascade, ModelClassifier, RuleClassifier
from jubii7.classifier.data import load_dataset
from jubii7.classifier.models import CANDIDATES, LogRegSpec, SgdSpec
from jubii7.classifier.portable import PortableModel, export_model
from jubii7.classifier.predictor import Predictor

ROOT = Path(__file__).resolve().parents[1]
LOGREG_SPECS = [s for s in CANDIDATES if isinstance(s, LogRegSpec)]


@pytest.fixture(scope="module")
def texts() -> list[str]:
    train = load_dataset(ROOT / "data" / "lines.csv").frame["text"].tolist()
    challenge = load_dataset(ROOT / "data" / "challenge_set.csv").frame["text"].tolist()
    return [*train, *challenge, "", "x", "a b", "١٢", "Rebar  16   MM"]


@pytest.mark.parametrize("spec", LOGREG_SPECS, ids=lambda s: s.name)
def test_portable_model_matches_scikit_learn(spec: LogRegSpec, texts: list[str]) -> None:
    """Parity test: same probabilities as scikit-learn, for word and char n-grams."""
    data = load_dataset(ROOT / "data" / "lines.csv")
    pipeline = spec.build().fit(data.texts, data.labels)
    weights = json.loads(json.dumps(export_model(pipeline)))  # round-trip through JSON like the site does
    portable = PortableModel(weights)
    assert portable.classes_ == list(pipeline.classes_)
    np.testing.assert_allclose(portable.predict_proba(texts), pipeline.predict_proba(texts), atol=1e-9)


def test_only_logistic_regression_is_exported() -> None:
    sgd = next(s for s in CANDIDATES if isinstance(s, SgdSpec))
    data = load_dataset(ROOT / "data" / "lines.csv")
    with pytest.raises(ValueError, match="logistic regression"):
        export_model(sgd.build().fit(data.texts, data.labels))


def test_portable_model_plugs_into_the_cascade() -> None:
    """Liskov: the portable model replaces the scikit-learn pipeline without code changes."""
    data = load_dataset(ROOT / "data" / "lines.csv")
    weights = export_model(LOGREG_SPECS[0].build().fit(data.texts, data.labels))
    cascade = Cascade([RuleClassifier(), ModelClassifier(Predictor(PortableModel(weights)))])
    assert cascade.predict("Rebar 12 mm").category == "steel"
    assert cascade.predict("Item 47").source == "model"
