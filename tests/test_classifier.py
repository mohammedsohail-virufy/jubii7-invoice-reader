from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from jubii7.classifier import LABELS
from jubii7.classifier.data import DatasetError, load_dataset
from jubii7.classifier.models import CANDIDATES, ModelSpec
from jubii7.classifier.predictor import ModelIntegrityError, Predictor
from jubii7.classifier.tracking import ExperimentTracker, JsonTracker
from jubii7.classifier.train import train

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_is_valid() -> None:
    ds = load_dataset(ROOT / "data" / "lines.csv")
    assert set(ds.labels) == set(LABELS)
    assert ds.summary()["arabic_rows"] > 0


def test_dataset_rejects_unknown_labels(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["x"], "label": ["timber"], "source": ["test"]}).to_csv(path, index=False)
    with pytest.raises(DatasetError, match="timber"):
        load_dataset(path)


def test_dataset_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "dup.csv"
    pd.DataFrame({"text": ["a", "a"], "label": ["steel", "steel"], "source": ["t", "t"]}).to_csv(
        path, index=False
    )
    with pytest.raises(DatasetError, match="duplicate"):
        load_dataset(path)


@pytest.mark.parametrize("spec", CANDIDATES, ids=lambda s: s.name)
def test_every_model_spec_is_substitutable(spec: ModelSpec) -> None:
    """Liskov: every spec builds a pipeline with fit / predict / predict_proba over the same labels."""
    ds = load_dataset(ROOT / "data" / "lines.csv")
    model = spec.build().fit(ds.texts, ds.labels)
    proba = model.predict_proba(["Rebar 16 mm", "حديد تسليح ١٦ مم"])
    assert proba.shape == (2, len(LABELS))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(model.classes_) == set(LABELS)


class _RecordingTracker(ExperimentTracker):
    def __init__(self) -> None:
        self.names: list[str] = []

    def log_run(self, name: str, params: dict[str, object], metrics: dict[str, float], model: Any) -> str:
        self.names.append(name)
        return f"test-{name}"


@pytest.fixture(scope="module")
def trained_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("project")
    shutil.copytree(ROOT / "data", root / "data")
    train(root, tracker=_RecordingTracker(), with_curves=False)
    return root


def test_training_produces_model_card_and_reports(trained_root: Path) -> None:
    assert (trained_root / "models" / "model.joblib").exists()
    assert (trained_root / "models" / "model_card.json").exists()
    assert (trained_root / "reports" / "figures" / "04_confusion_matrix.png").exists()


def test_model_beats_majority_baseline(trained_root: Path) -> None:
    import json

    card = json.loads((trained_root / "models" / "model_card.json").read_text(encoding="utf-8"))
    assert card["runs"][card["model"]]["cv_f1_macro_mean"] > 0.6  # majority class baseline is ~0.07


def test_predictor_end_to_end(trained_root: Path) -> None:
    predictor = Predictor.from_files(
        trained_root / "models/model.joblib", trained_root / "models/model_card.json"
    )
    assert predictor.predict("حديد تسليح ١٦ مم").category == "steel"


def test_tampered_model_is_refused(trained_root: Path, tmp_path: Path) -> None:
    model = tmp_path / "model.joblib"
    model.write_bytes((trained_root / "models/model.joblib").read_bytes() + b"tampered")
    with pytest.raises(ModelIntegrityError):
        Predictor.from_files(model, trained_root / "models/model_card.json")


def test_low_confidence_goes_to_review() -> None:
    class Unsure:
        classes_ = np.array(LABELS)

        def predict_proba(self, texts: list[str]) -> Any:
            return np.full((len(texts), len(LABELS)), 1 / len(LABELS))

    assert Predictor(Unsure()).predict("Waterproofing membrane").needs_review


def test_json_tracker_appends_runs(tmp_path: Path) -> None:
    tracker = JsonTracker(tmp_path / "runs.json")
    tracker.log_run("a", {}, {"f1": 0.5}, None)
    tracker.log_run("b", {}, {"f1": 0.6}, None)
    assert (tmp_path / "runs.json").read_text().count("run_id") == 2
