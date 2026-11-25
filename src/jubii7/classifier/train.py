"""Training pipeline: compare candidates, select by CV, evaluate, plot, save.

Run:  jubii7 train            (add --fast to skip the diagnostic curves)

Each stage is a small function with typed inputs and outputs, so it can be tested or
moved to its own pipeline step (e.g. a SageMaker Pipelines step) on its own.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from jubii7 import __version__
from jubii7.classifier import SEED, plots
from jubii7.classifier.data import Dataset, load_dataset
from jubii7.classifier.evaluate import (
    HoldoutReport,
    cross_validate_spec,
    cv_mistakes,
    epoch_curves,
    evaluate_cascade,
    evaluate_challenge,
    holdout_report,
    learning_curve_data,
    learning_rate_sweep,
    regularisation_curve_data,
)
from jubii7.classifier.models import CANDIDATES, LogRegSpec, ModelSpec, SgdSpec
from jubii7.classifier.portable import export_model
from jubii7.classifier.rules import KeywordRules
from jubii7.classifier.tracking import ExperimentTracker, make_tracker

logger = logging.getLogger(__name__)
TEST_SIZE = 0.3
Runs = dict[str, dict[str, float | str]]


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data" / "lines.csv"

    @property
    def challenge(self) -> Path:
        return self.root / "data" / "challenge_set.csv"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    @property
    def model(self) -> Path:
        return self.root / "models" / "model.joblib"

    @property
    def model_card(self) -> Path:
        return self.root / "models" / "model_card.json"


@dataclass(frozen=True)
class Split:
    x_train: pd.Series
    x_test: pd.Series
    y_train: pd.Series
    y_test: pd.Series

    @classmethod
    def of(cls, texts: pd.Series, labels: pd.Series) -> Split:
        x_tr, x_te, y_tr, y_te = train_test_split(
            texts, labels, test_size=TEST_SIZE, stratify=labels, random_state=SEED
        )
        return cls(x_tr, x_te, y_tr, y_te)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train(
    root: Path,
    candidates: tuple[ModelSpec, ...] = CANDIDATES,
    tracker: ExperimentTracker | None = None,
    with_curves: bool = True,
) -> dict[str, object]:
    """Compare -> select -> evaluate -> plot -> save. Returns the model card."""
    paths = Paths(root)
    dataset = load_dataset(paths.data)
    split = Split.of(dataset.texts, dataset.labels)

    runs, fold_scores = compare_candidates(candidates, dataset, split, tracker or make_tracker(root))
    best = max(
        candidates, key=lambda s: float(runs[s.name]["cv_f1_macro_mean"])
    )  # CV, not the tiny test split
    holdout = holdout_report(best.build().fit(split.x_train, split.y_train), split.x_test, split.y_test)

    paths.figures.mkdir(parents=True, exist_ok=True)
    write_holdout_reports(
        best=best, dataset=dataset, holdout=holdout, runs=runs, fold_scores=fold_scores, paths=paths
    )
    cascade, challenge = evaluate_rules_cascade(best, dataset, paths)
    curves = diagnostic_curves(best, candidates, dataset, paths) if with_curves else {}

    card = save_model(
        best=best,
        dataset=dataset,
        paths=paths,
        runs=runs,
        holdout=holdout,
        cascade=cascade,
        challenge=challenge,
    )
    evaluation = {
        "runs": runs,
        "chosen": best.name,
        "holdout": holdout.to_dict(),
        "cascade_cv": cascade,
        "challenge": challenge,
        "curves": curves,
    }
    (paths.reports / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    return card


def compare_candidates(
    candidates: tuple[ModelSpec, ...], dataset: Dataset, split: Split, tracker: ExperimentTracker
) -> tuple[Runs, dict[str, list[float]]]:
    """Repeated CV for every candidate, plus its hold-out score, logged to the tracker."""
    runs: Runs = {}
    fold_scores: dict[str, list[float]] = {}
    for spec in candidates:
        cv = cross_validate_spec(spec, dataset.texts, dataset.labels)
        model = spec.build().fit(split.x_train, split.y_train)
        holdout = holdout_report(model, split.x_test, split.y_test)
        metrics = {
            **cv.metrics(),
            "test_f1_macro": round(holdout.f1_macro, 4),
            "test_log_loss": round(holdout.log_loss, 4),
        }
        params = {**spec.params(), "rows": len(dataset.frame), "seed": SEED}
        runs[spec.name] = {**metrics, "run_id": tracker.log_run(spec.name, params, metrics, model)}
        fold_scores[spec.name] = cv.fold_scores
        logger.info(
            "%-20s cv_f1=%.3f±%.3f test_f1=%.3f",
            spec.name,
            cv.f1_macro_mean,
            cv.f1_macro_std,
            holdout.f1_macro,
        )
    return runs, fold_scores


def write_holdout_reports(
    *,
    best: ModelSpec,
    dataset: Dataset,
    holdout: HoldoutReport,
    runs: Runs,
    fold_scores: dict[str, list[float]],
    paths: Paths,
) -> None:
    (paths.reports / "classification_report.txt").write_text(holdout.report_text, encoding="utf-8")
    cv_mistakes(best, dataset.frame).to_csv(paths.reports / "mistakes_cv.csv", index=False, encoding="utf-8")
    plots.model_comparison(runs, best.name, paths.figures / "01_model_comparison.png")
    plots.cv_distribution(fold_scores, paths.figures / "02_cv_fold_spread.png")
    plots.per_class_f1(holdout.per_class, paths.figures / "03_per_class_f1.png")
    plots.confusion(holdout.confusion_matrix, paths.figures / "04_confusion_matrix.png")


def evaluate_rules_cascade(
    best: ModelSpec, dataset: Dataset, paths: Paths
) -> tuple[dict[str, float], dict[str, float]]:
    """Rules -> model, on CV (optimistic for rules) and on the unseen challenge set (honest)."""
    cascade = evaluate_cascade(best, dataset.frame, KeywordRules())
    plots.cascade_comparison(
        cascade, paths.figures / "10_cascade_vs_model_cv.png", "CV on training rows (optimistic)"
    )
    challenge: dict[str, float] = {}
    if paths.challenge.exists():
        challenge_set = load_dataset(paths.challenge).frame
        challenge, rows = evaluate_challenge(best, dataset.frame, challenge_set, KeywordRules())
        rows.to_csv(paths.reports / "challenge_predictions.csv", index=False, encoding="utf-8")
        plots.cascade_comparison(
            challenge, paths.figures / "09_cascade_vs_model_challenge.png", "unseen challenge set"
        )
        logger.info("challenge set (honest): %s", challenge)
    return cascade, challenge


def diagnostic_curves(
    best: ModelSpec, candidates: tuple[ModelSpec, ...], dataset: Dataset, paths: Paths
) -> dict[str, object]:
    """Learning curve, regularisation (C) curve, learning-rate sweep and per-epoch curves."""
    texts, labels = dataset.texts, dataset.labels
    learning = learning_curve_data(best, texts, labels)
    plots.learning_curve(learning, paths.figures / "05_learning_curve.png")
    curves: dict[str, object] = {"learning_curve": learning}

    logreg = next((s for s in candidates if isinstance(s, LogRegSpec)), None)
    if logreg:
        regularisation = regularisation_curve_data(logreg, texts, labels)
        plots.regularisation_curve(regularisation, paths.figures / "06_regularisation_C.png")
        curves["regularisation"] = regularisation

    sgd = next((s for s in candidates if isinstance(s, SgdSpec)), None)
    if sgd:
        rates = learning_rate_sweep(sgd, texts, labels)
        plots.learning_rate_curve(rates, paths.figures / "07_learning_rate_sweep.png")
        epochs = epoch_curves(sgd, texts, labels)
        plots.epoch_curves(epochs, paths.figures / "08_epoch_curves.png")
        curves["learning_rate"] = rates
        curves["epochs"] = {str(k): v for k, v in epochs.items()}
    return curves


def save_model(
    *,
    best: ModelSpec,
    dataset: Dataset,
    paths: Paths,
    runs: Runs,
    holdout: HoldoutReport,
    cascade: dict[str, float],
    challenge: dict[str, float],
) -> dict[str, object]:
    """Fit on all rows, save, and write a model card with a SHA-256 for safe loading."""
    final = best.build().fit(dataset.texts, dataset.labels)
    paths.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, paths.model)
    if isinstance(best, LogRegSpec):  # browser-portable copy for the GitHub Pages demo
        portable = json.dumps(export_model(final), ensure_ascii=False)
        (paths.model.parent / "model_portable.json").write_text(portable, encoding="utf-8")
    card: dict[str, object] = {
        "model": best.name,
        "params": best.params(),
        "package_version": __version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data": dataset.summary(),
        "selection": "highest mean macro-F1 over 5x5 repeated stratified CV",
        "runs": runs,
        "holdout": holdout.summary(),
        "cascade_cv": cascade,
        "challenge_set": challenge,
        "known_limits": [
            "57 labelled rows; metrics show the method, not production accuracy",
            "waterproofing membrane has no class (unmapped -> human review)",
            "keyword rules miss unseen synonyms (e.g. 'شحن' = shipping); see the challenge set",
        ],
        "sha256": sha256(paths.model),
    }
    paths.model_card.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    return card
