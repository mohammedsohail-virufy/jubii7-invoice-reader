"""Model evaluation: cross-validation, hold-out metrics, error analysis and diagnostic curves.

Every function takes a ModelSpec (or a fitted pipeline) and plain data, and returns plain
data. Plotting lives in plots.py, so evaluation is testable without matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    learning_curve,
    validation_curve,
)
from sklearn.pipeline import Pipeline

from jubii7.classifier import LABELS, SEED
from jubii7.classifier.models import LogRegSpec, ModelSpec, SgdSpec
from jubii7.classifier.rules import KeywordRules

N_SPLITS = 5
N_REPEATS = 5


@dataclass
class CvResult:
    name: str
    f1_macro_mean: float
    f1_macro_std: float
    accuracy_mean: float
    fold_scores: list[float] = field(default_factory=list)

    def metrics(self) -> dict[str, float]:
        return {
            "cv_f1_macro_mean": round(self.f1_macro_mean, 6),
            "cv_f1_macro_std": round(self.f1_macro_std, 6),
            "cv_accuracy_mean": round(self.accuracy_mean, 4),
        }


def cross_validate_spec(spec: ModelSpec, texts: pd.Series, labels: pd.Series) -> CvResult:
    """Repeated stratified 5-fold CV (25 fits): a stable estimate on a small dataset."""
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    scores = cross_validate(spec.build(), texts, labels, cv=cv, scoring=("f1_macro", "accuracy"))
    f1 = scores["test_f1_macro"]
    return CvResult(
        name=spec.name,
        f1_macro_mean=float(np.mean(f1)),
        f1_macro_std=float(np.std(f1)),
        accuracy_mean=float(np.mean(scores["test_accuracy"])),
        fold_scores=[float(s) for s in f1],
    )


@dataclass
class HoldoutReport:
    f1_macro: float
    accuracy: float
    log_loss: float
    per_class: dict[str, dict[str, float]]
    report_text: str
    confusion_matrix: list[list[int]]

    def summary(self) -> dict[str, float]:
        return {"f1_macro": self.f1_macro, "accuracy": self.accuracy, "log_loss": self.log_loss}

    def to_dict(self) -> dict[str, object]:
        return {**self.summary(), "per_class": self.per_class, "confusion_matrix": self.confusion_matrix}


def holdout_report(model: Pipeline, texts: pd.Series, labels: pd.Series) -> HoldoutReport:
    """Per-class precision / recall / F1, confusion matrix and log loss on a held-out set."""
    predicted = model.predict(texts)
    proba = model.predict_proba(texts)
    return HoldoutReport(
        f1_macro=float(f1_score(labels, predicted, average="macro")),
        accuracy=float(accuracy_score(labels, predicted)),
        log_loss=float(log_loss(labels, proba, labels=list(model.classes_))),
        per_class=classification_report(
            labels, predicted, labels=list(LABELS), output_dict=True, zero_division=0
        ),
        report_text=str(classification_report(labels, predicted, labels=list(LABELS), zero_division=0)),
        confusion_matrix=confusion_matrix(labels, predicted, labels=list(LABELS)).tolist(),
    )


def cv_mistakes(spec: ModelSpec, frame: pd.DataFrame) -> pd.DataFrame:
    """Rows the model gets wrong under 5-fold CV, with the predicted label and its confidence."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    proba = cross_val_predict(spec.build(), frame["text"], frame["label"], cv=cv, method="predict_proba")
    classes = np.array(sorted(frame["label"].unique()))
    predicted = classes[proba.argmax(axis=1)]
    wrong = predicted != frame["label"].to_numpy()
    return frame[wrong].assign(predicted=predicted[wrong], confidence=proba.max(axis=1)[wrong].round(3))


def evaluate_cascade(
    spec: ModelSpec, frame: pd.DataFrame, rules: KeywordRules, review_below: float = 0.5
) -> dict[str, float]:
    """Model alone vs rules -> model, under the same 5x5 repeated CV folds.

    Business metrics as well as F1:
    - automation_rate: share of lines decided without a person
    - automated_accuracy: accuracy of those automatic decisions (what finance cares about)
    """
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    texts, labels = frame["text"], frame["label"].to_numpy()
    rule_hits = [rules.match(t) for t in texts]
    totals: dict[str, list[float]] = {
        k: []
        for k in (
            "model_f1",
            "cascade_f1",
            "rule_coverage",
            "rule_precision",
            "model_automation",
            "cascade_automation",
            "model_automated_accuracy",
            "cascade_automated_accuracy",
        )
    }
    for tr, va in cv.split(texts, labels):
        model = spec.build().fit(texts.iloc[tr], labels[tr])
        proba = model.predict_proba(texts.iloc[va])
        model_pred = model.classes_[proba.argmax(axis=1)]
        confident = proba.max(axis=1) >= review_below
        hits = [rule_hits[i] for i in va]
        has_rule = np.array([h is not None for h in hits])
        cascade_pred = np.array([h.category if h else m for h, m in zip(hits, model_pred, strict=True)])
        cascade_auto = has_rule | confident
        truth = labels[va]

        totals["model_f1"].append(f1_score(truth, model_pred, average="macro"))
        totals["cascade_f1"].append(f1_score(truth, cascade_pred, average="macro"))
        totals["rule_coverage"].append(has_rule.mean())
        totals["rule_precision"].append(
            (cascade_pred[has_rule] == truth[has_rule]).mean() if has_rule.any() else 1.0
        )
        totals["model_automation"].append(confident.mean())
        totals["cascade_automation"].append(cascade_auto.mean())
        totals["model_automated_accuracy"].append(_acc(model_pred, truth, confident))
        totals["cascade_automated_accuracy"].append(_acc(cascade_pred, truth, cascade_auto))
    return {k: round(float(np.mean(v)), 4) for k, v in totals.items()}


def evaluate_challenge(
    spec: ModelSpec,
    train: pd.DataFrame,
    challenge: pd.DataFrame,
    rules: KeywordRules,
    review_below: float = 0.5,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Out-of-distribution test: new wording, never seen by the model or used to write the rules.

    This is the honest number. CV on the 57 rows flatters the rules, because the word list was
    written by someone who had read those rows.
    """
    model = spec.build().fit(train["text"], train["label"])
    proba = model.predict_proba(challenge["text"])
    model_pred = model.classes_[proba.argmax(axis=1)]
    confidence = proba.max(axis=1)
    hits = [rules.match(t) for t in challenge["text"]]
    has_rule = np.array([h is not None for h in hits])
    cascade_pred = np.array([h.category if h else m for h, m in zip(hits, model_pred, strict=True)])
    cascade_auto = has_rule | (confidence >= review_below)
    truth = challenge["label"].to_numpy()

    rows = challenge.assign(
        rule=[h.category if h else "" for h in hits],
        model=model_pred,
        model_confidence=confidence.round(3),
        cascade=cascade_pred,
        automatic=cascade_auto,
        correct=cascade_pred == truth,
    )
    summary = {
        "rows": len(challenge),
        "model_f1": round(float(f1_score(truth, model_pred, average="macro")), 4),
        "cascade_f1": round(float(f1_score(truth, cascade_pred, average="macro")), 4),
        "rule_coverage": round(float(has_rule.mean()), 4),
        "rule_precision": round(_acc(cascade_pred, truth, has_rule), 4),
        "model_automation": round(float((confidence >= review_below).mean()), 4),
        "cascade_automation": round(float(cascade_auto.mean()), 4),
        "model_automated_accuracy": round(_acc(model_pred, truth, confidence >= review_below), 4),
        "cascade_automated_accuracy": round(_acc(cascade_pred, truth, cascade_auto), 4),
    }
    return summary, rows


def _acc(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    return float((pred[mask] == truth[mask]).mean()) if mask.any() else 1.0


# ---------- diagnostic curves ----------


def learning_curve_data(spec: ModelSpec, texts: pd.Series, labels: pd.Series) -> dict[str, list[float]]:
    """Macro-F1 vs number of training rows. A rising validation curve means more data will help."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    sizes, train, valid = learning_curve(
        spec.build(),
        texts,
        labels,
        cv=cv,
        scoring="f1_macro",
        train_sizes=np.linspace(0.3, 1.0, 6),
        shuffle=True,
        random_state=SEED,
    )
    return _curve(sizes, train, valid, key="train_rows")


def regularisation_curve_data(
    spec: LogRegSpec, texts: pd.Series, labels: pd.Series
) -> dict[str, list[float]]:
    """Macro-F1 vs C for logistic regression (C = inverse regularisation strength)."""
    c_values = np.logspace(-2, 3, 11)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    train, valid = validation_curve(
        spec.build(), texts, labels, param_name="clf__C", param_range=c_values, cv=cv, scoring="f1_macro"
    )
    return _curve(c_values, train, valid, key="C")


def learning_rate_sweep(spec: SgdSpec, texts: pd.Series, labels: pd.Series) -> dict[str, list[float]]:
    """Macro-F1 vs SGD learning rate (eta0): finds the stable range."""
    rates = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0])
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    train, valid = validation_curve(
        spec.build(), texts, labels, param_name="clf__eta0", param_range=rates, cv=cv, scoring="f1_macro"
    )
    return _curve(rates, train, valid, key="eta0")


def epoch_curves(
    spec: SgdSpec,
    texts: pd.Series,
    labels: pd.Series,
    rates: tuple[float, ...] = (0.01, 0.1, 1.0),
    epochs: int = 40,
) -> dict[float, dict[str, list[float]]]:
    """Training loss and validation macro-F1 per epoch for a few learning rates (averaged over folds)."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    classes = np.array(sorted(labels.unique()))
    out: dict[float, dict[str, list[float]]] = {}
    for rate in rates:
        losses = np.zeros((N_SPLITS, epochs))
        f1s = np.zeros((N_SPLITS, epochs))
        for fold, (tr, va) in enumerate(cv.split(texts, labels)):
            vec = clone(spec.vectorizer())
            x_tr, x_va = vec.fit_transform(texts.iloc[tr]), vec.transform(texts.iloc[va])
            y_tr, y_va = labels.iloc[tr], labels.iloc[va]
            clf = SGDClassifier(
                loss="log_loss", learning_rate="constant", eta0=rate, alpha=spec.alpha, random_state=SEED
            )
            for epoch in range(epochs):
                clf.partial_fit(x_tr, y_tr, classes=classes)
                losses[fold, epoch] = log_loss(y_tr, clf.predict_proba(x_tr), labels=classes)
                f1s[fold, epoch] = f1_score(y_va, clf.predict(x_va), average="macro")
        out[rate] = {"train_loss": losses.mean(axis=0).tolist(), "val_f1": f1s.mean(axis=0).tolist()}
    return out


def _curve(x: np.ndarray, train: np.ndarray, valid: np.ndarray, key: str) -> dict[str, list[float]]:
    return {
        key: [float(v) for v in x],
        "train_mean": train.mean(axis=1).tolist(),
        "train_std": train.std(axis=1).tolist(),
        "valid_mean": valid.mean(axis=1).tolist(),
        "valid_std": valid.std(axis=1).tolist(),
    }
