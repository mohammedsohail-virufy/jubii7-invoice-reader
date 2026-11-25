"""Candidate models as small, interchangeable specs.

Liskov substitution: every ModelSpec.build() returns an unfitted scikit-learn Pipeline
that supports fit / predict / predict_proba. Training, evaluation and the API work with
any spec without knowing which one it is.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline

from jubii7.classifier import SEED
from jubii7.text import preprocess


@dataclass(frozen=True)
class ModelSpec(ABC):
    """A named, reproducible model configuration."""

    name: str
    analyzer: str = "word"
    ngram_range: tuple[int, int] = (1, 2)

    def vectorizer(self) -> TfidfVectorizer:
        return TfidfVectorizer(
            preprocessor=preprocess,
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            sublinear_tf=True,
        )

    def build(self) -> Pipeline:
        return Pipeline([("tfidf", self.vectorizer()), ("clf", self.classifier())])

    @abstractmethod
    def classifier(self) -> Any:
        """The estimator after TF-IDF. Must support predict_proba."""

    def params(self) -> dict[str, object]:
        """Flat parameters for experiment tracking."""
        return {"model": self.name, "analyzer": self.analyzer, "ngram_range": str(self.ngram_range)}


@dataclass(frozen=True)
class LogRegSpec(ModelSpec):
    """TF-IDF + logistic regression. C is the inverse regularisation strength."""

    C: float = 10.0

    def classifier(self) -> LogisticRegression:
        return LogisticRegression(C=self.C, max_iter=2000, class_weight="balanced", random_state=SEED)

    def params(self) -> dict[str, object]:
        return {**super().params(), "C": self.C}


@dataclass(frozen=True)
class SgdSpec(ModelSpec):
    """TF-IDF + logistic loss trained by SGD, so the learning rate is an explicit hyperparameter."""

    eta0: float = 0.1
    alpha: float = 1e-4
    epochs: int = 50
    extra: dict[str, Any] = field(default_factory=dict)

    def classifier(self) -> SGDClassifier:
        return SGDClassifier(
            loss="log_loss",
            learning_rate="constant",
            eta0=self.eta0,
            alpha=self.alpha,
            max_iter=self.epochs,
            tol=None,
            class_weight="balanced",
            random_state=SEED,
            **self.extra,
        )

    def params(self) -> dict[str, object]:
        return {**super().params(), "eta0": self.eta0, "alpha": self.alpha, "epochs": self.epochs}


CANDIDATES: tuple[ModelSpec, ...] = (
    LogRegSpec(name="word_tfidf_logreg", analyzer="word", ngram_range=(1, 2), C=10.0),
    LogRegSpec(name="char_tfidf_logreg", analyzer="char_wb", ngram_range=(2, 5), C=10.0),
    SgdSpec(name="char_tfidf_sgd", analyzer="char_wb", ngram_range=(2, 5), eta0=0.1),
)
