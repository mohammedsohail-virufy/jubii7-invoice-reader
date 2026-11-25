"""A cascade of line classifiers: rules -> ML model -> (optional LLM) -> human review.

Every stage implements LineClassifier, so stages are interchangeable (Liskov) and the
cascade itself is also a LineClassifier (composite pattern). Adding an LLM or a
multilingual-embedding model is one new class, with no change to the API or the tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from jubii7.classifier.predictor import Prediction, Predictor
from jubii7.classifier.rules import KeywordRules
from jubii7.text import preprocess

RULE_CONFIDENCE = 0.97  # measured rule precision is reported in reports/evaluation.json


class LineClassifier(ABC):
    @abstractmethod
    def predict(self, text: str) -> Prediction:
        """Classify one line. needs_review=True means 'I am not sure, pass it on'."""


class RuleClassifier(LineClassifier):
    def __init__(self, rules: KeywordRules | None = None) -> None:
        self.rules = rules or KeywordRules()

    def predict(self, text: str) -> Prediction:
        hit = self.rules.match(text)
        if hit is None:
            return Prediction(
                text, preprocess(text), "unknown", 0.0, True, [], source="rules", reason="no rule matched"
            )
        return Prediction(
            text,
            preprocess(text),
            hit.category,
            RULE_CONFIDENCE,
            False,
            [{"category": hit.category, "p": RULE_CONFIDENCE}],
            source="rules",
            reason=f"matched '{hit.matched}'",
        )


class ModelClassifier(LineClassifier):
    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor

    def predict(self, text: str) -> Prediction:
        return self.predictor.predict(text)


class Cascade(LineClassifier):
    """Returns the first confident answer; otherwise the last stage's answer, flagged for review."""

    def __init__(self, stages: Sequence[LineClassifier]) -> None:
        if not stages:
            raise ValueError("a cascade needs at least one stage")
        self.stages = tuple(stages)

    def predict(self, text: str) -> Prediction:
        prediction = self.stages[0].predict(text)
        for stage in self.stages[1:]:
            if not prediction.needs_review:
                break
            prediction = stage.predict(text)
        return prediction
