from __future__ import annotations

import pytest

from jubii7.classifier.cascade import Cascade, LineClassifier, RuleClassifier
from jubii7.classifier.predictor import Prediction
from jubii7.classifier.rules import KeywordRules


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("Rebar 16 mm", "steel"),
        ("حديد تسليح ١٦ مم", "steel"),
        ("Washed sand", "aggregates"),
        ("رمل مغسول", "aggregates"),
        ("Hollow block 15 cm", "blocks"),
        ("أسمنت بورتلاند ٥٠ كجم", "cement"),
        ("Delivery of cement bags to site", "services"),  # guide tie-break: bill = service
        ("توصيل أكياس أسمنت للموقع", "services"),
        ("Concrete pump hire per day", "services"),
    ],
)
def test_rules(text: str, category: str) -> None:
    hit = KeywordRules().match(text)
    assert hit is not None
    assert hit.category == category


@pytest.mark.parametrize("text", ["Waterproofing membrane", "رول عزل مائي", "Item 47", "cement block"])
def test_rules_abstain_when_unsure(text: str) -> None:
    assert KeywordRules().match(text) is None  # unknown or conflicting -> let the model decide


class Fixed(LineClassifier):
    def __init__(self, category: str, unsure: bool) -> None:
        self.category, self.unsure = category, unsure
        self.calls = 0

    def predict(self, text: str) -> Prediction:
        self.calls += 1
        return Prediction(
            text, text, self.category, 0.4 if self.unsure else 0.9, self.unsure, [], source=self.category
        )


def test_cascade_stops_at_first_confident_stage() -> None:
    first, second = Fixed("a", unsure=False), Fixed("b", unsure=False)
    assert Cascade([first, second]).predict("x").category == "a"
    assert second.calls == 0


def test_cascade_falls_through_and_keeps_review_flag() -> None:
    result = Cascade([Fixed("a", unsure=True), Fixed("b", unsure=True)]).predict("x")
    assert (result.category, result.needs_review) == ("b", True)


def test_cascade_is_itself_a_line_classifier() -> None:
    inner = Cascade([RuleClassifier()])
    outer = Cascade([inner, Fixed("model", unsure=False)])  # composite: cascades nest
    assert outer.predict("Rebar 12 mm").category == "steel"
    assert outer.predict("Item 47").source == "model"


def test_empty_cascade_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Cascade([])
