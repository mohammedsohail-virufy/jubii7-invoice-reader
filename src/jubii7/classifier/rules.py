"""Deterministic bilingual rules, taken from data/labelling_guide.md (not from the training rows).

Why rules before the model: finance teams need decisions they can audit. A rule says
*why* ("matched 'rebar'"), costs nothing to run and never drifts. The model handles what
the rules don't cover, and people handle what neither is sure about.

Guide rule for ties: label what is billed, so service words win over material words
("delivery of cement" -> services).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from jubii7.text import preprocess

# Service words are checked first: the guide's tie-break rule.
SERVICE_TERMS = (
    "delivery",
    "transport",
    "hire",
    "rental",
    "labour",
    "labor",
    "unloading",
    "test",
    "lab charges",
    "waste",
    "توصيل",
    "نقل",
    "إيجار",
    "ايجار",
    "أجور",
    "اجور",
    "عمال",
    "فحص",
    "رسوم",
)
MATERIAL_TERMS: dict[str, tuple[str, ...]] = {
    "cement": ("cement", "opc", "portland", "src", "grout", "أسمنت", "اسمنت", "بورتلاند"),
    "steel": (
        "rebar",
        "steel",
        "binding wire",
        "mesh",
        "i-beam",
        "anchor bolt",
        "حديد",
        "تسليح",
        "كمرة",
        "سلك ربط",
    ),
    "aggregates": (
        "sand",
        "gravel",
        "gabbro",
        "crushed",
        "road base",
        "aggregate",
        "رمل",
        "حصى",
        "كسارة",
        "جابرو",
    ),
    "blocks": ("block", "paver", "interlock", "kerb", "brick", "بلوك", "طابوق", "انترلوك", "بردورة"),
}


@dataclass(frozen=True)
class RuleHit:
    category: str
    matched: str


def _pattern(terms: tuple[str, ...]) -> re.Pattern[str]:
    # Word-start boundary so "src" doesn't fire inside other words; Arabic prefixes like "ال" allowed.
    alternatives = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    return re.compile(rf"(?:^|[\s\-/(]|ال)({alternatives})")


class KeywordRules:
    """Returns a hit only when exactly one category matches; ambiguity goes to the model."""

    def __init__(
        self,
        service_terms: tuple[str, ...] = SERVICE_TERMS,
        material_terms: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        self._services = _pattern(service_terms)
        self._materials = {cat: _pattern(terms) for cat, terms in (material_terms or MATERIAL_TERMS).items()}

    def match(self, text: str) -> RuleHit | None:
        normalised = preprocess(text)
        service = self._services.search(normalised)
        if service:
            return RuleHit("services", service.group(1))
        hits = [(cat, m.group(1)) for cat, p in self._materials.items() if (m := p.search(normalised))]
        if len(hits) == 1:
            return RuleHit(*hits[0])
        return None  # no match, or conflicting materials (e.g. "concrete block" vs "cement")
