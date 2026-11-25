"""Data profiling: where Arabic digits appear, which fields are empty, which were flagged for review."""

from __future__ import annotations

from dataclasses import dataclass, field

from jubii7.text import has_arabic_digits, to_western


@dataclass
class Profile:
    fields_with_arabic_digits: list[str] = field(default_factory=list)
    missing_values: list[str] = field(default_factory=list)
    flagged_for_verification: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "fields_with_arabic_digits": self.fields_with_arabic_digits,
            "missing_values": self.missing_values,
            "flagged_for_verification": self.flagged_for_verification,
        }


def profile(raw: object) -> Profile:
    """Walk the raw JSON once and record what a data engineer needs to know before parsing."""
    result = Profile()
    _walk(raw, "", result)
    return result


def _walk_dict(node: dict[str, object], path: str, result: Profile) -> None:
    for key, value in node.items():
        child = f"{path}.{key}" if path else key
        if key.startswith("_verify"):
            result.flagged_for_verification.append(child.replace("_verify_", ""))
        elif not key.startswith("_"):  # other "_" keys are extraction notes, not data
            _walk(value, child, result)


def _walk(node: object, path: str, result: Profile) -> None:
    if isinstance(node, dict):
        _walk_dict(node, path, result)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _walk(value, f"{path}[{index}]", result)
    elif node is None:
        result.missing_values.append(path)
    elif isinstance(node, str) and has_arabic_digits(node):
        result.fields_with_arabic_digits.append(f"{path} = {node!r} -> {to_western(node)!r}")
