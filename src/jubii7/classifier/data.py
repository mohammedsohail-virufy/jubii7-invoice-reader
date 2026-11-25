"""Load and validate the labelled line-item dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from jubii7.classifier import LABELS, UNMAPPED

REQUIRED_COLUMNS = ("text", "label", "source")


class DatasetError(ValueError):
    """The labelled CSV is not usable for training."""


@dataclass(frozen=True)
class Dataset:
    frame: pd.DataFrame
    excluded: pd.DataFrame

    @property
    def texts(self) -> pd.Series:
        return self.frame["text"]

    @property
    def labels(self) -> pd.Series:
        return self.frame["label"]

    def summary(self) -> dict[str, object]:
        return {
            "rows": len(self.frame),
            "excluded_unmapped": len(self.excluded),
            "by_label": self.frame["label"].value_counts().to_dict(),
            "by_source": self.frame["source"].value_counts().to_dict(),
            "arabic_rows": int(self.frame["text"].str.contains(r"[؀-ۿ]").sum()),
        }


def load_dataset(path: str | Path) -> Dataset:
    """Read lines.csv, drop out-of-taxonomy rows, and fail loudly on bad labels or duplicates."""
    frame = pd.read_csv(path, encoding="utf-8")
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise DatasetError(f"missing columns: {sorted(missing)}")

    frame["text"] = frame["text"].astype(str).str.strip()
    excluded = frame[frame["label"] == UNMAPPED]
    frame = frame[frame["label"] != UNMAPPED].reset_index(drop=True)

    unknown = set(frame["label"]) - set(LABELS)
    if unknown:
        raise DatasetError(f"unknown labels: {sorted(unknown)}")
    duplicated = frame[frame["text"].duplicated()]["text"].tolist()
    if duplicated:
        raise DatasetError(f"duplicate texts would leak between train and test: {duplicated}")
    return Dataset(frame=frame, excluded=excluded)
