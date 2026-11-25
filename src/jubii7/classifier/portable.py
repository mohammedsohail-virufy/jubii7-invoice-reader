"""Portable model: TF-IDF + logistic regression as plain JSON weights, scored in pure Python.

Why: the GitHub Pages demo runs this package in the browser (Pyodide), where pickled
scikit-learn models are not portable across versions. Exporting the learned weights gives
a dependency-free model that returns the same probabilities (tests/test_portable.py checks
parity with scikit-learn to 1e-9 on every labelled row).

PortableModel has `classes_` and `predict_proba`, so it plugs into Predictor and the
cascade unchanged (Liskov substitution).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from jubii7.text import preprocess

_TOKEN = re.compile(r"(?u)\b\w\w+\b")  # scikit-learn's default token_pattern


def export_model(pipeline: Any) -> dict[str, object]:
    """Pipeline(tfidf, LogisticRegression) -> JSON-serialisable weights."""
    tfidf, clf = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    if type(clf).__name__ != "LogisticRegression":
        raise ValueError("only logistic regression pipelines can be exported")
    if tfidf.analyzer not in ("word", "char_wb") or not tfidf.sublinear_tf or tfidf.norm != "l2":
        raise ValueError("unsupported TF-IDF configuration")
    return {
        "format": "jubii7-portable-tfidf-logreg/1",
        "analyzer": tfidf.analyzer,
        "ngram_range": list(tfidf.ngram_range),
        "vocabulary": {term: int(i) for term, i in tfidf.vocabulary_.items()},
        "idf": [float(v) for v in tfidf.idf_],
        "classes": [str(c) for c in clf.classes_],
        "coef": [[float(v) for v in row] for row in clf.coef_],
        "intercept": [float(v) for v in clf.intercept_],
    }


class PortableModel:
    def __init__(self, weights: dict[str, Any]) -> None:
        self.analyzer: str = weights["analyzer"]
        self.min_n, self.max_n = weights["ngram_range"]
        self.vocabulary: dict[str, int] = weights["vocabulary"]
        self.idf: list[float] = weights["idf"]
        self.classes_: list[str] = weights["classes"]
        self.coef: list[list[float]] = weights["coef"]
        self.intercept: list[float] = weights["intercept"]

    # ---- the same steps as TfidfVectorizer(sublinear_tf=True, norm="l2") ----
    def _terms(self, text: str) -> list[str]:
        doc = preprocess(text)
        return self._word_ngrams(doc) if self.analyzer == "word" else self._char_wb_ngrams(doc)

    def _word_ngrams(self, doc: str) -> list[str]:
        tokens = _TOKEN.findall(doc)
        return [
            " ".join(tokens[i : i + n])
            for n in range(self.min_n, self.max_n + 1)
            for i in range(len(tokens) - n + 1)
        ]

    def _char_wb_ngrams(self, doc: str) -> list[str]:
        """Same loop as scikit-learn: a word shorter than n is counted once, whole."""
        grams = []
        for word in doc.split():
            padded = f" {word} "
            for n in range(self.min_n, self.max_n + 1):
                offset = 0
                grams.append(padded[offset : offset + n])
                while offset + n < len(padded):
                    offset += 1
                    grams.append(padded[offset : offset + n])
                if offset == 0:
                    break
        return grams

    def _vector(self, text: str) -> dict[int, float]:
        counts = Counter(t for t in self._terms(text) if t in self.vocabulary)
        weights = {
            self.vocabulary[t]: (1 + math.log(c)) * self.idf[self.vocabulary[t]] for t, c in counts.items()
        }
        norm = math.sqrt(sum(v * v for v in weights.values()))
        return {i: v / norm for i, v in weights.items()} if norm else {}

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            x = self._vector(text)
            scores = [
                b + sum(row[i] * v for i, v in x.items())
                for row, b in zip(self.coef, self.intercept, strict=True)
            ]
            top = max(scores)
            exps = [math.exp(s - top) for s in scores]  # softmax (multinomial logistic regression)
            total = sum(exps)
            out.append([e / total for e in exps])
        return out
