"""Figures for the evaluation report. Each function writes one PNG and returns its path."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: works in CI and servers

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from jubii7.classifier import LABELS

BLUE, ORANGE, GREY = "#2563eb", "#ea580c", "#6b7280"


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _band(ax: Axes, x: list[float], data: dict[str, list[float]], log_x: bool = False) -> None:
    for prefix, colour, label in (("train", BLUE, "train"), ("valid", ORANGE, "validation (CV)")):
        mean, std = np.array(data[f"{prefix}_mean"]), np.array(data[f"{prefix}_std"])
        ax.plot(x, mean, "o-", color=colour, label=label)
        ax.fill_between(x, mean - std, mean + std, color=colour, alpha=0.15)
    if log_x:
        ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("macro-F1")
    ax.grid(alpha=0.3)
    ax.legend()


def model_comparison(cv: Mapping[str, Mapping[str, float | str]], chosen: str, path: Path) -> Path:
    names = list(cv)
    means = [float(cv[n]["cv_f1_macro_mean"]) for n in names]
    stds = [float(cv[n]["cv_f1_macro_std"]) for n in names]
    colours = [ORANGE if n == chosen else GREY for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, means, yerr=stds, color=colours, capsize=6)
    for i, m in enumerate(means):
        ax.text(i, m + 0.02, f"{m:.3f}", ha="center")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("macro-F1 (5x5 repeated CV)")
    ax.set_title(f"Model comparison (chosen: {chosen})")
    return _save(fig, path)


def learning_curve(data: dict[str, list[float]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    _band(ax, data["train_rows"], data)
    ax.set_xlabel("training rows")
    ax.set_title("Learning curve: is more data worth it?")
    return _save(fig, path)


def regularisation_curve(data: dict[str, list[float]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    _band(ax, data["C"], data, log_x=True)
    ax.set_xlabel("C (inverse regularisation strength, log scale)")
    ax.set_title("Logistic regression: under- vs over-fitting")
    return _save(fig, path)


def learning_rate_curve(data: dict[str, list[float]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    _band(ax, data["eta0"], data, log_x=True)
    ax.set_xlabel("SGD learning rate eta0 (log scale)")
    ax.set_title("Learning-rate sweep")
    return _save(fig, path)


def epoch_curves(data: dict[float, dict[str, list[float]]], path: Path) -> Path:
    fig, (loss_ax, f1_ax) = plt.subplots(1, 2, figsize=(11, 4))
    for rate, series in data.items():
        epochs = range(1, len(series["train_loss"]) + 1)
        loss_ax.plot(epochs, series["train_loss"], label=f"eta0={rate:g}")
        f1_ax.plot(epochs, series["val_f1"], label=f"eta0={rate:g}")
    loss_ax.set(xlabel="epoch", ylabel="training log loss", title="Training loss per epoch")
    f1_ax.set(xlabel="epoch", ylabel="validation macro-F1", title="Validation F1 per epoch", ylim=(0, 1.05))
    for ax in (loss_ax, f1_ax):
        ax.grid(alpha=0.3)
        ax.legend()
    return _save(fig, path)


def per_class_f1(per_class: dict[str, dict[str, float]], path: Path) -> Path:
    metrics = ("precision", "recall", "f1-score")
    x = np.arange(len(LABELS))
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1) * 0.27, [per_class[label][metric] for label in LABELS], width=0.27, label=metric)
    ax.set_xticks(x, LABELS)
    ax.set_ylim(0, 1.1)
    ax.set_title("Per-class precision, recall and F1 (hold-out)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


def confusion(matrix: list[list[int]], path: Path) -> Path:
    data = np.array(matrix)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=35, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(
                j,
                i,
                data[i, j],
                ha="center",
                va="center",
                color="white" if data[i, j] > data.max() / 2 else "black",
            )
    ax.set(xlabel="predicted", ylabel="true", title="Confusion matrix (hold-out)")
    return _save(fig, path)


def cascade_comparison(result: dict[str, float], path: Path, subtitle: str = "") -> Path:
    groups = ("macro-F1", "automation rate", "accuracy when automatic")
    model = [result["model_f1"], result["model_automation"], result["model_automated_accuracy"]]
    cascade = [result["cascade_f1"], result["cascade_automation"], result["cascade_automated_accuracy"]]
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(8, 4))
    for offset, values, colour, label in (
        (-0.2, model, GREY, "model only"),
        (0.2, cascade, ORANGE, "rules -> model"),
    ):
        ax.bar(x + offset, values, width=0.4, color=colour, label=label)
        for xi, v in zip(x + offset, values, strict=True):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center")
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 1.12)
    ax.set_title(
        f"Rules -> model vs model, {subtitle}\n"
        f"rules cover {result['rule_coverage']:.0%} of lines at {result['rule_precision']:.0%} precision",
        fontsize=10,
    )
    ax.legend(loc="lower right")
    return _save(fig, path)


def cv_distribution(cv: dict[str, list[float]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(list(cv.values()), tick_labels=list(cv))
    ax.set_ylabel("macro-F1 per fold")
    ax.set_title("Score spread across 25 CV folds")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)
