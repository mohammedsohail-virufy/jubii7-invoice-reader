"""Experiment tracking behind one interface.

Dependency inversion + Liskov: training talks to `ExperimentTracker`. MLflow is used
when installed; the local JSON tracker is a drop-in substitute (CI, laptops without MLflow).
"""

from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExperimentTracker(ABC):
    @abstractmethod
    def log_run(self, name: str, params: dict[str, object], metrics: dict[str, float], model: Any) -> str:
        """Record one run and return its id."""


class JsonTracker(ExperimentTracker):
    """Appends runs to a JSON file. No server, no extra dependency."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def log_run(self, name: str, params: dict[str, object], metrics: dict[str, float], model: Any) -> str:
        runs = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []
        run_id = uuid.uuid4().hex
        runs.append({"run_id": run_id, "name": name, "params": params, "metrics": metrics})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
        return run_id


class MlflowTracker(ExperimentTracker):
    """MLflow tracking. Any store URI works (local folder, SQLite, or a remote server)."""

    def __init__(self, tracking_uri: str, experiment: str) -> None:
        import mlflow  # imported lazily so MLflow stays optional

        self._mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

    def log_run(self, name: str, params: dict[str, object], metrics: dict[str, float], model: Any) -> str:
        with self._mlflow.start_run(run_name=name) as run:
            self._mlflow.log_params(params)
            self._mlflow.log_metrics(metrics)
            # MLflow 3 serialises with skops (safer than pickle): trust only our own preprocessor.
            self._mlflow.sklearn.log_model(
                model, name="model", skops_trusted_types=["jubii7.text.preprocess"]
            )
            return str(run.info.run_id)


def make_tracker(root: Path, experiment: str = "jubii7-invoice-line-category") -> ExperimentTracker:
    """MLflow (local SQLite: `mlflow ui --backend-store-uri sqlite:///mlflow.db`) or the JSON fallback.

    On AWS the same interface is implemented by SageMaker managed MLflow (pass its tracking
    server ARN as the URI): see docs/ARCHITECTURE.md.
    """
    try:
        return MlflowTracker(f"sqlite:///{(root / 'mlflow.db').resolve().as_posix()}", experiment)
    except Exception as error:  # MLflow missing or misconfigured: training must still work
        logger.warning("MLflow unavailable (%s); logging runs to JSON instead", error)
        return JsonTracker(root / "reports" / "runs_local.json")
