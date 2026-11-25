"""Read-only dashboard routes. Everything shown comes from files the pipeline already wrote.

Security: the code browser only serves whitelisted folders and file types, resolves paths
and rejects anything outside the project (path traversal), and never executes anything.
"""

from __future__ import annotations

import json

# Only parses the junit file written by our own pytest run (trusted input).
import xml.etree.ElementTree as ET  # nosec B405
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

STATIC = Path(__file__).parent / "static"
BROWSABLE_DIRS = ("src", "tests", "scripts", "docs", "data", ".github")
BROWSABLE_FILES = ("README.md", "pyproject.toml", "Dockerfile", ".pre-commit-config.yaml")
BROWSABLE_SUFFIXES = {".py", ".md", ".json", ".csv", ".toml", ".yml", ".yaml", ".txt", ".html", ".css", ".js"}
MAX_FILE_BYTES = 400_000


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


class ReportReader:
    """Turns the files under reports/ into one JSON payload for the dashboard."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.reports = root / "reports"
        self.quality = self.reports / "quality"

    def payload(self) -> dict[str, object]:
        return {
            "model_card": _read_json(self.root / "models" / "model_card.json"),
            "evaluation": _read_json(self.reports / "evaluation.json"),
            "challenge_rows": self._csv(self.reports / "challenge_predictions.csv"),
            "mistakes": self._csv(self.reports / "mistakes_cv.csv"),
            "figures": sorted(p.name for p in (self.reports / "figures").glob("*.png")),
            "quality": self.quality_payload(),
        }

    def quality_payload(self) -> dict[str, object]:
        return {
            "summary": _read_json(self.quality / "summary.json"),
            "tests": self._junit(),
            "coverage": self._coverage(),
            "lint": _safe_json(_read_text(self.quality / "lint.out")),
            "types": _read_text(self.quality / "types.out").strip(),
            "security": _safe_json(_read_text(self.quality / "security.out")),
            "deps": _safe_json(_read_text(self.quality / "deps.out")),
            "complexity": _safe_json(_read_text(self.quality / "complexity.out")),
            "maintainability": _safe_json(_read_text(self.quality / "maintainability.out")),
        }

    def _junit(self) -> list[dict[str, object]]:
        path = self.quality / "junit.xml"
        if not path.exists():
            return []
        tree = ET.parse(path)  # noqa: S314  # nosec B314
        cases: list[dict[str, object]] = []
        for case in tree.iter("testcase"):
            failed = case.find("failure") is not None or case.find("error") is not None
            cases.append(
                {
                    "file": (case.get("classname") or "").replace("tests.", ""),
                    "name": case.get("name"),
                    "seconds": float(case.get("time") or 0),
                    "status": "failed"
                    if failed
                    else ("skipped" if case.find("skipped") is not None else "passed"),
                }
            )
        return cases

    def _coverage(self) -> dict[str, object] | None:
        data = _read_json(self.quality / "coverage.json")
        if not data:
            return None
        files = {
            name.replace("\\", "/"): round(info["summary"]["percent_covered"], 1)
            for name, info in data["files"].items()
        }
        return {"total": round(data["totals"]["percent_covered"], 1), "files": files}

    @staticmethod
    def _csv(path: Path) -> list[dict[str, str]]:
        import csv

        if not path.exists():
            return []
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


def _safe_json(text: str) -> Any:
    try:
        return json.loads(text) if text.strip() else None
    except json.JSONDecodeError:
        return {"raw": text}


class CodeBrowser:
    """Lists and serves project files from a whitelist, with path-traversal protection."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def files(self) -> list[str]:
        found = [name for name in BROWSABLE_FILES if (self.root / name).is_file()]
        for folder in BROWSABLE_DIRS:
            base = self.root / folder
            if base.is_dir():
                found += [
                    p.relative_to(self.root).as_posix()
                    for p in sorted(base.rglob("*"))
                    if p.is_file() and p.suffix in BROWSABLE_SUFFIXES and "__pycache__" not in p.parts
                ]
        return found

    def read(self, relative: str) -> str:
        path = (self.root / relative).resolve()
        allowed = relative in self.files()
        if not allowed or not path.is_relative_to(self.root) or path.stat().st_size > MAX_FILE_BYTES:
            raise HTTPException(404, "file not available")
        return path.read_text(encoding="utf-8", errors="replace")


def mount_dashboard(app: FastAPI, root: Path) -> None:
    reader, browser = ReportReader(root), CodeBrowser(root)
    router = APIRouter(prefix="/api", tags=["dashboard"])

    @router.get("/dashboard")
    def dashboard() -> dict[str, object]:
        return reader.payload()

    @router.get("/sample-invoice")
    def sample_invoice(variant: str = "original") -> Any:
        name = (
            "invoice_WB-2026-11-0457_corrected.json"
            if variant == "corrected"
            else "invoice_WB-2026-11-0457.json"
        )
        return _read_json(root / "data" / name)

    @router.get("/files")
    def files() -> list[str]:
        return browser.files()

    @router.get("/file")
    def file(path: str) -> dict[str, str]:
        return {"path": path, "content": browser.read(path)}

    app.include_router(router)
    # check_dir=False: the app can start before the first training run creates the folder.
    app.mount(
        "/figures", StaticFiles(directory=root / "reports" / "figures", check_dir=False), name="figures"
    )

    @app.get("/sample-scan.png", include_in_schema=False)
    def sample_scan() -> FileResponse:
        return FileResponse(root / "data" / "sample_scan.png")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")
