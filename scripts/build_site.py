"""Build the static GitHub Pages site into site/.

    python scripts/build_site.py

The live parts (invoice check, line classifier) run this package's own Python code in the
browser with Pyodide; everything else is the same dashboard, fed by pre-built JSON.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from jubii7.web.dashboard import CodeBrowser, ReportReader

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
PACKAGE = ROOT / "src" / "jubii7"

# Pure-Python modules the browser needs (no scikit-learn, pandas, FastAPI or joblib).
BROWSER_MODULES = (
    "__init__.py",
    "text.py",
    "scan.py",
    "invoice/**/*.py",
    "classifier/__init__.py",
    "classifier/rules.py",
    "classifier/cascade.py",
    "classifier/predictor.py",
    "classifier/portable.py",
)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def python_bundle() -> dict[str, str]:
    files: dict[str, str] = {}
    for pattern in BROWSER_MODULES:
        for path in sorted(PACKAGE.glob(pattern)):
            files[f"jubii7/{path.relative_to(PACKAGE).as_posix()}"] = path.read_text(encoding="utf-8")
    return files


def main() -> None:
    portable = ROOT / "models" / "model_portable.json"
    if not portable.exists():
        raise SystemExit("models/model_portable.json missing: run `jubii7 train` first")

    SITE.mkdir(exist_ok=True)
    for child in SITE.iterdir():  # empty it in place: the folder may be open in a local preview server
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    (SITE / ".nojekyll").touch()  # serve files as-is

    html = (PACKAGE / "web" / "static" / "index.html").read_text(encoding="utf-8")
    html = html.replace(
        "<script>\nconst $ =", "<script>window.JUBII7_STATIC = true;</script>\n<script>\nconst $ =", 1
    )
    (SITE / "index.html").write_text(html, encoding="utf-8")

    shutil.copytree(ROOT / "reports" / "figures", SITE / "figures")
    shutil.copy(ROOT / "data" / "original_invoice_photo.jpg", SITE / "sample-scan.jpg")
    write_json(SITE / "data" / "dashboard.json", ReportReader(ROOT).payload())
    write_json(
        SITE / "data" / "invoice_original.json",
        json.loads((ROOT / "data/invoice_WB-2026-11-0457.json").read_text(encoding="utf-8")),
    )
    write_json(
        SITE / "data" / "invoice_corrected.json",
        json.loads((ROOT / "data/invoice_WB-2026-11-0457_corrected.json").read_text(encoding="utf-8")),
    )
    browser = CodeBrowser(ROOT)
    write_json(SITE / "data" / "files.json", {path: browser.read(path) for path in browser.files()})
    write_json(SITE / "py" / "bundle.json", python_bundle())
    shutil.copy(portable, SITE / "py" / "model_portable.json")

    size = sum(p.stat().st_size for p in SITE.rglob("*") if p.is_file()) / 1e6
    print(f"site/ built: {sum(1 for p in SITE.rglob('*') if p.is_file())} files, {size:.1f} MB")


if __name__ == "__main__":
    main()
