"""Run every quality gate and save machine-readable results to reports/quality/.

    python scripts/quality.py

Gates (the same ones CI runs):
  tests      pytest + coverage           correctness, regression
  lint       ruff                        style, bugs, complexity, security rules
  types      mypy --strict               type errors before runtime
  security   bandit                      insecure code patterns (SAST)
  deps       pip-audit                   known CVEs in dependencies (SCA)
  complexity radon                       cyclomatic complexity + maintainability index
"""

from __future__ import annotations

import json
import shutil

# Runs a fixed list of developer tools; no user input reaches the command line.
import subprocess  # nosec B404
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "quality"
BIN = Path(sys.executable).parent


def tool(name: str) -> str:
    found = shutil.which(name, path=str(BIN)) or shutil.which(name)
    if not found:
        raise SystemExit(f"{name} not installed: pip install -e .[dev]")
    return found


GATES: dict[str, list[str]] = {
    "tests": [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        f"--junitxml={OUT / 'junit.xml'}",
        "--cov=jubii7",
        f"--cov-report=json:{OUT / 'coverage.json'}",
        "--cov-report=term",
    ],
    "lint": [tool("ruff"), "check", "src", "tests", "scripts", "--output-format=json"],
    "format": [tool("ruff"), "format", "--check", "src", "tests", "scripts"],
    "types": [tool("mypy"), "src"],
    "security": [tool("bandit"), "-r", "src", "-f", "json", "-q"],
    "deps": [tool("pip-audit"), "--format=json", "--progress-spinner=off"],
    "complexity": [tool("radon"), "cc", "src", "-j", "-a"],
    "maintainability": [tool("radon"), "mi", "src", "-j"],
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, command in GATES.items():
        started = time.perf_counter()
        # Fixed tool list defined above, no user input: safe to run (bandit B603 / ruff S603).
        result = subprocess.run(  # noqa: S603  # nosec B603
            command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=False
        )
        seconds = round(time.perf_counter() - started, 1)
        (OUT / f"{name}.out").write_text(result.stdout, encoding="utf-8")
        (OUT / f"{name}.err").write_text(result.stderr, encoding="utf-8")
        summary[name] = {"passed": result.returncode == 0, "exit_code": result.returncode, "seconds": seconds}
        print(f"{'PASS' if result.returncode == 0 else 'FAIL'}  {name:16} {seconds:6.1f}s")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0 if all(g["passed"] for g in summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
