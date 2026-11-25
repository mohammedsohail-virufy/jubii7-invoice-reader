# Testing and validation methods

Two different questions, two toolkits:

- **Is the code correct and safe?** Software testing and static analysis.
- **Is the model good, and will it hold up on new data?** Statistical validation.

Run everything with `python scripts/quality.py` (code) and `jubii7 train` (model). CI runs
the same gates on every push (`.github/workflows/ci.yml`).

## 1. Code: testing methods

| Method | Tool | What it proves here | Where |
|---|---|---|---|
| **Unit tests** | pytest | each function and rule does one thing right (digit conversion, IoU, every check) | `tests/test_text.py`, `test_invoice_checker.py` |
| **Parametrised tests** | `pytest.mark.parametrize` | the same assertion over many inputs (EN + AR lines, edge-case boxes, bad payloads) | throughout |
| **Golden / answer-key test** | pytest | the checker finds *exactly* the 14 expected issues on the challenge invoice, and totals recompute to AED 23,504.04 | `test_challenge_invoice_matches_answer_key` |
| **Negative test** | pytest | a fully corrected invoice produces no errors and no warnings (no false alarms) | `test_clean_invoice_passes` |
| **Property-based tests** | Hypothesis | rules that must hold for *any* input, with hundreds of generated cases: digit conversion is idempotent, Arabic and Western numbers parse the same, brackets always mean negative, IoU is symmetric and between 0 and 1 | `test_text.py`, `test_puzzle_and_track_c.py` |
| **Contract tests (Liskov substitution)** | pytest | every `Check`, every `ModelSpec` and every `LineClassifier` honours its interface, so any one can replace another | `test_every_check_honours_the_contract`, `test_every_model_spec_is_substitutable`, `test_cascade_is_itself_a_line_classifier` |
| **Fault-injection test** | pytest | a crashing rule is reported as an issue, not a 500 error | `test_a_crashing_check_is_reported_not_raised` |
| **Test doubles (fakes)** | dependency injection | the API is tested with a fake model, so tests are fast and don't need a trained file | `tests/test_api.py` |
| **API / integration tests** | FastAPI TestClient | endpoints, status codes, input validation (empty, too long, bad JSON → 422) | `test_api.py`, `test_dashboard.py` |
| **Security tests** | pytest | path traversal (`../../`, absolute paths, `.venv`, model files) is blocked; a tampered model file is refused | `test_code_browser_blocks_path_traversal...`, `test_tampered_model_is_refused` |
| **End-to-end test** | pytest | train on a temp copy of the data → model card → load → predict an Arabic line | `test_predictor_end_to_end` |
| **Model quality gate** | pytest | the trained model must beat the majority-class baseline by a wide margin | `test_model_beats_majority_baseline` |
| **Coverage** | pytest-cov | 90% of lines executed by tests; uncovered lines listed per file | `reports/quality/coverage.json` |

## 2. Code: static analysis and scanning

| Gate | Tool | Catches |
|---|---|---|
| Lint | **ruff** (pyflakes, pycodestyle, bugbear, pyupgrade, naming, simplify, pylint subset, security rules) | bugs, dead code, unsafe patterns, style |
| Complexity limit | ruff `C901` (max 8) + **radon** CC / maintainability index | functions that are too branchy to test well |
| Formatting | `ruff format --check` | style drift |
| Types | **mypy --strict** | type errors before runtime |
| SAST (security) | **bandit** | insecure code: pickle loading, subprocess, XML parsing, etc. Every suppression has a written reason |
| SCA (dependencies) | **pip-audit** | known CVEs in installed packages |
| Supply chain on GitHub | **CodeQL** + Dependabot (in CI) | deeper data-flow security analysis and dependency updates |
| Pre-commit | `.pre-commit-config.yaml` | runs ruff, formatting and basic hygiene before each commit |

## 3. Model: validation methods

| Method | Why |
|---|---|
| **Stratified k-fold cross-validation (5 folds)** | every row is used for both training and validation; stratification keeps all 5 classes in every fold |
| **Repeated k-fold (5 × 5 = 25 fits)** | with 57 rows one split is noisy; repeating gives a mean **and a standard deviation** (0.846 ± 0.109) |
| **Model selection by CV, not by the test split** | the 18-row hold-out is too small to choose on; it only gives per-class detail |
| **Stratified hold-out (30%)** | per-class precision/recall/F1, confusion matrix, log loss |
| **Unseen challenge set (out-of-distribution)** | 30 lines with new wording, labelled from the guide only, never used to train or write rules. The honest number |
| **Baseline comparison** | majority class ≈ 0.07 macro-F1; models must beat it clearly |
| **Learning curve** | F1 vs training size: tells whether more data or a better model is the next step |
| **Validation curve (regularisation C)** | under- vs over-fitting for logistic regression |
| **Learning-rate sweep + per-epoch curves (SGD)** | stable learning-rate range and convergence speed |
| **Error analysis** | every CV mistake with its confidence, explained in MODEL_REPORT.md |
| **Leakage checks** | duplicate texts rejected at load time; rules evaluated on data they were not written from |
| **Business metrics** | automation rate and accuracy on automatic decisions, not only F1 |
| **Experiment tracking** | every run's params, metrics and model logged (MLflow locally; SageMaker managed MLflow on AWS) |
| **Model card + checksum** | what was trained, on what, with which scores; SHA-256 checked before loading |

Metric choice: **macro-F1** weights all 5 classes equally (a rare class matters as much as a
common one). For finance, **recall on errors** matters more than precision: a missed wrong
invoice costs money, a false flag costs a reviewer a minute (see Track C).
