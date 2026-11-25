# jubii.7 Live Challenge · Track A: Read the messy invoice

**Live demo: [https://mohammedsohail-virufy.github.io/jubii7-invoice-reader/](https://mohammedsohail-virufy.github.io/jubii7-invoice-reader/)** (works on mobile). Scan or upload an invoice, check it, classify lines, and review the tests, scans and model curves. Try the original challenge photo: [https://mohammedsohail-virufy.github.io/jubii7-invoice-reader/?demo#scan](https://mohammedsohail-virufy.github.io/jubii7-invoice-reader/?demo#scan)

Validation and classification core for a document-scanning pipeline. It takes a bilingual
(Arabic/English) building-materials tax invoice, already extracted to JSON, and:

0. **scans** a photo or PDF: on-device OCR (Arabic + English), then line maths, dates, TRNs, VAT and line categories from the text
1. **profiles** it: Arabic-Indic digits, gaps, and fields flagged to verify on the scan
2. **checks** it: 12 business rules, recomputing every total from the line items
3. **classifies** each line item: bilingual rules first, then an ML model; unsure lines go to a person
4. **serves** all of this as an API, plus a local dashboard to review results, tests, scans, curves and code

```
scan ─► OCR/layout ─► JSON ─► profile ─► check (12 rules) ─► classify lines (rules → model) ─► auto-post or human review
                              └──────────────────── this repo ────────────────────────────┘
```

## Results

| | |
|---|---|
| **Invoice** | 10 errors, 3 warnings, 1 info. Correct total **AED 23,504.04** vs printed 23,606.30: the customer was overcharged **AED 102.26** |
| **Root causes** | sand line 24 × 65 = 1,560 (printed 1,650); VAT charged on the pre-discount total; words ≠ figure; Cash + Credit both ticked; 45 vs ٣٠ days; 31/11 is not a date |
| **Model** (5×5 repeated stratified k-fold) | macro-F1 **0.846 ± 0.109** (word TF-IDF + logistic regression, best of 3) |
| **Unseen challenge set** (30 new lines) | model alone 0.655 → **rules → model 0.754**; lines decided automatically 23% → **63%** at 95% accuracy |
| **Quality gates** | 111 tests pass · 90% coverage · ruff, mypy --strict, bandit, pip-audit (0 CVEs), radon: all pass |
| **Puzzle** | brute force over 5!³ assignments, exactly one survives: **Chen is the chef, Dev lives in Abu Dhabi** |

## Quick start (about 3 minutes)

```bash
python -m venv .venv && .venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev,tracking]"

jubii7 puzzle                      # warm-up: unique answer
jubii7 check                       # profile + check the challenge invoice (add --strict for exit code 2 on errors)
jubii7 train                       # compare 3 models, k-fold CV, curves, challenge set, MLflow runs, model card
jubii7 predict "حديد تسليح ١٦ مم"    # rules → model cascade for one line
python scripts/quality.py          # every quality gate, results in reports/quality/
jubii7 serve                       # dashboard: http://127.0.0.1:8000   API docs: /docs
mlflow ui --backend-store-uri sqlite:///mlflow.db    # optional: compare runs in MLflow
```

Docker: `docker build -t jubii7 . && docker run -p 8000:8000 jubii7`

Static site: `python scripts/build_site.py` writes `site/` (deployed by `.github/workflows/pages.yml`).

## Dashboard

`jubii7 serve` opens the review dashboard locally; the same page is published to GitHub Pages, where the checker, classifier and scan analysis run as this package's Python code inside the browser (Pyodide), with no server.

| Tab | What you can do |
|---|---|
| Overview | pipeline, headline numbers, quality-gate status |
| Scan document | phone camera or upload (image / PDF), on-device OCR, editable text, line items with qty × price = amount, dates, TRNs, VAT, categories |
| Invoice check | edit the extracted JSON, run the checker live, see issues and printed vs recomputed totals |
| Line classifier | classify any EN/AR line and see the decision path: rules → model → human review |
| Model evaluation | candidate runs, interactive learning / learning-rate / epoch / regularisation curves, challenge set row by row, CV mistakes, all figures |
| Testing & quality | every test, coverage by file, lint, types, security scan, dependency CVEs, complexity |
| Code · Docs | read-only code browser (path-traversal protected) and these docs, including the findings log |

## API

| Method | Path | |
|---|---|---|
| GET | `/health` | liveness + version |
| POST | `/predict` | `{"text": "..."}` → category, confidence, `needs_review`, `source` (rules/model), `reason`, top 3 |
| POST | `/predict/batch` | up to 200 lines |
| POST | `/scan/analyse` | raw OCR text → line items (maths checked, classified), dates, TRNs, VAT issues |
| POST | `/invoice/check` | extracted invoice JSON → status, issues by severity, corrected totals, profile |

Input is validated (1–500 characters per line; 422 on bad JSON). The model file is loaded only
after its SHA-256 matches the model card.

## Repository

```
src/jubii7/
  text.py                 Arabic/Western digit and number normalisation (pure functions)
  invoice/                models · loader · profiling · totals · checker
    checks/               one class per business rule (arithmetic, tax, consistency, delivery)
  classifier/             data · models (specs) · rules · cascade · evaluate · plots · tracking · train · predictor
  api/app.py              FastAPI, dependency-injected
  web/                    dashboard routes + single-page UI
  scan.py                 OCR text -> line items, maths, dates, TRNs, VAT (pure Python)
  classifier/portable.py  model as JSON weights, scored in pure Python (browser build)
  puzzle/solver.py        warm-up
  track_c/                IoU and precision/recall/F1 as tested code
tests/                    111 tests: unit, property-based, contract (Liskov), API, security, end-to-end training
scripts/quality.py        runs every quality gate
data/                     invoice JSON (+ corrected copy), labelled lines, labelling guide, unseen challenge set
reports/                  evaluation.json, figures/, challenge predictions, CV mistakes, quality/
models/                   model.joblib + model_card.json (metrics, data summary, SHA-256)
docs/                     ARCHITECTURE · TESTING · MODEL_REPORT · FINDINGS · TRACK_C · PROMPTS
```

## Design in one paragraph

Each business rule is a small `Check` class with one method, `run(invoice) -> list[Issue]`;
the checker depends only on that abstraction, so rules are added or swapped per client
without editing it (single responsibility, open/closed, dependency inversion). The same idea
runs through the ML side: every `ModelSpec` builds a pipeline with the same interface, every
`LineClassifier` (rules, model, the cascade itself) returns the same `Prediction`, and every
`ExperimentTracker` (MLflow, local JSON, or SageMaker on AWS) logs runs the same way. Contract
tests check that any implementation can replace another (Liskov substitution). Details:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Known limits

- Extraction from the photo was AI-assisted and checked by hand; production needs OCR with
  Arabic support plus table-structure detection (see the AWS design).
- 57 labelled rows: the metrics show the method, not production accuracy. The learning curve
  is still rising, so more labelled lines are the biggest lever.
- The 5-class taxonomy has no home for the waterproofing membrane; it is kept out and routed
  to review (proposal: add `finishes / waterproofing`).
- Amount-in-words was transcribed by hand; production needs an Arabic number-words parser.

More: [testing methods](docs/TESTING.md) · [model report and improvement plan](docs/MODEL_REPORT.md) ·
[findings during the build](docs/FINDINGS.md) · [Track C answers](docs/TRACK_C.md) · [AI prompts used](docs/PROMPTS.md)
