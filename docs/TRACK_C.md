# Track C answers (pen and paper, also as tested code)

## 1. Intersection over union
Code: `src/jubii7/track_c/geometry.py` (tests include symmetry and 0 ≤ IoU ≤ 1 for any boxes).

a = (10, 10, 50, 50), b = (30, 30, 70, 70)
- intersection (30,30)–(50,50) = 20 × 20 = **400**
- each area = 1,600; union = 3,200 − 400 = **2,800**
- **IoU = 400 / 2,800 = 1/7 ≈ 0.143**

## 2. Arabic digits to Western
`src/jubii7/text.py::to_western`: Arabic-Indic ٠–٩ and Persian ۰–۹, plus the separators ٫ and ٬.
Property tests check it is idempotent and that Arabic and Western numbers parse the same.

## 3. Metrics
TP = 90, FP = 30, FN = 10 (`src/jubii7/track_c/metrics.py`)
- precision = 90 / 120 = **0.75**
- recall = 90 / 100 = **0.90**
- F1 = 2 · 0.75 · 0.9 / 1.65 ≈ **0.818**

For a finance team **recall matters more**: a missed wrong invoice means money paid out or VAT
filed incorrectly; a false flag costs a reviewer a minute. Keep precision high enough that
reviewers don't start ignoring the flags.

## 4. Pipeline: page in → checked JSON out
1. Image clean-up: deskew, denoise; find stamp and handwriting regions.
2. Layout detection: header form, line table, totals box, terms, stamp, signatures.
3. OCR per region with an Arabic-capable model; keep logical order (Arabic text is right to left, digits left to right).
4. Table structure: merged headers (Quantity → Ordered / Delivered), group rows (Site A / B) as a parent key, subtotals kept apart from lines.
5. Normalise: digits, `(x)` → negative, dates → ISO.
6. Field mapping: pair each English label with its Arabic twin and check they agree.
7. Business rules: this repo's 12 checks.
8. Route: confidence = min(OCR, field match, rule pass). A person steps in on any ERROR, low confidence, handwriting, or an amount above a limit.

**Labelling 1,000 invoices:** 2 finance clerks + 1 reviewer in Label Studio; model pre-labels;
10% double-labelled to measure agreement (Cohen's κ); disagreements resolved against the guide.
**Per run, track:** data version, params, features, macro-F1 and per-class F1 (CV mean ± std),
confusion matrix, model artifact, git commit.
**Deploy and monitor:** container + API behind auth; log every prediction; watch review rate,
low-confidence share, class-mix drift and human-override rate; retrain monthly from reviewed lines.
