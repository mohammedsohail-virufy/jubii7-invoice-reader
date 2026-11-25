# Model report: what the curves say and how the model improves

Task: map an invoice line (English or Arabic) to one of 5 categories: cement, steel,
aggregates, blocks, services. Data: 57 labelled lines (27 Arabic), 10 from the invoice and
47 AI-generated then checked by hand. All figures are in `reports/figures/` and on the dashboard.

## Candidates (5 × 5 repeated stratified k-fold, 25 fits each)

| Model | CV macro-F1 | Hold-out F1 (18 rows) | Hold-out log loss |
|---|---|---|---|
| **word TF-IDF (1–2 grams) + logistic regression** ✓ | **0.846 ± 0.109** | 0.769 | 0.811 |
| char TF-IDF (2–5 grams) + logistic regression | 0.833 ± 0.110 | 0.826 | 0.813 |
| char TF-IDF + logistic loss trained by SGD | 0.821 ± 0.112 | 0.767 | 0.936 |

Chosen by CV mean. The hold-out ranks the char model higher, but on 18 rows one line
changes F1 by about 0.06, and the ± 0.11 CV spread shows the three are statistically close.

## What each curve shows

| Figure | Reading | Decision |
|---|---|---|
| **Learning curve** (05) | validation F1 rises 0.53 → 0.64 → 0.76 → 0.79 → 0.84 → 0.89 as rows go 13 → 45; training F1 stays 1.0 | high variance, data-limited: **more labelled lines is the #1 lever**, not a bigger model |
| **Regularisation C** (06) | validation flat at 0.845–0.865 across C = 0.01 … 1000; best from C ≥ 1 | hyperparameters are not the bottleneck; keep C = 10 |
| **Learning-rate sweep** (07) | validation 0.78–0.80 across eta0 = 0.001 … 3 | the loss is convex and TF-IDF features are well scaled, so SGD is robust to the learning rate |
| **Per-epoch curves** (08) | eta0 = 1.0 reaches training loss 0.07 in 40 epochs; 0.1 reaches 0.50; 0.01 barely moves (1.39) | the learning rate sets *speed*, not final quality here; a small rate needs far more epochs |
| **Per-class F1** (03) | blocks 1.00, cement 0.86, services 0.75, aggregates 0.67, **steel 0.57** | steel and aggregates are confused through shared units ("20 mm") |
| **Cascade vs model, unseen set** (09) | macro-F1 0.655 → 0.754; automation 23% → 63%; rules 94% precise on 53% of lines | rules earn their place; unseen synonyms are the gap |
| **Cascade vs model, CV** (10) | 1.00 everywhere | **not trustworthy**: the rules were written after reading these rows (see FINDINGS #1) |

## Error analysis (CV mistakes)

| Line | True → predicted | Why |
|---|---|---|
| Gravel 20 mm | aggregates → steel | "mm" appears mostly on rebar, so the unit outweighs the material |
| Concrete pump hire per day | services → aggregates | "concrete" pulls toward materials; "hire" is rare in training |
| Crane hire per hour | services → aggregates | same: too few hire examples |
| توصيل أكياس أسمنت للموقع | services → cement | the material word dominates the service word |
| OPC cement bulk per tonne | cement → aggregates | "per tonne" is mostly seen on sand and road base |
| Anchor bolts M16 | steel → services | unseen words; low confidence (0.27), so it would go to review |

On the unseen set the model fails on vocabulary it has never seen (TMT bars, crusher run,
بحص, طوب): confidence sits near 0.27, which is the expected behaviour of a model saying
"I don't know", and those lines are routed to a person.

## How the model improves, in order of value

1. **More labelled data from real invoices**, using active learning: the review queue
   (`needs_review`) is the labelling queue. The learning curve says this pays most.
2. **Multilingual sentence embeddings** (e.g. multilingual-E5 or LaBSE) + logistic regression:
   synonyms and Arabic/English pairs land close together, which fixes the unseen-vocabulary
   failures. Drops in as one new `ModelSpec`.
3. **A synonym-aware rule layer** grown from reviewed lines (شحن = shipping, بحص = gravel,
   طوب = brick), versioned and tested like code.
4. **LLM stage for the long tail**: a `LineClassifier` that asks an LLM with the labelling
   guide in the prompt, only for lines the rules and model are unsure about, with structured output.
5. **Calibrated confidence** (`CalibratedClassifierCV`) so the 0.5 review threshold means
   what it says; then tune the threshold on cost (missed error vs reviewer minute).
6. **Taxonomy fix**: add `finishes / waterproofing` for the membrane.
7. **Monitoring**: review rate, override rate, class-mix drift and low-confidence share,
   with monthly retraining from reviewed lines.
