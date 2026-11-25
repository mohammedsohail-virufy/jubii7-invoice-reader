# Architecture

## 1. Where this repo sits in a document-scanning product

```
 upload / email / mobile scan
        │
        ▼
 ┌─────────────┐   ┌──────────────────┐   ┌────────────────────── this repo ───────────────────────┐
 │ image clean │──►│ OCR + layout +   │──►│ profile ─► 12 business rules ─► line classifier cascade │──► post to ERP
 │ deskew, crop│   │ table extraction │   │ (Arabic digits,  (recompute     (rules → model →        │
 └─────────────┘   └──────────────────┘   │  gaps, verify)    every total)    human review)         │──► review queue
                                          └─────────────────────────────────────────────────────────┘        │
                                                         ▲                                                    │
                                                         └──────── reviewed lines become new training data ◄──┘
```

The checker and classifier are deliberately independent of how extraction is done: they
take JSON, so the OCR vendor can change without touching business logic.

## 2. Code design (SOLID)

| Principle | Where |
|---|---|
| **Single responsibility** | one rule per `Check` class; parsing in `loader.py`, maths in `totals.py`, profiling in `profiling.py`, plotting separate from evaluation |
| **Open/closed** | new rule = new `Check` subclass added to `ALL_CHECKS`; new model = new `ModelSpec`; new stage = new `LineClassifier`. No existing class changes |
| **Liskov substitution** | every `Check` returns `list[Issue]` and never mutates or raises; every `ModelSpec.build()` returns a pipeline with `predict_proba`; every `LineClassifier` returns a `Prediction`; `JsonTracker` and `MlflowTracker` are interchangeable. Contract tests enforce all four |
| **Interface segregation** | small interfaces: `Check.run`, `LineClassifier.predict`, `ExperimentTracker.log_run` |
| **Dependency inversion** | `InvoiceChecker` depends on `Check`; the API receives its classifier and checker through `create_app(...)`; training receives a tracker |

Other patterns: **composite** (a `Cascade` is itself a `LineClassifier`, so cascades nest),
**strategy** (model specs, trackers), **value objects** (frozen dataclasses for the invoice,
issues and predictions), **validation at the boundary** (loader and Pydantic models), and
**fail-safe** behaviour (a crashing rule becomes an issue; missing MLflow falls back to JSON).

## 3. ML pipeline stages

`jubii7 train` runs typed stage functions, each small enough to become its own pipeline step:

```
load + validate data ─► split ─► compare candidates (5×5 CV, log runs) ─► select by CV
   ─► hold-out report ─► cascade evaluation (CV + unseen challenge set) ─► diagnostic curves
   ─► fit on all rows ─► model.joblib + model_card.json (metrics, data summary, SHA-256)
```

## 4. Production on AWS

```
S3 (raw scans) ──event──► Step Functions workflow
                             ├─ Lambda: image clean-up
                             ├─ Amazon Textract (layout, tables, forms) and/or a multimodal LLM on
                             │  Amazon Bedrock with structured JSON output for Arabic text
                             ├─ Lambda / ECS: this package's checker + profiler
                             ├─ SageMaker endpoint (or Lambda for the small TF-IDF model): rules → model cascade
                             ├─ Choice: any ERROR or needs_review ─► Amazon A2I human review ─► labels to S3
                             └─ DynamoDB (results, audit trail) ─► ERP / API Gateway
Training:  SageMaker Pipelines (the stages above) ─► SageMaker Model Registry (approval gate)
Tracking:  SageMaker managed MLflow (same ExperimentTracker interface: pass the tracking server ARN)
Monitoring: CloudWatch metrics + SageMaker Model Monitor: review rate, override rate,
            low-confidence share, class-mix drift; alarm → retraining pipeline
Security:  KMS-encrypted S3, IAM least privilege per step, VPC endpoints, PII kept in-region
```

Why this shape: each step scales and fails independently, a person is in the loop exactly
where confidence is low, and every human correction feeds the next training run
(active learning). The same code runs locally (`jubii7 serve`), in Docker, or on AWS:
only the tracker and the deployment target change.
