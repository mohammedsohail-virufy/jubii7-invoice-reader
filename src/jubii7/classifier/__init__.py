"""Track A, parts 2-6: line-item category classifier (data, models, tracking, evaluation, serving)."""

LABELS: tuple[str, ...] = ("cement", "steel", "aggregates", "blocks", "services")
UNMAPPED = "unmapped"
SEED = 42
