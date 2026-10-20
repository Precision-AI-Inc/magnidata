# task-00 — Training contract

**Depends on:** —
**Commit message:** `Add training contract for MLflow runs and per-sample loss artifacts`

## Description

Every later task — trainer, recorder, worker, API, dashboard — exchanges data through
MLflow. Define, once and in one module, how a MagniData-compatible run looks in MLflow:
which tags link it to a dataset, which metric keys it uses, where per-sample artifacts
live, and the column format of those artifacts. The module must import with **neither
torch nor mlflow installed** so the API, the trainer, and external trainers (pai-seed) can
all share it.

## Proposed solution

Create the package `precisionai/magnidata/training/` (`__init__.py` re-exports only, and
never imports torch- or mlflow-dependent modules) with `contract.py`:

- **Tags** (constants, all prefixed `magnidata.`):
  | Tag | Meaning |
  |---|---|
  | `magnidata.dataset_source` | dataset CSV path relative to `DATA_ROOT` — the join key to a MagniData dataset |
  | `magnidata.dataset_name` | display name |
  | `magnidata.origin` | `ui` (launched from MagniData) or `external` |
  | `magnidata.preset` | preset key (task-08) |
  | `magnidata.loss_terms` | comma-separated term names |
  | `magnidata.per_sample_schema` | `SCHEMA_VERSION`; presence means per-sample artifacts exist |
  | `magnidata.scored_epochs` | comma-separated epochs whose per-sample file is logged (updated each epoch) |
  | `magnidata.cancel_requested` | `true` when the UI asks to cancel |
  | `magnidata.parent_run_id` | lineage for retraining (task-16) |
  | `magnidata.worker` | worker id that claimed the run |
- **Metric keys** (helpers, not free strings): `train_loss/loss`, `train_loss/<term>`,
  `val_loss/loss`, `val_loss/<term>`, `Learning_rate/lr`, `val_metric/miou`,
  `val_metric/iou_<class>`. Step = epoch for epoch metrics, global step for `Learning_rate/lr`.
  These match pai-seed's `Trainer` so its runs plot identically.
- **Artifact layout** (paths inside the run's artifact root):
  ```
  magnidata/split.csv                          image_name,split   (train|val|test)
  magnidata/palette.json                       class index → RGB + name
  magnidata/per_sample/epoch_0001.csv          one row per scored sample
  magnidata/predictions/epoch_0001/<stem>.png  optional, top-K hardest only
  checkpoints/last.pt                          optional, trainer-owned
  ```
  Helpers: `per_sample_path(epoch) -> str`, `parse_per_sample_path(path) -> int | None`,
  `prediction_path(epoch, image_name) -> str`.
- **Per-sample columns**: required `image_name`, `split`, `loss_total`; one `loss_<term>`
  per term; optional `confidence`, `iou`. Term names match `^[a-z][a-z0-9_]{0,31}$`.
  Helpers `loss_column(term)`, `parse_loss_columns(header) -> list[str]`,
  `validate_header(header, terms) -> None`.
- `SCHEMA_VERSION = "magnidata-training/1.0.0"`.
- `image_key(value) -> str` = `os.path.basename(value)` — the same rule
  `api/embeddings_io.image_path_column` joins on. Move the pure `image_path_column`
  helper here (keep a re-export in `embeddings_io` so nothing else changes) so the
  trainer can find the image column without importing API code.
- `RunStatus` enum mirroring MLflow statuses: `SCHEDULED, RUNNING, FINISHED, FAILED, KILLED`.

Write `precisionai/magnidata/docs/training-contract.md` documenting tags, metric keys,
artifact layout, and an example per-sample file; link it from `docs/data-contract.md`.

Add `precisionai/magnidata/training/` to `Dockerfile.api`'s COPY (the package stays
importable without torch).

## Testing

`precisionai/magnidata/training/test_contract.py`:
- `per_sample_path(3) == "magnidata/per_sample/epoch_0003.csv"` and
  `parse_per_sample_path` round-trips; non-matching paths → `None`.
- `parse_loss_columns` returns terms in header order, ignores other columns; invalid term
  names → `pytest.raises(ValueError, match="term")`.
- `validate_header` rejects a missing `image_name` / `loss_total` / declared term.
- metric key helpers produce pai-seed-compatible keys (`train_loss/dice`, `val_loss/loss`).
- `image_key("a/b/c.png") == "c.png"`; `image_path_column` behaves exactly as before
  (existing `api/test_embeddings_io.py` still passes untouched).
- subprocess import check: importing `precisionai.magnidata.training.contract` leaves
  `"torch"` and `"mlflow"` out of `sys.modules`.
