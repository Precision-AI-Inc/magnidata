# task-08 — API: training presets, launch (SCHEDULED run) and cancel

**Depends on:** task-06
**Commit message:** `Add training presets and launch API backed by MLflow scheduled runs`

## Description

Let the dashboard start and cancel training without the API running torch, holding a
Docker socket, or executing user-provided commands. Launching creates an MLflow run with
status `SCHEDULED` carrying validated params and `magnidata.*` tags — MLflow *is* the
queue — and the worker (task-09) picks it up. Users choose from a fixed preset registry
and may only tune whitelisted parameters within bounds.

## Proposed solution

`api/training_presets.py` — static registry served as JSON:
- presets `tiny_unet` ("Tiny UNet — CPU-friendly, smoke tests") and `lraspp_mobilenet`
  ("LR-ASPP MobileNetV3 — GPU recommended"), each with `defaults` and `bounds`.
- tunable params with bounds: `epochs` (1–200), `score_every` (1–epochs), `batch_size`
  (1–64), `image_size` (128–1024, multiple of 32), `lr` (1e-6–1e-1), `val_fraction`
  (0.05–0.5), `seed`, `split_mode` (`random`|`column`), `loss_terms` ⊆ contract names
  (≥ 1), `loss_weights` (0–10 each), `top_k_predictions` (0–200), `pretrained` (bool).
- `validate_launch(body) -> LaunchSpec`; unknown keys rejected (not ignored); each
  violation → `ValueError` naming the field.

`api/trainer_backend.py`:
```python
class TrainerBackend(Protocol):
    def submit(self, spec: LaunchSpec) -> str: ...   # returns MLflow run id
    def cancel(self, run_id: str) -> str: ...        # returns resulting status
    def available(self) -> dict: ...                 # {"available", "reason", "worker_heartbeat"}
class MlflowQueueBackend: ...
```
- `submit`: experiment `magnidata/<dataset slug>` (create if missing; overridable via
  `MAGNIDATA_MLFLOW_EXPERIMENT_PREFIX`), `create_run` with run name, all params logged,
  tags `magnidata.origin=ui`, `dataset_source`, `dataset_name`, `preset`, `loss_terms`,
  then status `SCHEDULED`.
- `cancel`: `SCHEDULED` → `set_terminated(KILLED)` directly; `RUNNING` → tag
  `magnidata.cancel_requested=true`; terminal → error (409).
- `available`: worker heartbeat = mtime of `DATA_ROOT/data_user/.trainer_heartbeat`
  (written by task-09) younger than 30 s. Docstring note on where a `NomadBackend` would
  plug in — do not implement it.

Routes (`experiments` blueprint):

| Method | Route | Behaviour |
|---|---|---|
| GET | `/api/training/presets` | presets + bounds + `backend.available()` + MLflow status |
| POST | `/api/training/runs` | body `{name, dataset_source, preset, params, parent_run_id?}` → validate, dataset exists, `dataset_check.has_masks` (task-04) → submit → 201 `{run_id, mlflow_ui_url}` |
| POST | `/api/runs/<run_id>/cancel` | per the rules above |

Only one `SCHEDULED`/`RUNNING` UI-origin run at a time (409 otherwise), mirroring the
dataset builder's single build slot.

## Testing

`precisionai/magnidata/api/test_training_launch.py` (real sqlite MLflow fixture):
- presets endpoint: `available` false without heartbeat, true with a fresh one, false with
  a stale (mtime − 60 s) one.
- valid launch → 201; the MLflow run has status `SCHEDULED`, params logged, tags set,
  experiment `magnidata/<slug>` created.
- each bound violation → 400 naming the field (parametrised); unknown preset / unknown
  loss term / dataset without masks / extra key `"command"` → 400.
- second launch while one is scheduled → 409.
- cancel scheduled → `KILLED`; cancel running (status updated in test) → tag set; cancel
  finished → 409.
