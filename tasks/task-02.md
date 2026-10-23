# task-02 — Torch-free per-sample loss recorder with MLflow sink

**Depends on:** task-00, task-01
**Commit message:** `Add per-sample loss recorder that logs to MLflow`

## Description

Trainers need one small object that accumulates per-sample loss values during a scoring
pass, validates them against the contract, and logs them to the MLflow run as artifacts.
It must be torch-free (numpy + stdlib, mlflow optional) so the reference trainer and
external trainers like pai-seed use exactly the same writer. Runs that already exist in
MLflow (e.g. past pai-seed runs) also need a way to attach per-sample files after the fact.

## Proposed solution

`precisionai/magnidata/training/recorder.py`:

```python
class LossRecorder:
    def __init__(self, loss_terms: Sequence[str], sink: RecorderSink, scratch_dir: Path) -> None: ...
    def write_split(self, assignments: Mapping[str, str]) -> None: ...
    def write_palette(self, palette: Mapping[int, dict]) -> None: ...
    def begin_epoch(self, epoch: int) -> None: ...
    def add_batch(self, image_names: Sequence[str], split: str | Sequence[str],
                  losses: Mapping[str, ArrayLike], extras: Mapping[str, ArrayLike] | None = None) -> None: ...
    def end_epoch(self) -> str: ...          # returns the artifact path logged
    def cancel_requested(self) -> bool: ...  # delegated to the sink

class RecorderSink(Protocol):
    def log_file(self, local_path: Path, artifact_dir: str) -> None: ...
    def set_tag(self, key: str, value: str) -> None: ...
    def cancel_requested(self) -> bool: ...

class MlflowSink:      # uses mlflow.MlflowClient + run_id; cancel = tag magnidata.cancel_requested == "true",
                       # checked at most every `cancel_poll_s` (default 15 s) to avoid hammering the server
class LocalDirSink:    # writes into a folder with the same layout — used by tests and offline runs
```

Rules:
- `losses` values are 1-D array-likes of length `len(image_names)`; accept numpy arrays or
  anything with `.tolist()` (torch tensors after `.detach().cpu()`) **without** importing
  torch. Must contain `total` and every declared term. Raise `ValueError` on length
  mismatch, missing/unknown term, non-finite values (message lists offending image names),
  duplicate `image_name` within an epoch, invalid split label, `add_batch` before `begin_epoch`.
- `end_epoch` writes the CSV (contract column order, `%.6g`) to `scratch_dir`, logs it to
  `magnidata/per_sample/`, then updates tag `magnidata.scored_epochs` and sets
  `magnidata.per_sample_schema` on first use.
- CLI for runs that already exist:
  `python -m precisionai.magnidata.training.attach --run-id <id> --epoch 12 --file losses.csv [--split split.csv]`
  validates the file against the contract and logs it via `MlflowSink`. A long-format file
  with an `epoch` column is split into one artifact per epoch.
- Export `LossRecorder`, `LocalDirSink` from `training/__init__.py`; `MlflowSink` lives in
  the same module behind the optional-import flag (raises `ImportError` with an install
  hint if mlflow is missing).

## Testing

`precisionai/magnidata/training/test_recorder.py`:
- happy path with `LocalDirSink`: two batches → `end_epoch` → CSV header order, row
  count, values (parsed back, compared with tolerance); `scored_epochs` tag updated.
- every error case above, each with `pytest.raises(ValueError, match=…)`.
- numpy array and plain list accepted for the same term.
- `MlflowSink` against the real sqlite MLflow fixture (task-01): artifact appears at
  `magnidata/per_sample/epoch_0001.csv`; tags set; `cancel_requested()` becomes true after
  `set_tag("magnidata.cancel_requested","true")` (with `cancel_poll_s=0`).
- `attach` CLI: long-format file → one artifact per epoch; invalid file → non-zero exit,
  nothing logged.
- subprocess import check: importing `recorder` does not import torch.
