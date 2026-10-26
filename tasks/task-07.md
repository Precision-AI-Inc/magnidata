# task-07 — API: derived per-sample axes from MLflow artifacts

**Depends on:** task-06
**Commit message:** `Derive per-sample loss axes from MLflow run artifacts`

## Description

Raw per-epoch losses answer "how hard is this sample now". The feature's real value is
"which samples **stay** hard and why", which needs per-sample statistics across epochs,
plus rank-normalised versions because terms live on different scales (CCE vs Dice).
Compute these in the API (numpy only) from the per-sample artifacts in MLflow, serving
only the new columns so the dashboard never re-downloads the dataset CSV.

## Proposed solution

`api/artifact_cache.py` — local cache of per-sample artifacts under
`DATA_ROOT/data_user/.mlflow_cache/<run_id>/` (data_user is the only rw volume):
- `sync_per_sample(store, run_id) -> list[tuple[int, Path]]` downloads only epochs listed
  in `magnidata.scored_epochs` that are not cached yet (artifacts are immutable once
  logged, so cached files never need re-validation); also `split.csv`.
- size bound: evict least-recently-used run folders above `MLFLOW_CACHE_MAX_MB` (default 2 048).

`api/run_axes.py` (numpy + stdlib):
- `load_epoch_matrix(files, term, up_to_epoch=None) -> (names, epochs, M[n_samples, n_epochs])`, NaN for missing.
- `derive_axes(files, epoch="latest" | int, terms=None) -> DerivedAxes`, per term `t`:
  | column | definition |
  |---|---|
  | `loss_<t>` | value at the selected epoch |
  | `loss_<t>_mean` | nanmean over scored epochs ≤ selected |
  | `loss_<t>_delta` | first scored value − selected value (positive = improved) |
  | `loss_<t>_slope` | least-squares slope over the last `K=5` scored epochs ≤ selected (negative = getting easier); NaN with < 2 points |
  | `loss_<t>_rank` | percentile rank (0..1) at the selected epoch, within the sample's split |
  plus `split`, `hardness` (fraction of scored epochs ≤ selected where the sample's
  `loss_total` was in the top `q=20%` of its split), `iou`, `confidence` when present.
- `RunAxesCache`: LRU (8 entries) keyed by `(run_id, epoch, latest scored epoch)` so live
  polling recomputes only when a new epoch lands. Do **not** use the unbounded
  `_COMPUTE_CACHE` in `app.py` (known memory issue).

Routes (in the `experiments` blueprint):

| Method | Route | Behaviour |
|---|---|---|
| GET | `/api/runs/<run_id>/columns?epoch=latest\|N` | `text/csv`, `Cache-Control: no-store`: `image_name` + derived columns. 409 if the run has no per-sample artifacts |
| GET | `/api/runs/<run_id>/match?dataset=<source>` | `{matched, run_only, dataset_only}` comparing run image names with the dataset's image basenames |

## Testing

`precisionai/magnidata/api/test_run_axes.py` (hand-written CSVs in `tmp_path`):
- 3 samples × 4 epochs with known values → exact `mean`, `delta`, `slope` (vs `np.polyfit`), `rank`.
- `hardness` = 0.75 for a sample in the top 20% in 3 of 4 epochs; ranks computed within split.
- `epoch=2` ignores later epochs; missing sample in one epoch handled; single epoch → slope NaN, delta 0.
- cache hit/miss and eviction at 8 entries.

`precisionai/magnidata/api/test_artifact_cache.py` + route tests (real sqlite MLflow fixture):
- sync downloads only new epochs on the second call (log a 3rd epoch in between).
- size eviction removes the least-recently-used run folder.
- `/columns` returns CSV with `image_name` first and `loss_total_rank`, `hardness`;
  run without per-sample artifacts → 409; `/match` counts correct on partial overlap.
