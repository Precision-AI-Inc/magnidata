# task-20 — Per-sample data from the start of a run: early scoring and split before scoring

**Depends on:** task-19
**Commit message:** `Make per-sample split and losses available early in a run`

## Description

Nothing per-sample is visible until the first scoring pass ends: the
`magnidata.per_sample_schema` tag (→ `has_per_sample`) is only set in
`recorder.end_epoch()` after a full train + val + scoring pass (`train.py:~453`). On CPU
that was 15–20 min; with a larger `score_every`, a run might only score at its last epoch.
Meanwhile `split.csv` is logged at run start (`train.py:354`) but `/api/runs/<id>/columns`
ignores it (`experiments.py:~365`, `with_split=False`) and returns 409 until a scored epoch exists.
Also `/api/runs` caches results for 5 s (`experiments.py:56,216`) and is not invalidated on
launch/cancel/link.

## Proposed solution

Trainer:
- Always score epoch 1 (and the last epoch), regardless of `score_every`:
  `epoch == 1 or epoch % score_every == 0 or epoch == epochs`.
- Set a new tag `magnidata.has_split=true` right after `split.csv` is logged
  (add the key to `contract.py`).

API (`api/experiments.py`, `api/run_axes.py`, `api/artifact_cache.py`):
- Run summaries expose `has_split` (tag present or `split.csv` artifact) alongside
  `has_per_sample`; `/api/runs?dataset=…&per_sample=any|split|scored` server-side filter
  (default `any` — return everything linked to the dataset).
- `GET /api/runs/<id>/columns`: if no epoch is scored yet but `split.csv` exists, return
  200 `image_name,split` (header `X-Magnidata-Scored-Epochs: 0`); 409 only if neither exists.
  When scored epochs exist, `split` must still come first after `image_name` (unchanged).
- Clear the `/api/runs` cache after launch, cancel, and link.

## Testing

- `training/test_train.py`: with `epochs=4, score_every=3` scored epochs are `[1, 3, 4]`;
  `magnidata.has_split` set before epoch 1 finishes (check tag after `prepare_data`).
- `api/test_experiments.py` (real sqlite MLflow fixture): run with only `split.csv` →
  `/columns` 200 with exactly `image_name,split`; run with neither → 409; run summary
  `has_split` true / `has_per_sample` false; `per_sample=scored` filter excludes it;
  a launch followed immediately by `/api/runs?dataset=…` includes the new run (cache cleared).
- `api/test_run_axes.py` unchanged behaviour for scored runs.
