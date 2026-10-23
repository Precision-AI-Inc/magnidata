# task-38 — Run detail and compare plot every metric

**Depends on:** task-37
**Commit message:** `Load every metric history in run detail and run compare`

## Description

Run detail and run compare requested histories for the first 20 metric keys only (the API's per-request limit),
so an AgriTune run, which logs ~140 keys (per-class dice, IoU, precision and recall for train and val), showed
empty charts for most metrics. Both views now fetch all visible keys in batches of 20 and merge the results.

## Testing

- `batches` splits 142 keys into 8 requests of at most 20 and keeps order; dashboard tests, typecheck, build.
