# task-06 — API: browse experiments, runs, params, metric history, artifacts

**Depends on:** task-01
**Commit message:** `Add API endpoints to browse MLflow experiments and runs`

## Description

Let the dashboard show what is in MLflow — any experiment, any run, including pai-seed
runs that were never launched from MagniData — with the information people normally open
the MLflow UI for: run list with key metrics/params, status, metric curves, params, tags,
and artifacts. The API proxies MLflow through the adapter (task-01) so the browser never
talks to MLflow directly and credentials stay server-side.

## Proposed solution

New Flask blueprint `api/experiments.py` (registered in `app.py`; keeps `app.py` from
growing). All handlers go through `MlflowStore`; `MlflowUnavailable` → 503
`{"error": …, "mlflow_unavailable": true}`; `MlflowNotFound` → 404.

| Method | Route | Behaviour |
|---|---|---|
| GET | `/api/experiments?query=&max_results=&page_token=` | experiments (id, name, tags, last update, run count if cheap); hides deleted |
| GET | `/api/runs?experiment_ids=a,b&dataset=<source>&status=&filter=&order_by=&max_results=50&page_token=` | `search_runs`; `dataset` adds `tags."magnidata.dataset_source" = '…'`; `filter` passes a raw MLflow filter string **validated** against an allow-list grammar (metrics./params./tags./attributes., comparison ops, AND) to avoid surprises; returns run id, name, experiment, status, start/end, duration, user, latest metrics, params, `magnidata.*` tags, `has_per_sample` flag, `mlflow_ui_url` deep link |
| GET | `/api/runs/<run_id>` | full run: params, tags, latest metrics, metric keys, status, artifact root summary, `scored_epochs` |
| GET | `/api/runs/<run_id>/metrics?keys=a,b` | metric histories `{key: [{step, value, timestamp}]}`; all keys when omitted; capped at 20 keys per call; downsample to ≤ 2 000 points per key (keep min/max per bucket so spikes survive) |
| POST | `/api/runs/metrics` | body `{run_ids: [...], keys: [...]}` → histories for several runs (for compare, task-12); cap 10 runs |
| GET | `/api/runs/<run_id>/artifacts?path=` | one directory level of artifacts (path, is_dir, size) |
| GET | `/api/runs/<run_id>/artifact?path=` | stream a single artifact (size cap 50 MB; images get their proper mimetype, others `application/octet-stream` + `Content-Disposition`) |
| POST | `/api/runs/<run_id>/link` | body `{dataset_source}` → sets `magnidata.dataset_source` / `dataset_name` tags, so an external run (e.g. pai-seed) can be attached to a dataset |

- `run_id` validated as `^[0-9a-f]{32}$`; artifact `path` must be relative, no `..`.
- Short-TTL cache (5 s) for `search_runs` results keyed by query, bounded size — the UI
  polls; MLflow should not get one request per tab per second.

## Testing

`precisionai/magnidata/api/test_experiments.py` (real sqlite MLflow fixture, Flask test client):
- seed 2 experiments × 3 runs with params/metrics/tags; list experiments; search runs by
  experiment, by dataset tag, by status; order by a metric; pagination token round-trip.
- filter validation: a valid filter passes; `"1=1; DROP"` style input → 400 with `match`-able message.
- metric history: correct steps/values; downsampling keeps the max spike; >20 keys → 400.
- multi-run metrics: two runs returned; 11 runs → 400.
- artifacts: listing and download of a seeded file; traversal (`../x`) → 400; unknown run → 404.
- link: tags set and the run then appears in `/api/runs?dataset=…`.
- MLflow down (store pointed at `http://127.0.0.1:1`) → 503 with `mlflow_unavailable: true`.
