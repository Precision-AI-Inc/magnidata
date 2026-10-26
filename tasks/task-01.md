# task-01 — MLflow service, configuration, and API adapter

**Depends on:** task-00
**Commit message:** `Add MLflow tracking service and API adapter`

## Description

MLflow becomes the system of record for experiments. Developers need a local tracking
server that works out of the box with `docker compose up`, and the API needs one narrow,
testable adapter to talk to the local MLflow server
— instead of `mlflow` calls scattered across route handlers.

## Proposed solution

**docker-compose.yml** — new `mlflow` service:
- image `ghcr.io/mlflow/mlflow:v3.11.1` (pin; same version pai-seed uses),
- command `mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:////mlflow/mlflow.db --artifacts-destination /mlflow/artifacts --serve-artifacts`,
- named volume `magnidata-mlflow:/mlflow`, published port `5052:5000` (MLflow UI on the host),
- `api` gets env `MLFLOW_TRACKING_URI: http://mlflow:5000`,
  `MAGNIDATA_MLFLOW_UI_URL: http://localhost:5052` (for deep links from the dashboard).
- MLflow is **local only**: no auth, no remote-server or token configuration. Comment in
  the compose file that host-side tools (e.g. a pai-seed run on the host) log to it with
  `MLFLOW_TRACKING_URI=http://localhost:5052`.
- `restart: unless-stopped` and a `healthcheck` (`curl -f http://localhost:5000/health`, or
  a python one-liner if curl is absent in the image); `api` uses
  `depends_on: mlflow: condition: service_healthy`.
- Update the header comment listing services/ports, and `api/.env.example`.

**Dependencies** — `mlflow-skinny==3.11.1` in `api/requirements.txt` (client only; no
server, no torch). `mlflow==3.11.1` in `requirements-dev.txt` (tests need the sqlite store).

**Adapter** — `precisionai/magnidata/api/mlflow_store.py` (copy into the API image via
`Dockerfile.api`):
- module-level optional import with `_MLFLOW_AVAILABLE` flag (CLAUDE.md pattern).
- `class MlflowStore` wrapping `mlflow.MlflowClient(tracking_uri)`; constructed once from
  env (`get_store()`), injectable for tests (`set_store()` / app config).
- methods used by later tasks (implement the ones needed now, signatures for the rest):
  `status() -> dict` (reachable, tracking_uri, server version if available, ui_url), `search_experiments`, `search_runs`, `get_run`, `get_metric_history`,
  `list_artifacts`, `download_artifact(run_id, path, dst_dir) -> Path`, `create_run`,
  `set_tag`, `set_terminated`.
- every method converts MLflow entities to plain dicts (no MLflow objects leak into
  routes) and maps errors to `MlflowUnavailable` / `MlflowNotFound` exceptions.
- timeouts: set `MLFLOW_HTTP_REQUEST_TIMEOUT` (default 10 s) and
  `MLFLOW_HTTP_REQUEST_MAX_RETRIES` (default 2) so a down server never hangs the API.

**Route** — `GET /api/mlflow/status` → `status()`; 200 with `reachable: false` (not 500)
when the server is down, so the UI can show a banner.

**Shared test fixture** — `precisionai/magnidata/conftest.py` (or extend the existing
one): `mlflow_store` fixture creating a real `MlflowStore` on
`sqlite:///{tmp_path}/mlflow.db` with artifact root `{tmp_path}/artifacts`, plus a
`seed_run(...)` helper that logs params/metrics/tags/artifacts for tests.

## Testing

`precisionai/magnidata/api/test_mlflow_store.py` (real sqlite-backed MLflow, no mocks):
- `status()` reachable on the fixture store; unreachable for `http://127.0.0.1:1`
  (returns quickly, `reachable: False`, no exception).
- entity-to-dict conversion: a seeded run returns params/metrics/tags as plain dicts.
- `MlflowNotFound` for an unknown run id (`pytest.raises(..., match="run")`).
- `GET /api/mlflow/status` via the Flask test client: 200 both when reachable and unreachable.
- Manual (not pytest): `docker compose up --build` brings up portal, API, and MLflow;
  `http://localhost:5052` serves the MLflow UI; `curl http://localhost:5051/api/mlflow/status`
  reports `reachable: true`; data survives `docker compose down && docker compose up`
  (named volume).
