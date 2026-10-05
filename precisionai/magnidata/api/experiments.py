# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Browse MLflow experiments, runs, metric curves and artifacts through the API.

The browser never talks to MLflow directly; every handler goes through `mlflow_store`.
"""

import mimetypes
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import mlflow_store
from flask import Blueprint, Response, jsonify, request
from flask.typing import ResponseReturnValue
from mlflow_store import MlflowNotFoundError, MlflowUnavailableError
from werkzeug.exceptions import BadRequest

from precisionai.magnidata.training import contract

bp = Blueprint("experiments", __name__)

RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")
PAGE_TOKEN_RE = re.compile(r"^[A-Za-z0-9+/=_-]{1,2000}$")
MAX_FILTER_LENGTH = 500
MAX_RESULTS_CAP = 200
DEFAULT_RUN_RESULTS = 50
MAX_METRIC_KEYS = 20
MAX_COMPARE_RUNS = 10
MAX_POINTS = 2000
MAX_ARTIFACT_BYTES = 50 * 1024 * 1024
MAX_EXPERIMENTS = 20

_IDENT = r"(?:metrics?|params?|tags?|attributes?|run)\.(?:`[^`]{1,200}`|[A-Za-z0-9_.\-/]{1,200})"
_OP = r"(?:=|!=|<=|>=|<|>|LIKE|ILIKE)"
_VALUE = r"(?:'[^']{0,200}'|\"[^\"]{0,200}\"|-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
_CLAUSE = rf"{_IDENT}\s*{_OP}\s*{_VALUE}"
FILTER_RE = re.compile(rf"^{_CLAUSE}(?:\s+AND\s+{_CLAUSE})*$", re.IGNORECASE)
ORDER_BY_RE = re.compile(
    r"^(?:metrics?|params?|tags?|attributes?|run)\.(?:`[^`]{1,200}`|[A-Za-z0-9_.\-/]{1,200})(?:\s+(?:ASC|DESC))?$",
    re.IGNORECASE,
)

RUNS_CACHE_TTL_S = 5.0
RUNS_CACHE_MAX = 128
_cache: dict[tuple, tuple[float, Any]] = {}
_cache_lock = threading.Lock()


def clear_cache() -> None:
    """Drop every cached run search."""
    with _cache_lock:
        _cache.clear()


def _cached(key: tuple, compute: Any) -> Any:
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < RUNS_CACHE_TTL_S:
            return hit[1]
    value = compute()
    with _cache_lock:
        if len(_cache) >= RUNS_CACHE_MAX:
            _cache.pop(min(_cache, key=lambda k: _cache[k][0]))
        _cache[key] = (now, value)
    return value


@bp.errorhandler(MlflowUnavailableError)
def _unavailable(err: MlflowUnavailableError) -> ResponseReturnValue:
    return jsonify({"error": str(err), "mlflow_unavailable": True}), 503


@bp.errorhandler(MlflowNotFoundError)
def _not_found(err: MlflowNotFoundError) -> ResponseReturnValue:
    return jsonify({"error": str(err)}), 404


@bp.errorhandler(BadRequest)
def _bad_request(err: BadRequest) -> ResponseReturnValue:
    return jsonify({"error": err.description}), 400


def _bad(message: str) -> BadRequest:
    return BadRequest(message)


def _run_id(value: str) -> str:
    if not RUN_ID_RE.match(value):
        raise _bad("invalid run id")
    return value


def _int_arg(name: str, default: int, maximum: int) -> int:
    raw = request.args.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise _bad(f"{name} must be an integer") from None
    if not 1 <= value <= maximum:
        raise _bad(f"{name} must be between 1 and {maximum}")
    return value


def _page_token() -> str | None:
    token = request.args.get("page_token") or None
    if token is not None and not PAGE_TOKEN_RE.match(token):
        raise _bad("invalid page_token")
    return token


def _csv_arg(name: str) -> list[str]:
    return [part.strip() for part in request.args.get(name, "").split(",") if part.strip()]


def _artifact_path(raw: str | None) -> str | None:
    if not raw:
        return None
    parts = raw.split("/")
    if raw.startswith("/") or "\\" in raw or ".." in parts or "\x00" in raw:
        raise _bad("invalid artifact path")
    return raw.strip("/") or None


def _build_filter() -> str:
    clauses = []
    if dataset := request.args.get("dataset"):
        if "'" in dataset or "`" in dataset or len(dataset) > 500:
            raise _bad("invalid dataset")
        clauses.append(f"tags.`{contract.TAG_DATASET_SOURCE}` = '{dataset}'")
    if status := request.args.get("status"):
        if status not in {s.value for s in contract.RunStatus}:
            raise _bad("invalid status")
        clauses.append(f"attributes.status = '{status}'")
    if raw := request.args.get("filter", "").strip():
        if len(raw) > MAX_FILTER_LENGTH or not FILTER_RE.match(raw):
            raise _bad("invalid filter: expected clauses like metrics.x > 1 AND params.y = 'z'")
        clauses.append(raw)
    return " AND ".join(clauses)


def _build_order_by() -> list[str] | None:
    raw = request.args.get("order_by", "").strip()
    if not raw:
        return None
    entries = [e.strip() for e in raw.split(",") if e.strip()]
    if len(entries) > 3 or not all(ORDER_BY_RE.match(e) for e in entries):
        raise _bad("invalid order_by")
    return entries


def run_summary(run: dict[str, Any]) -> dict[str, Any]:
    """Return the compact run representation used in lists."""
    ui = mlflow_store.get_store().ui_url
    start, end = run["start_time"], run["end_time"]
    tags = run["tags"]
    return {
        "run_id": run["run_id"],
        "name": run["name"],
        "experiment_id": run["experiment_id"],
        "status": run["status"],
        "start_time": start,
        "end_time": end,
        "duration_ms": end - start if start and end else None,
        "user": run["user"],
        "metrics": run["metrics"],
        "params": run["params"],
        "magnidata": {k: v for k, v in tags.items() if k.startswith("magnidata.")},
        "has_per_sample": contract.TAG_PER_SAMPLE_SCHEMA in tags,
        "mlflow_ui_url": f"{ui.rstrip('/')}/#/experiments/{run['experiment_id']}/runs/{run['run_id']}" if ui else None,
    }


def parse_scored_epochs(tags: dict[str, str]) -> list[int]:
    """Return the epochs listed in a run's ``magnidata.scored_epochs`` tag."""
    raw = tags.get(contract.TAG_SCORED_EPOCHS, "")
    return sorted(int(e) for e in raw.split(",") if e.strip().isdigit())


@bp.get("/api/experiments")
def list_experiments() -> ResponseReturnValue:
    """List active experiments, optionally filtered by name."""
    query = request.args.get("query", "").strip()
    if "'" in query or "%" in query or len(query) > 100:
        raise _bad("invalid query")
    page = mlflow_store.get_store().search_experiments_page(query, _int_arg("max_results", 100, 1000), _page_token())
    return jsonify({"experiments": page["items"], "next_page_token": page["next_page_token"]})


@bp.get("/api/runs")
def list_runs() -> ResponseReturnValue:
    """Search runs across experiments with allow-listed filters."""
    store = mlflow_store.get_store()
    experiment_ids = _csv_arg("experiment_ids")
    if len(experiment_ids) > MAX_EXPERIMENTS:
        raise _bad(f"at most {MAX_EXPERIMENTS} experiment_ids")
    if not experiment_ids:
        experiment_ids = [e["experiment_id"] for e in store.search_experiments(1000)]
    filter_string, order_by = _build_filter(), _build_order_by()
    max_results, token = _int_arg("max_results", DEFAULT_RUN_RESULTS, MAX_RESULTS_CAP), _page_token()
    key = (id(store), tuple(experiment_ids), filter_string, tuple(order_by or ()), max_results, token)
    page = _cached(key, lambda: store.search_runs_page(experiment_ids, filter_string, max_results, order_by, token))
    return jsonify({"runs": [run_summary(r) for r in page["items"]], "next_page_token": page["next_page_token"]})


@bp.get("/api/runs/<run_id>")
def get_run(run_id: str) -> ResponseReturnValue:
    """Return one run in full: params, tags, latest metrics, scored epochs and top-level artifacts."""
    store = mlflow_store.get_store()
    run = store.get_run(_run_id(run_id))
    return jsonify(
        {
            **run_summary(run),
            "tags": run["tags"],
            "metric_keys": sorted(run["metrics"]),
            "scored_epochs": parse_scored_epochs(run["tags"]),
            "artifact_uri": run["artifact_uri"],
            "artifacts": store.list_artifacts(run_id),
        }
    )


def downsample(points: list[dict[str, Any]], limit: int = MAX_POINTS) -> list[dict[str, Any]]:
    """Reduce a history to at most ``limit`` points, keeping each bucket's min and max (and the ends)."""
    if len(points) <= limit:
        return points
    buckets = (limit - 2) // 2
    size = len(points) / buckets
    keep = {0, len(points) - 1}
    for b in range(buckets):
        chunk = range(int(b * size), min(int((b + 1) * size), len(points)))
        if not chunk:
            continue
        keep.add(min(chunk, key=lambda i: points[i]["value"]))
        keep.add(max(chunk, key=lambda i: points[i]["value"]))
    return [points[i] for i in sorted(keep)]


def _histories(run_id: str, keys: list[str]) -> dict[str, list[dict[str, Any]]]:
    store = mlflow_store.get_store()
    out = {}
    for key in keys:
        history = store.get_metric_history(run_id, key)
        out[key] = downsample([{"step": m["step"], "value": m["value"], "timestamp": m["timestamp"]} for m in history])
    return out


@bp.get("/api/runs/<run_id>/metrics")
def run_metrics(run_id: str) -> ResponseReturnValue:
    """Return metric histories for the requested keys (all keys, capped, when omitted)."""
    run = mlflow_store.get_store().get_run(_run_id(run_id))
    keys = _csv_arg("keys")
    truncated = False
    if len(keys) > MAX_METRIC_KEYS:
        raise _bad(f"at most {MAX_METRIC_KEYS} keys per call")
    if not keys:
        all_keys = sorted(run["metrics"])
        keys, truncated = all_keys[:MAX_METRIC_KEYS], len(all_keys) > MAX_METRIC_KEYS
    return jsonify({"run_id": run_id, "metrics": _histories(run_id, keys), "truncated": truncated})


@bp.post("/api/runs/metrics")
def runs_metrics() -> ResponseReturnValue:
    """Return metric histories for several runs at once (for comparison)."""
    body = request.get_json(silent=True) or {}
    run_ids, keys = body.get("run_ids"), body.get("keys")
    if not isinstance(run_ids, list) or not run_ids or not all(isinstance(r, str) for r in run_ids):
        raise _bad("run_ids must be a non-empty list")
    if not isinstance(keys, list) or not keys or not all(isinstance(k, str) for k in keys):
        raise _bad("keys must be a non-empty list")
    if len(run_ids) > MAX_COMPARE_RUNS:
        raise _bad(f"at most {MAX_COMPARE_RUNS} runs")
    if len(keys) > MAX_METRIC_KEYS:
        raise _bad(f"at most {MAX_METRIC_KEYS} keys per call")
    return jsonify({"runs": {_run_id(r): _histories(r, keys) for r in run_ids}})


@bp.get("/api/runs/<run_id>/artifacts")
def run_artifacts(run_id: str) -> ResponseReturnValue:
    """List one directory level of a run's artifacts."""
    path = _artifact_path(request.args.get("path"))
    store = mlflow_store.get_store()
    store.get_run(_run_id(run_id))
    return jsonify({"path": path or "", "artifacts": store.list_artifacts(run_id, path)})


@bp.get("/api/runs/<run_id>/artifact")
def run_artifact(run_id: str) -> ResponseReturnValue:
    """Return one artifact file (up to 50 MB); images keep their mimetype, others download."""
    path = _artifact_path(request.args.get("path"))
    if path is None:
        raise _bad("path is required")
    store = mlflow_store.get_store()
    store.get_run(_run_id(run_id))
    parent = os.path.dirname(path) or None
    entry = next((f for f in store.list_artifacts(run_id, parent) if f["path"] == path), None)
    if entry is None or entry["is_dir"]:
        raise MlflowNotFoundError(f"artifact not found: {path}")
    if (entry["file_size"] or 0) > MAX_ARTIFACT_BYTES:
        return jsonify({"error": "artifact larger than 50 MB"}), 413
    with tempfile.TemporaryDirectory() as tmp:
        content = Path(store.download_artifact(run_id, path, tmp)).read_bytes()
    mimetype = mimetypes.guess_type(path)[0] or "application/octet-stream"
    headers = {"Cache-Control": "no-store"}
    if not mimetype.startswith("image/"):
        mimetype = "application/octet-stream"
        headers["Content-Disposition"] = f'attachment; filename="{os.path.basename(path)}"'
    return Response(content, mimetype=mimetype, headers=headers)


@bp.post("/api/runs/<run_id>/link")
def link_run(run_id: str) -> ResponseReturnValue:
    """Attach a run (e.g. an external pai-seed run) to a dataset by setting its tags."""
    source = (request.get_json(silent=True) or {}).get("dataset_source")
    if not isinstance(source, str) or not source or len(source) > 500:
        raise _bad("dataset_source is required")
    if source.startswith("/") or "\\" in source or ".." in source.split("/") or "://" in source:
        raise _bad("dataset_source must be a relative path inside the data root")
    store = mlflow_store.get_store()
    store.get_run(_run_id(run_id))
    store.set_tag(run_id, contract.TAG_DATASET_SOURCE, source)
    store.set_tag(run_id, contract.TAG_DATASET_NAME, Path(source).stem)
    clear_cache()
    return jsonify(run_summary(store.get_run(run_id)))
