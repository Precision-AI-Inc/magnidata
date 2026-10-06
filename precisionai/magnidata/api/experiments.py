# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Browse MLflow experiments, runs, metric curves and artifacts through the API.

The browser never talks to MLflow directly; every handler goes through `mlflow_store`.
"""

import csv
import mimetypes
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import artifact_cache
import image_overlay
import mlflow_store
import run_axes
import trainer_backend
import training_presets
from flask import Blueprint, Response, jsonify, request
from flask.typing import ResponseReturnValue
from mlflow_store import MlflowNotFoundError, MlflowUnavailableError
from werkzeug.exceptions import BadRequest, Conflict

from precisionai.magnidata.training import contract
from precisionai.magnidata.training.dataset_check import has_masks, resolve_under

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

PER_SAMPLE_MODES = ("any", "split", "scored")
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


@bp.errorhandler(Conflict)
def _conflict(err: Conflict) -> ResponseReturnValue:
    return jsonify({"error": err.description}), 409


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
    mode = request.args.get("per_sample", "any")
    if mode not in PER_SAMPLE_MODES:
        raise _bad(f"per_sample must be one of {', '.join(PER_SAMPLE_MODES)}")
    if mode == "scored":
        clauses.append(f"tags.`{contract.TAG_PER_SAMPLE_SCHEMA}` != ''")
    elif mode == "split":
        clauses.append(f"tags.`{contract.TAG_HAS_SPLIT}` = 'true'")
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
        "has_split": tags.get(contract.TAG_HAS_SPLIT) == "true" or contract.TAG_PER_SAMPLE_SCHEMA in tags,
        "mlflow_ui_url": f"{ui.rstrip('/')}/#/experiments/{run['experiment_id']}/runs/{run['run_id']}" if ui else None,
    }


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
            "scored_epochs": contract.parse_scored_epochs(run["tags"]),
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


AXES_CACHE = run_axes.RunAxesCache()


def _per_sample_run(run_id: str) -> dict[str, Any]:
    """Return the run, or raise 409 if it has no per-sample artifacts."""
    run = mlflow_store.get_store().get_run(_run_id(run_id))
    if contract.TAG_PER_SAMPLE_SCHEMA not in run["tags"] or not contract.parse_scored_epochs(run["tags"]):
        raise Conflict("run has no per-sample artifacts")
    return run


def _split_only_csv(run_id: str) -> Response | None:
    """Serve ``image_name,split`` for a run that has logged its split but scored no epoch yet."""
    store = mlflow_store.get_store()
    run = store.get_run(run_id)
    if contract.TAG_HAS_SPLIT not in run["tags"] and contract.TAG_PER_SAMPLE_SCHEMA not in run["tags"]:
        return None
    path = artifact_cache.fetch_artifact(store, run_id, contract.SPLIT_ARTIFACT)
    if path is None:
        return None
    return Response(
        path.read_text(),
        mimetype="text/csv",
        headers={"Cache-Control": "no-store", "X-Magnidata-Epoch": "0", "X-Magnidata-Scored-Epochs": "0"},
    )


@bp.get("/api/runs/<run_id>/columns")
def run_columns(run_id: str) -> ResponseReturnValue:
    """Return the derived per-sample axes at an epoch as CSV (``split`` right after ``image_name``).

    Before the first scored epoch the CSV is just ``image_name,split`` (when the run has logged its split).
    """
    run = mlflow_store.get_store().get_run(_run_id(run_id))
    if not contract.parse_scored_epochs(run["tags"]):
        early = _split_only_csv(run_id)
        if early is None:
            raise Conflict("run has no per-sample artifacts")
        return early
    run = _per_sample_run(run_id)
    scored = contract.parse_scored_epochs(run["tags"])
    raw = request.args.get("epoch", "latest")
    if raw != "latest" and not raw.isdigit():
        raise _bad("epoch must be 'latest' or a number")
    selected = scored[-1] if raw == "latest" else int(raw)
    if selected not in scored:
        raise _bad(f"epoch {selected} has no per-sample file (scored: {', '.join(map(str, scored))})")

    def compute() -> run_axes.DerivedAxes:
        files = artifact_cache.sync_per_sample(mlflow_store.get_store(), run_id, with_split=False)
        return run_axes.derive_axes(files, selected)

    axes = AXES_CACHE.get_or_compute((run_id, selected, scored[-1]), compute)
    return Response(
        axes.to_csv(),
        mimetype="text/csv",
        headers={"Cache-Control": "no-store", "X-Magnidata-Epoch": str(axes.epoch)},
    )


def _dataset_names(source: str) -> set[str]:
    path = resolve_under(os.environ.get("DATA_ROOT", "."), source)
    if path is None or not path.is_file():
        raise MlflowNotFoundError(f"dataset not found: {source}")
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        column = contract.image_path_column(reader.fieldnames or [])
        return {contract.image_key(row[column]) for row in reader} if column else set()


def _run_names(run_id: str, scored: list[int]) -> set[str]:
    files = artifact_cache.sync_per_sample(mlflow_store.get_store(), run_id, epochs=[scored[-1]])
    split = artifact_cache.cached_split(run_id)
    if split is None and not files:
        raise Conflict("run has no per-sample artifacts")
    path = split or files[0][1]
    with open(path, newline="") as fh:
        return {row["image_name"] for row in csv.DictReader(fh)}


@bp.get("/api/runs/<run_id>/match")
def run_match(run_id: str) -> ResponseReturnValue:
    """Compare the run's image names with a dataset's, so linking can warn about mismatches."""
    source = request.args.get("dataset", "")
    if not source:
        raise _bad("dataset is required")
    run = _per_sample_run(run_id)
    run_names = _run_names(run_id, contract.parse_scored_epochs(run["tags"]))
    dataset_names = _dataset_names(source)
    return jsonify(
        {
            "matched": len(run_names & dataset_names),
            "run_only": len(run_names - dataset_names),
            "dataset_only": len(dataset_names - run_names),
            "run_only_examples": sorted(run_names - dataset_names)[:5],
            "dataset_only_examples": sorted(dataset_names - run_names)[:5],
        }
    )


@bp.get("/api/training/presets")
def training_presets_view() -> ResponseReturnValue:
    """Return the presets with their bounds, plus trainer-worker and MLflow availability."""
    store = mlflow_store.get_store()
    return jsonify(
        {
            **training_presets.describe(),
            "backend": trainer_backend.get_backend(store).available(),
            "mlflow": store.status(),
        }
    )


def _check_dataset(source: str) -> None:
    root = os.environ.get("DATA_ROOT", ".")
    path = resolve_under(root, source)
    if path is None or not path.is_file():
        raise _bad("dataset_source: dataset not found")
    if not has_masks(path, root):
        raise _bad("dataset_source: dataset has no segmentation masks to train on")


@bp.post("/api/training/runs")
def launch_training() -> ResponseReturnValue:
    """Validate a launch request and queue it as a ``SCHEDULED`` MLflow run."""
    try:
        spec = training_presets.validate_launch(request.get_json(silent=True))
    except ValueError as err:
        raise _bad(str(err)) from None
    _check_dataset(spec.dataset_source)
    store = mlflow_store.get_store()
    if spec.parent_run_id:
        try:
            store.get_run(spec.parent_run_id)
        except MlflowNotFoundError:
            raise _bad("parent_run_id: run not found") from None
    try:
        run_id = trainer_backend.get_backend(store).submit(spec)
    except trainer_backend.TrainerBusyError as err:
        raise Conflict(str(err)) from None
    clear_cache()
    return jsonify(run_summary(store.get_run(run_id)) | {"run_id": run_id}), 201


@bp.post("/api/runs/<run_id>/cancel")
def cancel_run(run_id: str) -> ResponseReturnValue:
    """Cancel a run: a scheduled one is killed at once, a running one is asked to stop."""
    try:
        status = trainer_backend.get_backend(mlflow_store.get_store()).cancel(_run_id(run_id))
    except trainer_backend.RunNotCancellableError as err:
        raise Conflict(str(err)) from None
    clear_cache()
    return jsonify({"run_id": run_id, "status": status, "cancel_requested": status != "KILLED"})


def _image_name(value: str) -> str:
    if not value or "/" in value or "\\" in value or ".." in value or "\x00" in value:
        raise _bad("invalid image name")
    return value


_PREDICTION_LISTINGS: dict[tuple[str, int], set[str]] = {}


def prediction_epochs(run_id: str, scored: list[int], image_name: str) -> list[int]:
    """Epochs that have a logged prediction mask for an image.

    Listings of finished epochs never change and are cached; the newest epoch is always
    re-listed because its predictions are logged just after it is marked scored.
    """
    store = mlflow_store.get_store()
    stem = os.path.splitext(image_name)[0]
    found = []
    for epoch in scored:
        key = (run_id, epoch)
        names = _PREDICTION_LISTINGS.get(key)
        if names is None:
            folder = os.path.dirname(contract.prediction_path(epoch, "x.png"))
            names = {os.path.splitext(os.path.basename(f["path"]))[0] for f in store.list_artifacts(run_id, folder)}
            if epoch != scored[-1]:
                _PREDICTION_LISTINGS[key] = names
        if stem in names:
            found.append(epoch)
    return found


@bp.get("/api/runs/<run_id>/samples/<image_name>")
def run_sample(run_id: str, image_name: str) -> ResponseReturnValue:
    """Return one image's per-term loss trajectory across scored epochs, and where predictions exist."""
    run = _per_sample_run(run_id)
    name = _image_name(image_name)
    scored = contract.parse_scored_epochs(run["tags"])
    files = artifact_cache.sync_per_sample(mlflow_store.get_store(), run_id, with_split=False)
    history = run_axes.sample_history(files, name)
    if history is None:
        raise MlflowNotFoundError(f"image {name} is not in this run")
    return jsonify({**history, "prediction_epochs": prediction_epochs(run_id, scored, name)})


def _dataset_image_bytes(run: dict[str, Any], image_name: str) -> bytes:
    source = run["tags"].get(contract.TAG_DATASET_SOURCE)
    if not source:
        raise _bad("run is not linked to a dataset")
    root = os.environ.get("DATA_ROOT", ".")
    csv_path = resolve_under(root, source)
    if csv_path is None or not csv_path.is_file():
        raise MlflowNotFoundError(f"dataset not found: {source}")
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        column = contract.image_path_column(reader.fieldnames or [])
        rel = next((r[column] for r in reader if column and contract.image_key(r[column]) == image_name), None)
    image = resolve_under(root, rel) if rel else None
    if image is None or not image.is_file():
        raise MlflowNotFoundError(f"image {image_name} not found in {source}")
    return image.read_bytes()


@bp.get("/api/runs/<run_id>/prediction")
def run_prediction(run_id: str) -> ResponseReturnValue:
    """Return the dataset image with the run's predicted mask composited on it (JPEG)."""
    run = _per_sample_run(run_id)
    name = _image_name(request.args.get("image", ""))
    scored = contract.parse_scored_epochs(run["tags"])
    raw = request.args.get("epoch", "latest")
    if raw != "latest" and not raw.isdigit():
        raise _bad("epoch must be 'latest' or a number")
    epoch = scored[-1] if raw == "latest" else int(raw)
    try:
        alpha = max(0.0, min(1.0, float(request.args.get("alpha", 0.5))))
        max_size = max(64, min(4000, int(request.args.get("max_size", 1600))))
    except ValueError:
        raise _bad("alpha and max_size must be numbers") from None

    artifact = contract.prediction_path(epoch, name)
    mask_file = artifact_cache.fetch_artifact(mlflow_store.get_store(), run_id, artifact)
    if mask_file is None:
        raise MlflowNotFoundError(f"no prediction for {name} at epoch {epoch}")
    base = image_overlay.img_from_bytes(_dataset_image_bytes(run, name), max_size)
    result = image_overlay.composite_mask(base, image_overlay.mask_as_rgba(mask_file.read_bytes()), alpha)
    return Response(image_overlay.to_jpeg(result), mimetype="image/jpeg", headers={"X-Magnidata-Epoch": str(epoch)})


def _axes_for(run_id: str, epoch: int | str) -> run_axes.DerivedAxes:
    run = _per_sample_run(run_id)
    files = artifact_cache.sync_per_sample(mlflow_store.get_store(), run_id, with_split=False)
    scored = contract.parse_scored_epochs(run["tags"])
    wanted = scored[-1] if epoch == "latest" else int(epoch)
    if wanted not in scored:
        raise _bad(f"epoch {wanted} has no per-sample file in run {run_id[:8]} (scored: {', '.join(map(str, scored))})")
    return run_axes.derive_axes(files, wanted)


@bp.get("/api/runs/compare-samples")
def compare_samples() -> ResponseReturnValue:
    """Compare two runs image by image: hardness and loss change from run ``a`` to run ``b`` (CSV)."""
    a, b = _run_id(request.args.get("a", "")), _run_id(request.args.get("b", ""))
    epoch = request.args.get("epoch", "latest")
    if epoch != "latest" and not epoch.isdigit():
        raise _bad("epoch must be 'latest' or a number")
    axes_a, axes_b = _axes_for(a, epoch), _axes_for(b, epoch)
    try:
        merged = run_axes.compare_samples(axes_a, axes_b)
    except ValueError as err:
        raise _bad(str(err)) from None
    return Response(merged.to_csv(include_split=False), mimetype="text/csv", headers={"Cache-Control": "no-store"})
