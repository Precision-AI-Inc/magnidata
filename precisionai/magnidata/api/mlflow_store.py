# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Narrow adapter between the API and the local MLflow tracking server.

MLflow is the system of record for training experiments. Route handlers never touch
MLflow entities: every method here returns plain dicts and maps client failures onto
`MlflowUnavailableError` / `MlflowNotFoundError` (also exported as `MlflowUnavailable` / `MlflowNotFound`).
"""

import logging
import os
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mlflow
import mlflow.entities
import mlflow.exceptions
import requests
from mlflow.protos import databricks_pb2

DEFAULT_TRACKING_URI = "http://localhost:5052"
logger = logging.getLogger(__name__)

# A down server must never hang the API: bound the per-request wait and the retry backoff.
_HTTP_DEFAULTS = {
    "MLFLOW_HTTP_REQUEST_TIMEOUT": "10",
    "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "2",
    "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR": "1",
}


class MlflowUnavailableError(Exception):
    """The MLflow server could not be reached or failed to answer."""


class MlflowNotFoundError(Exception):
    """The requested experiment, run or artifact does not exist."""


MlflowUnavailable = MlflowUnavailableError
MlflowNotFound = MlflowNotFoundError


def _run_to_dict(run: mlflow.entities.Run) -> dict[str, Any]:
    info = run.info
    return {
        "run_id": info.run_id,
        "experiment_id": info.experiment_id,
        "name": info.run_name,
        "user": info.user_id,
        "status": info.status,
        "start_time": info.start_time,
        "end_time": info.end_time,
        "artifact_uri": info.artifact_uri,
        "params": dict(run.data.params),
        "metrics": dict(run.data.metrics),
        "tags": {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.log-model")},
    }


def _experiment_to_dict(exp: mlflow.entities.Experiment) -> dict[str, Any]:
    return {
        "experiment_id": exp.experiment_id,
        "name": exp.name,
        "artifact_location": exp.artifact_location,
        "lifecycle_stage": exp.lifecycle_stage,
        "creation_time": exp.creation_time,
        "last_update_time": exp.last_update_time,
        "tags": dict(exp.tags),
    }


def _metric_to_dict(metric: mlflow.entities.Metric) -> dict[str, Any]:
    return {"key": metric.key, "value": metric.value, "step": metric.step, "timestamp": metric.timestamp}


def _token(found: Any) -> str | None:
    """Return a paged result's continuation token as text (stores differ between str and bytes)."""
    token = found.token
    return token.decode() if isinstance(token, bytes) else token or None


def _file_to_dict(info: mlflow.entities.FileInfo) -> dict[str, Any]:
    return {"path": info.path, "is_dir": info.is_dir, "file_size": info.file_size}


class MlflowStore:
    """Plain-dict facade over `mlflow.MlflowClient`.

    Parameters
    ----------
    tracking_uri : str
        MLflow tracking URI (an ``http://`` server or a ``sqlite:///`` store).
    ui_url : str | None
        Browser-facing URL of the MLflow UI, used for deep links.
    artifact_root : str | None
        Directory (or URI) under which new experiments store artifacts. Leave unset against
        a server, which decides; a bare sqlite store has no sensible default.
    """

    def __init__(self, tracking_uri: str, ui_url: str | None = None, artifact_root: str | None = None) -> None:
        for key, value in _HTTP_DEFAULTS.items():
            os.environ.setdefault(key, value)
        self.tracking_uri = tracking_uri
        self.artifact_root = artifact_root
        self.ui_url = ui_url or (tracking_uri if tracking_uri.startswith("http") else None)
        self._client = mlflow.MlflowClient(tracking_uri=tracking_uri)

    def _call(self, what: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Run a client call, mapping its failures onto the adapter's exceptions."""
        try:
            return fn(*args, **kwargs)
        except mlflow.exceptions.MlflowException as err:
            if err.error_code == databricks_pb2.ErrorCode.Name(databricks_pb2.RESOURCE_DOES_NOT_EXIST):
                raise MlflowNotFoundError(f"{what} not found: {err.message}") from err
            raise MlflowUnavailableError(f"MLflow request for {what} failed: {err.message}") from err
        except (OSError, requests.RequestException) as err:
            raise MlflowUnavailableError(f"MLflow request for {what} failed: {err}") from err

    def status(self) -> dict[str, Any]:
        """Return reachability, tracking URI, server version (if known) and UI URL."""
        result: dict[str, Any] = {
            "reachable": False,
            "tracking_uri": self.tracking_uri,
            "ui_url": self.ui_url,
            "version": None,
            "error": None,
        }
        try:
            self._call("status", self._client.search_experiments, max_results=1)
        except MlflowUnavailableError as err:
            # The underlying error can name hosts, ports or paths; it is logged, and clients get a fixed message.
            logger.warning("MLflow status check failed: %s", err)
            result["error"] = f"MLflow is not reachable at {self.tracking_uri}"
            return result
        result["reachable"] = True
        result["version"] = self._server_version()
        return result

    def _server_version(self) -> str | None:
        if not self.tracking_uri.startswith("http"):
            return mlflow.__version__
        try:
            resp = requests.get(self.tracking_uri.rstrip("/") + "/version", timeout=5)
            resp.raise_for_status()
        except requests.RequestException:
            return None
        return resp.text.strip() or None

    def search_experiments_page(
        self, name_contains: str = "", max_results: int = 100, page_token: str | None = None
    ) -> dict[str, Any]:
        """Return one page of active experiments, newest first, with the next page token."""
        found = self._call(
            "experiments",
            self._client.search_experiments,
            max_results=max_results,
            filter_string=f"name LIKE '%{name_contains}%'" if name_contains else None,
            order_by=["last_update_time DESC"],
            page_token=page_token,
        )
        return {"items": [_experiment_to_dict(e) for e in found], "next_page_token": _token(found)}

    def search_experiments(self, max_results: int = 100) -> list[dict[str, Any]]:
        """Return the active experiments, newest first."""
        return self.search_experiments_page(max_results=max_results)["items"]

    def search_runs_page(
        self,
        experiment_ids: Sequence[str],
        filter_string: str = "",
        max_results: int = 100,
        order_by: Sequence[str] | None = None,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        """Return one page of runs matching an MLflow filter string, with the next page token."""
        found = self._call(
            "runs",
            self._client.search_runs,
            list(experiment_ids),
            filter_string=filter_string,
            max_results=max_results,
            order_by=list(order_by) if order_by else ["attributes.start_time DESC"],
            page_token=page_token,
        )
        return {"items": [_run_to_dict(r) for r in found], "next_page_token": _token(found)}

    def search_runs(
        self,
        experiment_ids: Sequence[str],
        filter_string: str = "",
        max_results: int = 100,
        order_by: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return runs of the given experiments matching an MLflow filter string."""
        return self.search_runs_page(experiment_ids, filter_string, max_results, order_by)["items"]

    def get_run(self, run_id: str) -> dict[str, Any]:
        """Return one run, raising `MlflowNotFoundError` if it does not exist."""
        return _run_to_dict(self._call(f"run {run_id}", self._client.get_run, run_id))

    def get_metric_history(self, run_id: str, key: str) -> list[dict[str, Any]]:
        """Return every logged (step, value, timestamp) of one metric."""
        found = self._call(f"run {run_id}", self._client.get_metric_history, run_id, key)
        return [_metric_to_dict(m) for m in found]

    def list_artifacts(self, run_id: str, path: str | None = None) -> list[dict[str, Any]]:
        """List the artifacts directly under ``path`` (the artifact root by default)."""
        found = self._call(f"run {run_id}", self._client.list_artifacts, run_id, path)
        return [_file_to_dict(f) for f in found]

    def download_artifact(self, run_id: str, path: str, dst_dir: str | Path) -> Path:
        """Download one artifact into ``dst_dir`` and return its local path."""
        Path(dst_dir).mkdir(parents=True, exist_ok=True)
        local = self._call(f"run {run_id} artifact {path}", self._client.download_artifacts, run_id, path, str(dst_dir))
        return Path(local)

    def get_or_create_experiment(self, name: str) -> str:
        """Return the id of the experiment called ``name``, creating it if needed."""
        found = self._call(f"experiment {name}", self._client.get_experiment_by_name, name)
        if found is not None:
            return found.experiment_id
        location = f"{self.artifact_root.rstrip('/')}/{name}" if self.artifact_root else None
        return self._call(f"experiment {name}", self._client.create_experiment, name, location)

    def create_run(
        self,
        experiment_id: str,
        run_name: str | None = None,
        tags: Mapping[str, str] | None = None,
        params: Mapping[str, str] | None = None,
        status: str = "RUNNING",
    ) -> dict[str, Any]:
        """Create a run, optionally with params, and put it in ``status`` (e.g. ``SCHEDULED``)."""
        run = self._call(
            "new run",
            self._client.create_run,
            experiment_id,
            tags=dict(tags or {}),
            run_name=run_name,
        )
        run_id = run.info.run_id
        for key, value in (params or {}).items():
            self._call(f"run {run_id}", self._client.log_param, run_id, key, value)
        if status != "RUNNING":
            self.set_status(run_id, status)
        return self.get_run(run_id)

    def set_status(self, run_id: str, status: str) -> None:
        """Set a run's status without ending it (used for SCHEDULED → RUNNING)."""
        self._call(f"run {run_id}", self._client.update_run, run_id, status=status)

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        """Set one tag on a run."""
        self._call(f"run {run_id}", self._client.set_tag, run_id, key, value)

    def set_terminated(self, run_id: str, status: str = "FINISHED") -> None:
        """End a run with a terminal status (FINISHED, FAILED or KILLED)."""
        self._call(f"run {run_id}", self._client.set_terminated, run_id, status)


class OfflineStore(MlflowStore):
    """Stand-in for a deployment that has no MLflow server, such as the Hugging Face Space.

    Every request fails at once with ``MlflowUnavailableError``, without a connection attempt or a log line, so a
    dashboard that keeps polling for runs costs nothing and does not fill the log.
    """

    def __init__(self) -> None:
        """Create the store; no connection is made until a request, and no request is ever sent."""
        super().__init__(DEFAULT_TRACKING_URI)

    def _call(self, what: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Refuse the request."""
        raise MlflowUnavailableError(f"MLflow request for {what} failed: this deployment has no MLflow server")

    def status(self) -> dict[str, Any]:
        """Report that there is no MLflow server."""
        return {
            "reachable": False,
            "tracking_uri": None,
            "ui_url": None,
            "version": None,
            "error": "This deployment has no MLflow server",
        }


_state: dict[str, MlflowStore | None] = {"store": None}
_state_lock = threading.Lock()


def get_store() -> MlflowStore:
    """Return the process-wide store, built from the environment on first use."""
    with _state_lock:
        store = _state["store"]
        if store is None:
            store = MlflowStore(
                os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
                os.environ.get("MAGNIDATA_MLFLOW_UI_URL"),
            )
            _state["store"] = store
        return store


def set_store(store: MlflowStore | None) -> None:
    """Replace the process-wide store (``None`` rebuilds it from the environment)."""
    with _state_lock:
        _state["store"] = store
