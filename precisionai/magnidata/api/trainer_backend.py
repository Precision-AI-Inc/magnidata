# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Where training jobs go once they are launched.

The only backend is MLflow itself: launching creates a ``SCHEDULED`` run and the trainer
worker (a separate container with torch) claims it. A different executor, for example Nomad,
would be another `TrainerBackend` whose ``submit`` creates the same SCHEDULED run and
dispatches a job that runs the trainer on it; the API and dashboard would not change.
"""

import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Protocol

from mlflow_store import MlflowStore
from training_presets import LaunchSpec

from precisionai.magnidata.training import contract

HEARTBEAT_FILENAME = ".trainer_heartbeat"
HEARTBEAT_MAX_AGE_S = 30
DEFAULT_EXPERIMENT_PREFIX = "magnidata"
ACTIVE_STATUSES = (contract.RunStatus.SCHEDULED.value, contract.RunStatus.RUNNING.value)


class RunNotCancellableError(Exception):
    """The run already reached a terminal status."""


class TrainerBusyError(Exception):
    """Another UI-launched run is still scheduled or running."""


class TrainerBackend(Protocol):
    """Starts, stops and reports on training jobs."""

    def submit(self, spec: LaunchSpec) -> str:
        """Queue a job and return its MLflow run id."""
        ...

    def cancel(self, run_id: str) -> str:
        """Cancel a job and return the resulting status."""
        ...

    def available(self) -> dict[str, Any]:
        """Return ``{"available", "reason", "worker_heartbeat"}``."""
        ...


def slugify(text: str) -> str:
    """Return a lowercase, dash-separated slug safe for an experiment name."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "dataset"


def heartbeat_path() -> Path:
    """Return the file the trainer worker touches while alive."""
    return Path(os.environ.get("DATA_ROOT", ".")) / "data_user" / HEARTBEAT_FILENAME


class MlflowQueueBackend:
    """Backend that uses MLflow ``SCHEDULED`` runs as the job queue.

    Parameters
    ----------
    store : MlflowStore
        The MLflow adapter.
    """

    def __init__(self, store: MlflowStore) -> None:
        self.store = store
        self._lock = threading.Lock()

    @staticmethod
    def experiment_prefix() -> str:
        """Return the experiment-name prefix (``MAGNIDATA_MLFLOW_EXPERIMENT_PREFIX``)."""
        return os.environ.get("MAGNIDATA_MLFLOW_EXPERIMENT_PREFIX", DEFAULT_EXPERIMENT_PREFIX)

    def active_run_id(self) -> str | None:
        """Return the id of a UI-launched run that is scheduled or running, if any."""
        prefix = self.experiment_prefix() + "/"
        ids = [e["experiment_id"] for e in self.store.search_experiments(1000) if e["name"].startswith(prefix)]
        if not ids:
            return None
        for status in ACTIVE_STATUSES:
            found = self.store.search_runs(
                ids,
                f"tags.`{contract.TAG_ORIGIN}` = '{contract.ORIGIN_UI}' AND attributes.status = '{status}'",
                max_results=1,
            )
            if found:
                return found[0]["run_id"]
        return None

    def submit(self, spec: LaunchSpec) -> str:
        """Create the ``SCHEDULED`` run for ``spec``; raises `TrainerBusyError` if one is active."""
        with self._lock:
            if self.active_run_id() is not None:
                raise TrainerBusyError("another training run is scheduled or running")
            stem = Path(spec.dataset_source).stem
            experiment_id = self.store.get_or_create_experiment(f"{self.experiment_prefix()}/{slugify(stem)}")
            tags = {
                contract.TAG_ORIGIN: contract.ORIGIN_UI,
                contract.TAG_DATASET_SOURCE: spec.dataset_source,
                contract.TAG_DATASET_NAME: stem,
                contract.TAG_PRESET: spec.preset,
                contract.TAG_ENGINE: spec.engine,
                contract.TAG_LOSS_TERMS: ",".join(spec.loss_terms),
            }
            if spec.parent_run_id:
                tags[contract.TAG_PARENT_RUN_ID] = spec.parent_run_id
            run = self.store.create_run(
                experiment_id,
                run_name=spec.name,
                tags=tags,
                params=spec.params,
                status=contract.RunStatus.SCHEDULED.value,
            )
            return run["run_id"]

    def cancel(self, run_id: str) -> str:
        """Kill a scheduled run, or ask a running one to stop; terminal runs raise `RunNotCancellableError`."""
        status = self.store.get_run(run_id)["status"]
        if status == contract.RunStatus.SCHEDULED.value:
            self.store.set_terminated(run_id, contract.RunStatus.KILLED.value)
            return contract.RunStatus.KILLED.value
        if status == contract.RunStatus.RUNNING.value:
            self.store.set_tag(run_id, contract.TAG_CANCEL_REQUESTED, "true")
            return contract.RunStatus.RUNNING.value
        raise RunNotCancellableError(f"run already {status}")

    def available(self) -> dict[str, Any]:
        """Report whether a worker heartbeat younger than 30 s exists."""
        path = heartbeat_path()
        try:
            age = time.time() - path.stat().st_mtime
        except OSError:
            return {"available": False, "reason": "no trainer worker has reported in", "worker_heartbeat": None}
        if age > HEARTBEAT_MAX_AGE_S:
            return {
                "available": False,
                "reason": f"trainer worker last seen {int(age)} s ago",
                "worker_heartbeat": age,
            }
        return {"available": True, "reason": None, "worker_heartbeat": age}


_state: dict[str, TrainerBackend | None] = {"backend": None}


def get_backend(store: MlflowStore) -> TrainerBackend:
    """Return the configured backend (an `MlflowQueueBackend` unless one was set)."""
    backend = _state["backend"]
    return backend if backend is not None else MlflowQueueBackend(store)


def set_backend(backend: TrainerBackend | None) -> None:
    """Override the backend (``None`` restores the default)."""
    _state["backend"] = backend
