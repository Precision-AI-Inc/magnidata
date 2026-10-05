# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Recorder sink that logs to an MLflow run.

Kept apart from `recorder` so that importing the training package never imports mlflow.
"""

import time
from pathlib import Path
from typing import Any

from precisionai.magnidata.training import contract

try:
    import mlflow as _mlflow_module

    _mlflow: Any = _mlflow_module
    _MLFLOW_AVAILABLE = True
except ImportError:
    _mlflow = None
    _MLFLOW_AVAILABLE = False


class MlflowSink:
    """Sink that logs to an MLflow run; cancellation is the ``magnidata.cancel_requested`` tag.

    Parameters
    ----------
    run_id : str
        The run to log to.
    tracking_uri : str | None
        MLflow tracking URI; the environment's default when None.
    cancel_poll_s : float
        Minimum seconds between reads of the cancel tag, so a training loop can call
        `cancel_requested` every batch without hammering the server.
    """

    def __init__(self, run_id: str, tracking_uri: str | None = None, cancel_poll_s: float = 15.0) -> None:
        if not _MLFLOW_AVAILABLE:
            raise ImportError("mlflow is required for MlflowSink: pip install mlflow-skinny") from None
        self.run_id = run_id
        self.cancel_poll_s = cancel_poll_s
        self._client = _mlflow.MlflowClient(tracking_uri=tracking_uri)
        self._last_poll = float("-inf")
        self._cancelled = False

    def log_file(self, local_path: Path, artifact_dir: str) -> None:
        """Upload ``local_path`` to ``artifact_dir`` of the run."""
        self._client.log_artifact(self.run_id, str(local_path), artifact_path=artifact_dir)

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the run."""
        self._client.set_tag(self.run_id, key, value)

    def get_tag(self, key: str) -> str | None:
        """Return a tag of the run, or None when unset."""
        return self._client.get_run(self.run_id).data.tags.get(key)

    def cancel_requested(self) -> bool:
        """Return True once the cancel tag reads ``true`` (polled at most every ``cancel_poll_s``)."""
        if self._cancelled:
            return True
        now = time.monotonic()
        if now - self._last_poll >= self.cancel_poll_s:
            self._last_poll = now
            self._cancelled = self.get_tag(contract.TAG_CANCEL_REQUESTED) == "true"
        return self._cancelled
