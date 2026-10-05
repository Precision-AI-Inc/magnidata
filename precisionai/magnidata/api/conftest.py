# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the API tests: a real MLflow store on a temporary sqlite database."""

from collections.abc import Mapping
from pathlib import Path

import pytest
from mlflow_store import MlflowStore


def seed_run(
    store: MlflowStore,
    experiment: str = "test-experiment",
    *,
    params: Mapping[str, str] | None = None,
    metrics: Mapping[str, list[float]] | None = None,
    tags: Mapping[str, str] | None = None,
    artifacts: Mapping[str, str] | None = None,
    status: str = "FINISHED",
) -> str:
    """Log a run with params, metric histories (step = list index), tags and text artifacts.

    Returns
    -------
    str
        The new run's id.
    """
    client = store._client
    experiment_id = store.get_or_create_experiment(experiment)
    run_id = client.create_run(experiment_id, tags=dict(tags or {})).info.run_id
    for key, value in (params or {}).items():
        client.log_param(run_id, key, value)
    for key, values in (metrics or {}).items():
        for step, value in enumerate(values):
            client.log_metric(run_id, key, value, step=step)
    for path, text in (artifacts or {}).items():
        client.log_text(run_id, text, path)
    client.set_terminated(run_id, status)
    return run_id


@pytest.fixture
def mlflow_store(tmp_path: Path) -> MlflowStore:
    """Return a real MlflowStore backed by sqlite and a local artifact root in ``tmp_path``."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    return MlflowStore(f"sqlite:///{tmp_path / 'mlflow.db'}", artifact_root=artifacts.as_uri())
