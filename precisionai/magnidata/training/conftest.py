# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the training tests: a real MLflow run on a temporary sqlite database."""

from collections.abc import Callable
from pathlib import Path

import mlflow
import pytest


@pytest.fixture
def mlflow_tracking(tmp_path: Path, mlflow_db: Path) -> Callable[[], tuple[str, mlflow.MlflowClient, str]]:
    """Return a factory creating a run; yields (tracking_uri, client, run_id)."""
    artifacts = tmp_path / "mlflow-artifacts"
    artifacts.mkdir()
    uri = f"sqlite:///{mlflow_db}"
    client = mlflow.MlflowClient(tracking_uri=uri)

    def make_run() -> tuple[str, mlflow.MlflowClient, str]:
        name = f"exp-{len(client.search_experiments())}"
        experiment_id = client.create_experiment(name, artifact_location=(artifacts / name).as_uri())
        return uri, client, client.create_run(experiment_id).info.run_id

    return make_run
