# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the API tests: a real MLflow store on a temporary sqlite database."""

import importlib
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

import pytest
from mlflow.entities import Metric
from mlflow_store import MlflowStore


def load_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **env: str | None) -> ModuleType:
    """Import ``app`` afresh against a temporary data root, with the given environment variables.

    app.py reads its configuration when it is imported, so every call reloads the module. A
    ``None`` value removes that variable. The environment is restored when the test ends.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("CONFIG_PATH", str(tmp_path / "confi.yaml"))
    (tmp_path / "confi.yaml").write_text("datasets: []\n")
    (tmp_path / "data_user").mkdir(exist_ok=True)
    for name, value in env.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    app_module = importlib.import_module("app")
    monkeypatch.setattr(app_module.db, "DB_PATH", str(tmp_path / "test.db"))
    importlib.reload(app_module)
    app_module.app.config["TESTING"] = True
    return app_module


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
    points = [
        Metric(key, value, 0, step) for key, values in (metrics or {}).items() for step, value in enumerate(values)
    ]
    for start in range(0, len(points), 1000):
        client.log_batch(run_id, metrics=points[start : start + 1000])
    for path, text in (artifacts or {}).items():
        client.log_text(run_id, text, path)
    client.set_terminated(run_id, status)
    return run_id


@pytest.fixture
def mlflow_store(tmp_path: Path, mlflow_db: Path) -> MlflowStore:
    """Return a real MlflowStore backed by sqlite and a local artifact root in ``tmp_path``."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    return MlflowStore(f"sqlite:///{mlflow_db}", artifact_root=artifacts.as_uri())
