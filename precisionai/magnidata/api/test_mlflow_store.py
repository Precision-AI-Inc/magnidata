# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MLflow adapter, against a real sqlite-backed MLflow store.

Run:  python -m pytest precisionai/magnidata/api/test_mlflow_store.py -q
"""

import time

import mlflow_store
import pytest
from conftest import seed_run
from mlflow_store import MlflowNotFound, MlflowStore, MlflowUnavailable

from precisionai.magnidata.training import contract


def test_status_reachable(mlflow_store):
    status = mlflow_store.status()
    assert status["reachable"] is True
    assert status["error"] is None
    assert status["tracking_uri"].startswith("sqlite:///")
    assert status["version"]


def test_status_unreachable_returns_quickly(closed_port_url):
    url = closed_port_url
    store = MlflowStore(url)
    started = time.monotonic()
    status = store.status()
    assert time.monotonic() - started < 15
    assert status["reachable"] is False
    # A fixed message: the underlying connection error is logged, not returned to clients.
    assert status["error"] == f"MLflow is not reachable at {url}"


def test_run_entities_are_plain_dicts(mlflow_store):
    run_id = seed_run(
        mlflow_store,
        params={"lr": "0.001"},
        metrics={"train_loss/loss": [0.9, 0.5]},
        tags={contract.TAG_ORIGIN: contract.ORIGIN_EXTERNAL},
    )
    run = mlflow_store.get_run(run_id)
    assert run["run_id"] == run_id
    assert run["status"] == "FINISHED"
    assert run["params"] == {"lr": "0.001"}
    assert run["metrics"] == {"train_loss/loss": 0.5}
    assert run["tags"][contract.TAG_ORIGIN] == contract.ORIGIN_EXTERNAL
    assert isinstance(run["tags"], dict)


def test_get_run_unknown_raises_not_found(mlflow_store):
    with pytest.raises(MlflowNotFound, match="run"):
        mlflow_store.get_run("does-not-exist")


def test_search_experiments_and_runs(mlflow_store):
    run_id = seed_run(mlflow_store, experiment="exp-a", tags={contract.TAG_ORIGIN: "ui"})
    seed_run(mlflow_store, experiment="exp-a", tags={contract.TAG_ORIGIN: "external"})
    experiments = {e["name"]: e["experiment_id"] for e in mlflow_store.search_experiments()}
    assert "exp-a" in experiments
    runs = mlflow_store.search_runs([experiments["exp-a"]], filter_string="tags.`magnidata.origin` = 'ui'")
    assert [r["run_id"] for r in runs] == [run_id]


def test_metric_history(mlflow_store):
    run_id = seed_run(mlflow_store, metrics={"val_loss/loss": [1.0, 0.7, 0.4]})
    history = mlflow_store.get_metric_history(run_id, "val_loss/loss")
    assert [(m["step"], m["value"]) for m in history] == [(0, 1.0), (1, 0.7), (2, 0.4)]


def test_list_and_download_artifacts(mlflow_store, tmp_path):
    csv_text = "image_name,split,loss_total\na.png,train,0.5\n"
    run_id = seed_run(mlflow_store, artifacts={contract.per_sample_path(1): csv_text})
    top = mlflow_store.list_artifacts(run_id)
    assert [(f["path"], f["is_dir"]) for f in top] == [("magnidata", True)]
    files = mlflow_store.list_artifacts(run_id, "magnidata/per_sample")
    assert [f["path"] for f in files] == [contract.per_sample_path(1)]
    local = mlflow_store.download_artifact(run_id, contract.per_sample_path(1), tmp_path / "dl")
    assert local.read_text() == csv_text


def test_create_scheduled_run_and_update(mlflow_store):
    experiment_id = mlflow_store.get_or_create_experiment("train")
    assert mlflow_store.get_or_create_experiment("train") == experiment_id
    run = mlflow_store.create_run(
        experiment_id,
        run_name="r1",
        tags={contract.TAG_ORIGIN: contract.ORIGIN_UI},
        params={"epochs": "3"},
        status=contract.RunStatus.SCHEDULED.value,
    )
    assert run["status"] == "SCHEDULED"
    assert run["params"] == {"epochs": "3"}

    mlflow_store.set_status(run["run_id"], "RUNNING")
    mlflow_store.set_tag(run["run_id"], contract.TAG_CANCEL_REQUESTED, "true")
    mlflow_store.set_terminated(run["run_id"], "KILLED")
    done = mlflow_store.get_run(run["run_id"])
    assert done["status"] == "KILLED"
    assert done["tags"][contract.TAG_CANCEL_REQUESTED] == "true"


def test_get_store_builds_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'env.db'}")
    monkeypatch.setenv("MAGNIDATA_MLFLOW_UI_URL", "http://localhost:5052")
    mlflow_store.set_store(None)
    try:
        store = mlflow_store.get_store()
        assert store.tracking_uri.endswith("env.db")
        assert store.ui_url == "http://localhost:5052"
        assert mlflow_store.get_store() is store
    finally:
        mlflow_store.set_store(None)


def test_unreachable_server_maps_to_unavailable(closed_port_url):
    store = MlflowStore(closed_port_url)
    with pytest.raises(MlflowUnavailable, match="MLflow"):
        store.get_run("abc")
