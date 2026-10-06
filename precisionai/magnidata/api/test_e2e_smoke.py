# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""End to end: launch from the API, train in the worker, read the results back through the API.

Run:  python -m pytest precisionai/magnidata/api/test_e2e_smoke.py -q
"""

import csv
import io

import experiments
import mlflow
import mlflow_store as store_module
import pytest
from flask import Flask

from precisionai.magnidata.training import contract, worker


@pytest.fixture
def client(mlflow_store, synthetic_dataset, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(synthetic_dataset.root))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # the trainer subprocess inherits this: stay on the CPU
    experiments.clear_cache()
    store_module.set_store(mlflow_store)
    app = Flask(__name__)
    app.register_blueprint(experiments.bp)
    yield app.test_client()
    store_module.set_store(None)
    experiments.clear_cache()


def test_launch_train_and_read_back(client, mlflow_store, synthetic_dataset):
    pytest.importorskip("torch", reason="the worker's trainer subprocess needs torch (pip install -e '.[training]')")
    launch = client.post(
        "/api/training/runs",
        json={
            "name": "smoke",
            "dataset_source": "data_user/syn.csv",
            "preset": "tiny_unet",
            "params": {"epochs": 2, "image_size": 128, "batch_size": 4, "top_k_predictions": 3},
        },
    )
    assert launch.status_code == 201
    run_id = launch.get_json()["run_id"]
    assert mlflow_store.get_run(run_id)["status"] == "SCHEDULED"

    # One pass of the worker claims the scheduled run and trains it to completion.
    mlflow_client = mlflow.MlflowClient(tracking_uri=mlflow_store.tracking_uri)
    worker.loop(mlflow_client, synthetic_dataset.root, max_iterations=1)

    detail = client.get(f"/api/runs/{run_id}").get_json()
    assert detail["status"] == "FINISHED", detail["tags"].get("magnidata.error")
    assert detail["scored_epochs"] == [1, 2]
    assert detail["has_per_sample"] is True

    metrics = client.get(f"/api/runs/{run_id}/metrics?keys=train_loss/loss,val_loss/loss").get_json()["metrics"]
    assert len(metrics["train_loss/loss"]) == 2
    assert len(metrics["val_loss/loss"]) == 2

    columns = client.get(f"/api/runs/{run_id}/columns")
    assert columns.status_code == 200
    rows = list(csv.DictReader(io.StringIO(columns.get_data(as_text=True))))
    assert sorted(r["image_name"] for r in rows) == synthetic_dataset.names
    assert all(0.0 <= float(r["hardness"]) <= 1.0 for r in rows)
    assert {"loss_total", "loss_total_rank", "loss_dice", "split"} <= set(rows[0])

    sample = client.get(f"/api/runs/{run_id}/samples/{synthetic_dataset.names[0]}").get_json()
    assert sample["epochs"] == [1, 2]
    assert len(sample["terms"]["total"]) == 2

    artifacts = {
        a["path"] for a in client.get(f"/api/runs/{run_id}/artifacts?path=magnidata/per_sample").get_json()["artifacts"]
    }
    assert artifacts == {contract.per_sample_path(1), contract.per_sample_path(2)}
