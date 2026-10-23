# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for training presets, validation, launch and cancel, against a real sqlite MLflow store.

Run:  python -m pytest precisionai/magnidata/api/test_training_launch.py -q
"""

import os
import time

import experiments
import mlflow_store as store_module
import pytest
import trainer_backend
import training_presets
from flask import Flask

from precisionai.magnidata.training import contract

VALID = {"name": "first", "dataset_source": "data_user/syn.csv", "preset": "tiny_unet", "params": {"epochs": 3}}


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    folder = tmp_path / "data_user" / "syn"
    (folder / "images").mkdir(parents=True)
    (folder / "masks").mkdir()
    (folder / "images" / "a.png").write_bytes(b"x")
    (folder / "masks" / "a.png").write_bytes(b"x")
    (tmp_path / "data_user" / "syn.csv").write_text("image_path\ndata_user/syn/images/a.png\n")
    (tmp_path / "data_user" / "plain.csv").write_text("image_path\ndata_user/plain/images/a.png\n")
    return tmp_path


@pytest.fixture
def client(mlflow_store, root):
    experiments.clear_cache()
    store_module.set_store(mlflow_store)
    app = Flask(__name__)
    app.register_blueprint(experiments.bp)
    yield app.test_client()
    store_module.set_store(None)
    experiments.clear_cache()


def _heartbeat(root, age_s=0.0):
    path = root / "data_user" / ".trainer_heartbeat"
    path.write_text("")
    stamp = time.time() - age_s
    os.utime(path, (stamp, stamp))


def test_presets_endpoint_reports_worker_availability(client, root):
    body = client.get("/api/training/presets").get_json()
    assert [p["key"] for p in body["presets"]] == [
        "tiny_unet",
        "lraspp_mobilenet",
        "agritune_mlp_probe",
        "agritune_token_fpn",
    ]
    assert [p["engine"] for p in body["presets"]] == ["reference", "reference", "agritune", "agritune"]
    assert body["bounds"]["epochs"] == {"type": "int", "min": 1, "max": 200}
    assert body["loss_terms"] == ["dice", "cce", "bce"]
    assert body["mlflow"]["reachable"] is True
    assert body["backend"]["available"] is False
    assert body["backend"]["worker_heartbeat"] is None

    _heartbeat(root)
    assert client.get("/api/training/presets").get_json()["backend"]["available"] is True
    _heartbeat(root, age_s=60)
    stale = client.get("/api/training/presets").get_json()["backend"]
    assert stale["available"] is False
    assert "last seen" in stale["reason"]


def test_valid_launch_creates_scheduled_run(client, mlflow_store):
    resp = client.post("/api/training/runs", json=VALID)
    assert resp.status_code == 201
    body = resp.get_json()
    run = mlflow_store.get_run(body["run_id"])
    assert run["status"] == "SCHEDULED"
    assert run["name"] == "first"
    assert run["params"]["epochs"] == "3"
    assert run["params"]["loss_terms"] == "dice,cce"
    assert run["params"]["pretrained"] == "false"
    assert run["params"]["preset"] == "tiny_unet"
    tags = run["tags"]
    assert tags[contract.TAG_ORIGIN] == contract.ORIGIN_UI
    assert tags[contract.TAG_DATASET_SOURCE] == "data_user/syn.csv"
    assert tags[contract.TAG_DATASET_NAME] == "syn"
    assert tags[contract.TAG_PRESET] == "tiny_unet"
    assert tags[contract.TAG_ENGINE] == "reference"
    assert tags[contract.TAG_LOSS_TERMS] == "dice,cce"
    assert [e["name"] for e in mlflow_store.search_experiments() if e["name"].startswith("magnidata/")] == [
        "magnidata/syn"
    ]


def test_agritune_launch_records_engine_and_component_losses(client, mlflow_store):
    response = client.post(
        "/api/training/runs",
        json={
            "dataset_source": "data_user/syn.csv",
            "preset": "agritune_mlp_probe",
            "params": {"epochs": 2, "agritune_loss": "bce_dice", "feature_provider": "online"},
        },
    )

    assert response.status_code == 201
    run = mlflow_store.get_run(response.get_json()["run_id"])
    assert run["tags"][contract.TAG_ENGINE] == "agritune"
    assert run["tags"][contract.TAG_LOSS_TERMS] == "bce,dice"
    assert run["params"]["loss_terms"] == "bce,dice"
    assert run["params"]["feature_provider"] == "online"


@pytest.mark.parametrize("preset", ["agritune_mlp_probe", "agritune_token_fpn"])
def test_agritune_presets_launch_with_their_served_defaults(client, mlflow_store, preset):
    served = next(p for p in client.get("/api/training/presets").get_json()["presets"] if p["key"] == preset)
    params = {
        k: v
        for k, v in served["defaults"].items()
        if k not in {"loss_terms", "num_workers", "split_mode", "pretrained"}
    }

    response = client.post(
        "/api/training/runs", json={"dataset_source": "data_user/syn.csv", "preset": preset, "params": params}
    )

    assert response.status_code == 201


def test_agritune_presets_default_to_cached_features_preloaded_in_host_memory(client, mlflow_store):
    response = client.post(
        "/api/training/runs", json={"dataset_source": "data_user/syn.csv", "preset": "agritune_token_fpn"}
    )

    assert response.status_code == 201
    params = mlflow_store.get_run(response.get_json()["run_id"])["params"]
    assert params["feature_provider"] == "cached"
    assert params["feature_preload"] == "host"


def test_choosing_a_provider_that_cannot_preload_turns_preload_off(client, mlflow_store):
    response = client.post(
        "/api/training/runs",
        json={
            "dataset_source": "data_user/syn.csv",
            "preset": "agritune_mlp_probe",
            "params": {"feature_provider": "hybrid", "augmentation_mode": "online"},
        },
    )

    assert response.status_code == 201
    params = mlflow_store.get_run(response.get_json()["run_id"])["params"]
    assert params["feature_preload"] == "none"
    assert params["augmentation_mode"] == "online"


def test_cached_features_can_be_preloaded_on_the_device(client, mlflow_store):
    response = client.post(
        "/api/training/runs",
        json={
            "dataset_source": "data_user/syn.csv",
            "preset": "agritune_mlp_probe",
            "params": {"feature_preload": "device"},
        },
    )

    assert response.status_code == 201
    assert mlflow_store.get_run(response.get_json()["run_id"])["params"]["feature_preload"] == "device"


def test_agritune_launch_records_the_report_settings(client, mlflow_store):
    params = {
        "input_resolution": "native",
        "batch_size": 128,
        "accumulation_steps": 32,
        "grad_clip_norm": 1.0,
        "lr": 0.001,
        "weight_decay": 0.01,
        "amp": False,
        "val_fraction": 0.25,
        "epochs": 100,
        "score_every": 100,
    }
    response = client.post(
        "/api/training/runs",
        json={"dataset_source": "data_user/syn.csv", "preset": "agritune_mlp_probe", "params": params},
    )

    assert response.status_code == 201
    logged = mlflow_store.get_run(response.get_json()["run_id"])["params"]
    assert logged["input_resolution"] == "native"
    assert logged["batch_size"] == "128"
    assert logged["accumulation_steps"] == "32"
    assert logged["grad_clip_norm"] == "1.0"
    assert logged["weight_decay"] == "0.01"


def test_reference_presets_train_at_native_resolution_on_request(client, mlflow_store):
    default = client.post("/api/training/runs", json=VALID).get_json()["run_id"]
    client.post(f"/api/runs/{default}/cancel")
    native = client.post("/api/training/runs", json={**VALID, "params": {"input_resolution": "native"}})

    assert native.status_code == 201
    assert mlflow_store.get_run(default)["params"]["input_resolution"] == "image_size"
    assert mlflow_store.get_run(native.get_json()["run_id"])["params"]["input_resolution"] == "native"


def test_presets_serve_the_advanced_agritune_settings(client):
    advanced = client.get("/api/training/presets").get_json()["advanced"]

    keys = {item["key"] for item in advanced}
    assert {
        "optimizer.name",
        "scheduler.name",
        "decoder.hidden_dims",
        "feature_augmentation.patch_dropout_probability",
    } <= keys
    assert next(item for item in advanced if item["key"] == "decoder.hidden_dim")["presets"] == ["agritune_token_fpn"]


def test_advanced_agritune_settings_are_validated_and_recorded(client, mlflow_store):
    params = {
        "optimizer.name": "sgd",
        "optimizer.betas": [0.8, 0.99],
        "scheduler.name": "cosine",
        "scheduler.warmup_steps": 100,
        "decoder.hidden_dims": [256, 128],
        "loss.class_weights": [],
        "data.pin_memory": True,
    }
    response = client.post(
        "/api/training/runs",
        json={"dataset_source": "data_user/syn.csv", "preset": "agritune_mlp_probe", "params": params},
    )

    assert response.status_code == 201
    logged = mlflow_store.get_run(response.get_json()["run_id"])["params"]
    assert logged["optimizer.name"] == "sgd"
    assert logged["optimizer.betas"] == "0.8,0.99"
    assert logged["scheduler.warmup_steps"] == "100"
    assert logged["decoder.hidden_dims"] == "256,128"
    assert logged["loss.class_weights"] == ""
    assert logged["data.pin_memory"] == "true"
    assert "trainer.precision" not in logged


@pytest.mark.parametrize(
    ("preset", "params", "message"),
    [
        ("tiny_unet", {"optimizer.name": "sgd"}, "only available for AgriTune presets"),
        ("agritune_token_fpn", {"decoder.hidden_dims": [64]}, "only available for agritune_mlp_probe"),
        ("agritune_mlp_probe", {"scheduler.name": "warm"}, "must be one of"),
        ("agritune_mlp_probe", {"augmentation.brightness_range": [2, 1]}, "must not exceed"),
    ],
)
def test_advanced_settings_are_refused_where_they_do_not_apply(client, preset, params, message):
    resp = client.post("/api/training/runs", json={**VALID, "preset": preset, "params": params})

    assert resp.status_code == 400
    assert message in resp.get_json()["error"]


def test_reference_presets_accept_weight_decay(client, mlflow_store):
    response = client.post("/api/training/runs", json={**VALID, "params": {"weight_decay": 0.05}})

    assert response.status_code == 201
    assert mlflow_store.get_run(response.get_json()["run_id"])["params"]["weight_decay"] == "0.05"


def test_launch_records_parent_and_weights(client, mlflow_store):
    parent = client.post("/api/training/runs", json=VALID).get_json()["run_id"]
    client.post(f"/api/runs/{parent}/cancel")
    body = {**VALID, "parent_run_id": parent, "params": {"loss_weights": {"dice": 2.0}, "loss_terms": ["dice", "bce"]}}
    run = mlflow_store.get_run(client.post("/api/training/runs", json=body).get_json()["run_id"])
    assert run["tags"][contract.TAG_PARENT_RUN_ID] == parent
    assert run["params"]["weight_dice"] == "2.0"
    assert run["params"]["loss_terms"] == "dice,bce"


def test_experiment_prefix_is_configurable(client, mlflow_store, monkeypatch):
    monkeypatch.setenv("MAGNIDATA_MLFLOW_EXPERIMENT_PREFIX", "lab")
    assert client.post("/api/training/runs", json=VALID).status_code == 201
    assert "lab/syn" in [e["name"] for e in mlflow_store.search_experiments()]


@pytest.mark.parametrize(
    ("params", "field"),
    [
        ({"epochs": 0}, "params.epochs"),
        ({"epochs": 201}, "params.epochs"),
        ({"epochs": 1.5}, "params.epochs"),
        ({"epochs": True}, "params.epochs"),
        ({"epochs": 5, "score_every": 6}, "params.score_every"),
        ({"score_every": 0}, "params.score_every"),
        ({"batch_size": 257}, "params.batch_size"),
        ({"weight_decay": 2}, "params.weight_decay"),
        ({"image_size": 100}, "params.image_size"),
        ({"image_size": 130}, "params.image_size"),
        ({"image_size": 2048}, "params.image_size"),
        ({"lr": 0}, "params.lr"),
        ({"lr": 0.5}, "params.lr"),
        ({"lr": "fast"}, "params.lr"),
        ({"val_fraction": 0.01}, "params.val_fraction"),
        ({"val_fraction": 0.9}, "params.val_fraction"),
        ({"seed": -1}, "params.seed"),
        ({"split_mode": "stratified"}, "params.split_mode"),
        ({"loss_terms": []}, "params.loss_terms"),
        ({"loss_terms": ["dice", "focal"]}, "params.loss_terms"),
        ({"loss_terms": ["dice", "dice"]}, "params.loss_terms"),
        ({"loss_weights": {"dice": 11}}, "params.loss_weights.dice"),
        ({"loss_weights": {"bce": 1}}, "params.loss_weights.bce"),
        ({"top_k_predictions": 201}, "params.top_k_predictions"),
        ({"pretrained": "yes"}, "params.pretrained"),
        ({"amp": "yes"}, "params.amp"),
        ({"num_workers": -1}, "params.num_workers"),
        ({"num_workers": 17}, "params.num_workers"),
        ({"num_workers": 2.5}, "params.num_workers"),
        ({"command": "rm -rf /"}, "params: unknown parameter"),
    ],
)
def test_parameter_violations_name_the_field(client, params, field):
    resp = client.post("/api/training/runs", json={**VALID, "params": params})
    assert resp.status_code == 400
    assert field in resp.get_json()["error"]


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"preset": "huge"}, "preset"),
        ({"command": "rm -rf /"}, "unknown field"),
        ({"dataset_source": "data_user/plain.csv"}, "no segmentation masks"),
        ({"dataset_source": "data_user/missing.csv"}, "dataset not found"),
        ({"dataset_source": "../etc/passwd"}, "dataset not found"),
        ({"dataset_source": ""}, "dataset_source"),
        ({"parent_run_id": "nope"}, "parent_run_id"),
        ({"parent_run_id": "0" * 32}, "parent_run_id: run not found"),
        ({"name": "x" * 101}, "name"),
        ({"params": ["epochs"]}, "params"),
    ],
)
def test_other_request_violations(client, patch, message):
    resp = client.post("/api/training/runs", json={**VALID, **patch})
    assert resp.status_code == 400
    assert message in resp.get_json()["error"]


def test_non_object_body_is_rejected(client):
    assert client.post("/api/training/runs", json=["a"]).status_code == 400
    assert client.post("/api/training/runs", data="not json").status_code == 400


def test_second_launch_while_active_is_refused(client):
    assert client.post("/api/training/runs", json=VALID).status_code == 201
    resp = client.post("/api/training/runs", json=VALID)
    assert resp.status_code == 409
    assert "scheduled or running" in resp.get_json()["error"]


def test_running_run_also_blocks_launch(client, mlflow_store):
    run_id = client.post("/api/training/runs", json=VALID).get_json()["run_id"]
    mlflow_store.set_status(run_id, "RUNNING")
    assert client.post("/api/training/runs", json=VALID).status_code == 409
    mlflow_store.set_terminated(run_id, "FINISHED")
    assert client.post("/api/training/runs", json=VALID).status_code == 201


def test_cancel_scheduled_run_kills_it(client, mlflow_store):
    run_id = client.post("/api/training/runs", json=VALID).get_json()["run_id"]
    body = client.post(f"/api/runs/{run_id}/cancel").get_json()
    assert body["status"] == "KILLED"
    assert body["cancel_requested"] is False
    assert mlflow_store.get_run(run_id)["status"] == "KILLED"


def test_cancel_running_run_sets_tag(client, mlflow_store):
    run_id = client.post("/api/training/runs", json=VALID).get_json()["run_id"]
    mlflow_store.set_status(run_id, "RUNNING")
    body = client.post(f"/api/runs/{run_id}/cancel").get_json()
    assert body["cancel_requested"] is True
    run = mlflow_store.get_run(run_id)
    assert run["status"] == "RUNNING"
    assert run["tags"][contract.TAG_CANCEL_REQUESTED] == "true"


def test_cancel_finished_run_conflicts(client, mlflow_store):
    run_id = client.post("/api/training/runs", json=VALID).get_json()["run_id"]
    mlflow_store.set_terminated(run_id, "FINISHED")
    resp = client.post(f"/api/runs/{run_id}/cancel")
    assert resp.status_code == 409
    assert "FINISHED" in resp.get_json()["error"]
    assert client.post("/api/runs/bad/cancel").status_code == 400
    assert client.post(f"/api/runs/{'0' * 32}/cancel").status_code == 404


def test_validate_launch_defaults_are_complete():
    spec = training_presets.validate_launch({"dataset_source": "a/b.csv", "preset": "lraspp_mobilenet"})
    assert spec.name == "lraspp_mobilenet on b"
    assert spec.params["pretrained"] == "true"
    assert spec.params["epochs"] == "20"
    assert spec.params["split_mode"] == "random"
    assert spec.loss_terms == ["dice", "cce"]


def test_amp_and_loader_workers_are_launch_params():
    spec = training_presets.validate_launch({"dataset_source": "a/b.csv", "preset": "tiny_unet"})
    assert spec.params["amp"] == "true"
    assert "num_workers" not in spec.params
    tuned = training_presets.validate_launch(
        {"dataset_source": "a/b.csv", "preset": "tiny_unet", "params": {"amp": False, "num_workers": 4}}
    )
    assert tuned.params["amp"] == "false"
    assert tuned.params["num_workers"] == "4"


def test_custom_backend_can_be_injected(client):
    class Fake:
        def submit(self, spec):
            return "0" * 32

        def cancel(self, run_id):
            return "KILLED"

        def available(self):
            return {"available": True, "reason": None, "worker_heartbeat": 0}

    trainer_backend.set_backend(Fake())
    try:
        assert client.get("/api/training/presets").get_json()["backend"]["available"] is True
    finally:
        trainer_backend.set_backend(None)


def test_launch_and_cancel_are_visible_immediately_despite_the_runs_cache(client, mlflow_store):
    url = "/api/runs?dataset=data_user/syn.csv"
    assert client.get(url).get_json()["runs"] == []
    run_id = client.post("/api/training/runs", json=VALID).get_json()["run_id"]
    listed = client.get(url).get_json()["runs"]
    assert [r["run_id"] for r in listed] == [run_id]
    assert listed[0]["status"] == "SCHEDULED"
    client.post(f"/api/runs/{run_id}/cancel")
    assert client.get(url).get_json()["runs"][0]["status"] == "KILLED"


@pytest.mark.parametrize(
    ("preset", "params", "field"),
    [
        ("tiny_unet", {"feature_provider": "online"}, "params.feature_provider"),
        ("tiny_unet", {"augmentation_mode": "online"}, "params.augmentation_mode"),
        ("lraspp_mobilenet", {"agritune_loss": "dice"}, "params.agritune_loss"),
        ("agritune_mlp_probe", {"loss_terms": ["dice"]}, "params.loss_terms"),
        ("agritune_mlp_probe", {"num_workers": 2}, "params.num_workers"),
        ("agritune_mlp_probe", {"split_mode": "column"}, "params.split_mode"),
        ("agritune_token_fpn", {"pretrained": True}, "params.pretrained"),
        ("agritune_mlp_probe", {"feature_provider": "bogus"}, "params.feature_provider"),
        ("tiny_unet", {"feature_preload": "host"}, "params.feature_preload"),
        ("agritune_mlp_probe", {"feature_preload": "disk"}, "params.feature_preload"),
        ("agritune_mlp_probe", {"feature_provider": "hybrid", "feature_preload": "host"}, "params.feature_preload"),
        ("agritune_mlp_probe", {"augmentation_mode": "online"}, "params.augmentation_mode"),
        ("tiny_unet", {"accumulation_steps": 4}, "params.accumulation_steps"),
        ("tiny_unet", {"grad_clip_norm": 1.0}, "params.grad_clip_norm"),
        ("tiny_unet", {"input_resolution": "full"}, "params.input_resolution"),
        ("agritune_mlp_probe", {"input_resolution": "full"}, "params.input_resolution"),
        ("agritune_mlp_probe", {"accumulation_steps": 0}, "params.accumulation_steps"),
        ("agritune_mlp_probe", {"agritune_loss": "focal"}, "params.agritune_loss"),
    ],
)
def test_engine_specific_settings_are_rejected_for_the_wrong_engine(client, preset, params, field):
    resp = client.post("/api/training/runs", json={**VALID, "preset": preset, "params": params})

    assert resp.status_code == 400
    assert field in resp.get_json()["error"]


def test_agritune_loss_weights_follow_the_chosen_objective(client, mlflow_store):
    resp = client.post(
        "/api/training/runs",
        json={
            **VALID,
            "preset": "agritune_mlp_probe",
            "params": {"agritune_loss": "bce_dice", "loss_weights": {"bce": 2, "dice": 0.5}},
        },
    )

    run = mlflow_store.get_run(resp.get_json()["run_id"])
    assert run["params"]["weight_bce"] == "2.0"
    assert run["params"]["weight_dice"] == "0.5"
    assert run["tags"][contract.TAG_LOSS_TERMS] == "bce,dice"
