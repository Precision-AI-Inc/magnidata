# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the trainer worker, against a real sqlite MLflow store."""

import os
import sys

import pytest

from precisionai.magnidata.training import contract

worker = pytest.importorskip("precisionai.magnidata.training.worker")

UI = {contract.TAG_ORIGIN: contract.ORIGIN_UI}
QUICK = {"epochs": "1", "batch_size": "4", "image_size": "32", "score_every": "1", "top_k_predictions": "2"}


@pytest.fixture
def env(mlflow_tracking, tmp_path):
    uri, client, _ = mlflow_tracking()
    experiment_id = client.create_experiment("magnidata/syn", artifact_location=(tmp_path / "worker-art").as_uri())
    return uri, client, experiment_id


def _run(client, experiment_id, tags=None, status="SCHEDULED", params=None):
    run = client.create_run(experiment_id, tags={**UI, **(tags or {})})
    run_id = run.info.run_id
    for key, value in (params or {}).items():
        client.log_param(run_id, key, value)
    if status != "RUNNING":
        client.update_run(run_id, status=status)
    return run_id


def test_find_scheduled_filters_and_orders(env):
    _, client, exp = env
    first = _run(client, exp)
    second = _run(client, exp)
    _run(client, exp, status="RUNNING")
    _run(client, exp, tags={contract.TAG_ORIGIN: contract.ORIGIN_EXTERNAL})
    other = client.create_experiment("elsewhere/x", artifact_location=None)
    _run(client, other)
    assert [r.info.run_id for r in worker.find_scheduled(client)] == [first, second]


def test_find_scheduled_without_experiments(mlflow_tracking):
    _, client, _ = mlflow_tracking()
    assert worker.find_scheduled(client) == []


def test_find_inference_requests_filters_and_orders(env):
    _, client, exp = env
    first = _run(client, exp, {contract.TAG_INFERENCE_STATUS: "queued"}, status="FINISHED")
    second = _run(client, exp, {contract.TAG_INFERENCE_STATUS: "queued"}, status="FINISHED")
    _run(client, exp, {contract.TAG_INFERENCE_STATUS: "ready"}, status="FINISHED")
    assert [run.info.run_id for run in worker.find_inference_requests(client)] == [first, second]


def test_claim_tags_run_once(env):
    _, client, exp = env
    run_id = _run(client, exp)
    assert worker.claim(client, run_id, "host:1:aaaa") is True
    assert client.get_run(run_id).data.tags[contract.TAG_WORKER] == "host:1:aaaa"
    assert worker.claim(client, run_id, "host:2:bbbb") is False


def test_claim_refuses_non_scheduled(env):
    _, client, exp = env
    run_id = _run(client, exp, status="RUNNING")
    assert worker.claim(client, run_id, "host:1:aaaa") is False


def test_recover_stale(env):
    _, client, exp = env
    me = worker.worker_id()
    host = me.split(":")[0]
    dead = _run(client, exp, {contract.TAG_WORKER: f"{host}:99999999:dead"}, "RUNNING")
    reused_pid = _run(client, exp, {contract.TAG_WORKER: f"{host}:{os.getpid()}:old0"}, "RUNNING")
    mine = _run(client, exp, {contract.TAG_WORKER: me}, "RUNNING")
    foreign = _run(client, exp, {contract.TAG_WORKER: "otherhost:99999999:dead"}, "RUNNING")
    untagged = _run(client, exp, status="RUNNING")

    assert sorted(worker.recover_stale(client, me)) == sorted([dead, reused_pid])
    for run_id in (dead, reused_pid):
        run = client.get_run(run_id)
        assert run.info.status == "FAILED"
        assert run.data.tags["magnidata.error"] == "worker restarted during run"
    for run_id in (mine, foreign, untagged):
        assert client.get_run(run_id).info.status == "RUNNING"


def test_trainer_argv_passes_device_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MAGNIDATA_TRAIN_DEVICE", "cuda")
    argv = worker.trainer_argv("abc", tmp_path, "http://mlflow:5000")
    assert argv[argv.index("--device") + 1] == "cuda"
    monkeypatch.delenv("MAGNIDATA_TRAIN_DEVICE")
    argv = worker.trainer_argv("abc", tmp_path, "http://mlflow:5000")
    assert argv[argv.index("--device") + 1] == "auto"


def test_agritune_argv_uses_product_adapter(tmp_path):
    argv = worker.agritune_argv("abc", tmp_path, "http://mlflow:5000")
    assert argv[2:4] == ["precisionai.magnidata.training.agritune", "--run-id"]
    assert argv[argv.index("--run-id") + 1] == "abc"


def test_inference_argv_passes_image_and_device(monkeypatch, tmp_path):
    monkeypatch.setenv("MAGNIDATA_TRAIN_DEVICE", "cpu")
    argv = worker.inference_argv("abc", "sample.png", tmp_path, "sqlite:///mlflow.db")
    assert argv[argv.index("--image") + 1] == "sample.png"
    assert argv[argv.index("--device") + 1] == "cpu"


def test_log_upload_interval_is_near_live():
    assert worker.LOG_UPLOAD_S <= 60


def test_run_one_fails_run_when_trainer_exits_badly(env, tmp_path):
    _, client, exp = env
    run_id = _run(client, exp)
    argv = [sys.executable, "-c", "import sys; print('boom happened'); sys.exit(3)"]
    assert worker.run_one(client, run_id, tmp_path, argv=argv) == "FAILED"
    error = client.get_run(run_id).data.tags["magnidata.error"]
    assert "code 3" in error
    assert "boom happened" in error
    assert [a.path for a in client.list_artifacts(run_id, "logs")] == ["logs/trainer.log"]


def test_run_one_skips_unclaimable_run(env, tmp_path):
    _, client, exp = env
    run_id = _run(client, exp, {contract.TAG_WORKER: "someone:1:else"})
    assert worker.run_one(client, run_id, tmp_path, argv=[sys.executable, "-c", "pass"]) is None
    assert client.get_run(run_id).info.status == "SCHEDULED"


def test_run_inference_records_success_and_failure(env, tmp_path):
    _, client, exp = env
    success = _run(
        client,
        exp,
        {contract.TAG_INFERENCE_STATUS: "queued", contract.TAG_INFERENCE_IMAGE: "a.png"},
        status="FINISHED",
    )
    assert worker.run_inference(client, success, tmp_path, [sys.executable, "-c", "print('saved')"])
    assert client.get_run(success).data.tags[contract.TAG_INFERENCE_STATUS] == "ready"
    assert [a.path for a in client.list_artifacts(success, "logs")] == ["logs/inference.log"]

    failed = _run(
        client,
        exp,
        {contract.TAG_INFERENCE_STATUS: "queued", contract.TAG_INFERENCE_IMAGE: "b.png"},
        status="FINISHED",
    )
    argv = [sys.executable, "-c", "import sys; print('bad inference'); sys.exit(4)"]
    assert worker.run_inference(client, failed, tmp_path, argv)
    run = client.get_run(failed)
    assert run.data.tags[contract.TAG_INFERENCE_STATUS] == "error"
    assert "code 4" in run.data.tags[contract.TAG_INFERENCE_ERROR]
    assert "bad inference" in run.data.tags[contract.TAG_INFERENCE_ERROR]


def test_run_one_trains_for_real(env, synthetic_dataset, monkeypatch):
    pytest.importorskip("torch", reason="the trainer subprocess needs torch (pip install -e '.[training]')")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    _, client, exp = env
    run_id = _run(
        client,
        exp,
        {contract.TAG_DATASET_SOURCE: "data_user/syn.csv", contract.TAG_PRESET: "tiny_unet"},
        params=QUICK,
    )
    assert worker.run_one(client, run_id, synthetic_dataset.root) == "FINISHED"
    run = client.get_run(run_id)
    assert run.data.tags[contract.TAG_SCORED_EPOCHS] == "1"
    assert "logs/trainer.log" in [a.path for a in client.list_artifacts(run_id, "logs")]
    assert (synthetic_dataset.root / "data_user" / worker.HEARTBEAT_FILENAME).exists()


def test_loop_touches_heartbeat_and_runs_nothing_when_idle(env, tmp_path):
    _, client, _ = env
    worker.loop(client, tmp_path, max_iterations=1)
    assert (tmp_path / "data_user" / worker.HEARTBEAT_FILENAME).is_file()


def test_loop_swallows_errors_and_stops(env, tmp_path, capsys, monkeypatch):
    _, client, _ = env

    def broken(_client):
        raise RuntimeError("mlflow hiccup")

    monkeypatch.setattr(worker, "find_scheduled", broken)
    worker.loop(client, tmp_path, max_iterations=1)
    assert "mlflow hiccup" in capsys.readouterr().err


def test_loop_runs_a_scheduled_run(env, tmp_path, monkeypatch):
    _, client, exp = env
    run_id = _run(client, exp)
    monkeypatch.setattr(worker, "trainer_argv", lambda *_args: [sys.executable, "-c", "import sys; sys.exit(1)"])
    worker.loop(client, tmp_path, max_iterations=1)
    assert client.get_run(run_id).info.status == "FAILED"


def test_stop_request_kills_child(env, tmp_path, monkeypatch):
    _, client, exp = env
    run_id = _run(client, exp)
    monkeypatch.setattr(worker, "TERM_GRACE_S", 5)
    worker.stop_requested.set()
    try:
        status = worker.run_one(client, run_id, tmp_path, argv=[sys.executable, "-c", "import time; time.sleep(60)"])
    finally:
        worker.stop_requested.clear()
    assert status == "KILLED"
