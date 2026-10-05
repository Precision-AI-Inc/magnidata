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


def test_run_one_trains_for_real(env, synthetic_dataset, monkeypatch):
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
