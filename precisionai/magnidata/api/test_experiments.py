# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MLflow browsing endpoints, against a real sqlite-backed MLflow store.

Run:  python -m pytest precisionai/magnidata/api/test_experiments.py -q
"""

import csv
import io

import experiments
import mlflow_store as store_module
import numpy as np
import pytest
from conftest import seed_run
from flask import Flask
from mlflow_store import MlflowStore
from PIL import Image

from precisionai.magnidata.training import contract


@pytest.fixture
def client(mlflow_store, tmp_path, monkeypatch):
    monkeypatch.setenv(
        "DATA_ROOT", str(tmp_path / "data-root")
    )  # keeps the artifact cache out of the working directory
    experiments.clear_cache()
    store_module.set_store(mlflow_store)
    app = Flask(__name__)
    app.register_blueprint(experiments.bp)
    yield app.test_client()
    store_module.set_store(None)
    experiments.clear_cache()


@pytest.fixture
def seeded(mlflow_store, client, monkeypatch):
    monkeypatch.setattr(experiments, "RUNS_CACHE_TTL_S", 0)
    ids = {}
    for exp in ("alpha", "beta"):
        for i in range(3):
            tags = {contract.TAG_DATASET_SOURCE: "data_user/a.csv" if exp == "alpha" else "data_user/b.csv"}
            if i == 0:
                tags[contract.TAG_PER_SAMPLE_SCHEMA] = contract.SCHEMA_VERSION
                tags[contract.TAG_SCORED_EPOCHS] = "1,2"
            ids[(exp, i)] = seed_run(
                mlflow_store,
                exp,
                params={"lr": f"0.0{i + 1}"},
                metrics={"val_loss/loss": [1.0 - 0.1 * i, 0.5 - 0.1 * i], "train_loss/loss": [1.2, 0.8]},
                tags=tags,
                status="FAILED" if i == 2 else "FINISHED",
                artifacts={"magnidata/note.txt": "hello", "magnidata/pred.png": "not really a png"},
            )
    return ids


def test_list_experiments_and_query(client, seeded):
    names = {e["name"] for e in client.get("/api/experiments").get_json()["experiments"]}
    assert {"alpha", "beta"} <= names
    found = client.get("/api/experiments?query=alp").get_json()["experiments"]
    assert [e["name"] for e in found] == ["alpha"]
    assert client.get("/api/experiments?query=a'b").status_code == 400


def _experiment_id(client, name):
    return next(
        e["experiment_id"] for e in client.get("/api/experiments").get_json()["experiments"] if e["name"] == name
    )


def test_search_runs_by_experiment_dataset_and_status(client, seeded):
    alpha = _experiment_id(client, "alpha")
    runs = client.get(f"/api/runs?experiment_ids={alpha}").get_json()["runs"]
    assert {r["run_id"] for r in runs} == {seeded[("alpha", i)] for i in range(3)}

    by_dataset = client.get("/api/runs?dataset=data_user/b.csv").get_json()["runs"]
    assert {r["run_id"] for r in by_dataset} == {seeded[("beta", i)] for i in range(3)}

    failed = client.get("/api/runs?status=FAILED").get_json()["runs"]
    assert {r["run_id"] for r in failed} == {seeded[("alpha", 2)], seeded[("beta", 2)]}
    assert client.get("/api/runs?status=BOGUS").status_code == 400


def test_run_summary_fields(client, seeded):
    run = next(r for r in client.get("/api/runs?dataset=data_user/a.csv").get_json()["runs"] if r["has_per_sample"])
    assert run["run_id"] == seeded[("alpha", 0)]
    assert run["params"] == {"lr": "0.01"}
    assert run["metrics"]["val_loss/loss"] == 0.5
    assert run["magnidata"][contract.TAG_DATASET_SOURCE] == "data_user/a.csv"
    assert run["status"] == "FINISHED"
    assert run["duration_ms"] is None or run["duration_ms"] >= 0


def test_order_by_metric_and_pagination(client, seeded):
    ordered = client.get("/api/runs?dataset=data_user/a.csv&order_by=metrics.`val_loss/loss` ASC").get_json()["runs"]
    assert [r["metrics"]["val_loss/loss"] for r in ordered] == pytest.approx([0.3, 0.4, 0.5])

    first = client.get("/api/runs?max_results=4&order_by=metrics.`val_loss/loss` ASC").get_json()
    assert len(first["runs"]) == 4
    assert first["next_page_token"]
    second = client.get(
        f"/api/runs?max_results=4&order_by=metrics.`val_loss/loss` ASC&page_token={first['next_page_token']}"
    )
    rest = second.get_json()["runs"]
    assert len(rest) == 2
    assert not {r["run_id"] for r in first["runs"]} & {r["run_id"] for r in rest}
    assert client.get("/api/runs?order_by=1; DROP").status_code == 400
    assert client.get("/api/runs?page_token=bad token!").status_code == 400


def test_filter_validation(client, seeded):
    ok = client.get("/api/runs?filter=metrics.`val_loss/loss` < 0.45 AND params.lr = '0.03'")
    assert ok.status_code == 200
    assert [r["run_id"] for r in ok.get_json()["runs"]] == [seeded[("alpha", 2)], seeded[("beta", 2)]] or len(
        ok.get_json()["runs"]
    ) == 2
    bad = client.get("/api/runs?filter=1=1; DROP TABLE runs")
    assert bad.status_code == 400
    assert "invalid filter" in bad.get_json()["error"]
    assert client.get("/api/runs?filter=" + "params.a = 1 AND " * 50 + "params.b = 2").status_code == 400
    assert client.get("/api/runs?dataset=a'b").status_code == 400


def test_get_run_detail(client, seeded):
    run_id = seeded[("alpha", 0)]
    detail = client.get(f"/api/runs/{run_id}").get_json()
    assert detail["metric_keys"] == ["train_loss/loss", "val_loss/loss"]
    assert detail["scored_epochs"] == [1, 2]
    assert detail["tags"][contract.TAG_PER_SAMPLE_SCHEMA] == contract.SCHEMA_VERSION
    assert [a["path"] for a in detail["artifacts"]] == ["magnidata"]
    assert client.get("/api/runs/not-an-id").status_code == 400
    assert client.get(f"/api/runs/{'0' * 32}").status_code == 404


def test_metric_history(client, seeded):
    run_id = seeded[("alpha", 0)]
    body = client.get(f"/api/runs/{run_id}/metrics?keys=val_loss/loss").get_json()
    assert [(p["step"], p["value"]) for p in body["metrics"]["val_loss/loss"]] == [(0, 1.0), (1, 0.5)]
    everything = client.get(f"/api/runs/{run_id}/metrics").get_json()
    assert set(everything["metrics"]) == {"train_loss/loss", "val_loss/loss"}
    assert everything["truncated"] is False
    too_many = ",".join(f"k{i}" for i in range(21))
    assert client.get(f"/api/runs/{run_id}/metrics?keys={too_many}").status_code == 400


def test_downsample_keeps_spike():
    points = [{"step": i, "value": 1.0, "timestamp": i} for i in range(10_000)]
    points[4321]["value"] = 99.0
    points[7777]["value"] = -50.0
    reduced = experiments.downsample(points)
    assert len(reduced) <= experiments.MAX_POINTS
    values = [p["value"] for p in reduced]
    assert 99.0 in values
    assert -50.0 in values
    assert reduced[0]["step"] == 0
    assert reduced[-1]["step"] == 9999
    assert [p["step"] for p in reduced] == sorted(p["step"] for p in reduced)
    short = points[:10]
    assert experiments.downsample(short) == short


def test_long_history_is_downsampled_via_api(client, mlflow_store):
    run_id = seed_run(mlflow_store, metrics={"x": [float(i % 7) for i in range(5000)]})
    points = client.get(f"/api/runs/{run_id}/metrics?keys=x").get_json()["metrics"]["x"]
    assert len(points) <= experiments.MAX_POINTS
    assert max(p["value"] for p in points) == 6.0


def test_multi_run_metrics(client, seeded):
    ids = [seeded[("alpha", 0)], seeded[("beta", 1)]]
    body = client.post("/api/runs/metrics", json={"run_ids": ids, "keys": ["val_loss/loss"]}).get_json()
    assert set(body["runs"]) == set(ids)
    assert len(body["runs"][ids[0]]["val_loss/loss"]) == 2

    many = [f"{i:032x}" for i in range(11)]
    assert client.post("/api/runs/metrics", json={"run_ids": many, "keys": ["x"]}).status_code == 400
    assert client.post("/api/runs/metrics", json={"run_ids": [], "keys": ["x"]}).status_code == 400
    assert client.post("/api/runs/metrics", json={"run_ids": ids}).status_code == 400


def test_artifacts_list_and_download(client, seeded):
    run_id = seeded[("alpha", 0)]
    listing = client.get(f"/api/runs/{run_id}/artifacts?path=magnidata").get_json()["artifacts"]
    assert {a["path"] for a in listing} == {"magnidata/note.txt", "magnidata/pred.png"}

    text = client.get(f"/api/runs/{run_id}/artifact?path=magnidata/note.txt")
    assert text.status_code == 200
    assert text.data == b"hello"
    assert text.mimetype == "application/octet-stream"
    assert "attachment" in text.headers["Content-Disposition"]

    image = client.get(f"/api/runs/{run_id}/artifact?path=magnidata/pred.png")
    assert image.mimetype == "image/png"
    assert "Content-Disposition" not in image.headers


def test_artifact_path_validation_and_missing(client, seeded):
    run_id = seeded[("alpha", 0)]
    for bad in ("../x", "magnidata/../../x", "/etc/passwd", "a\\b"):
        assert client.get(f"/api/runs/{run_id}/artifact", query_string={"path": bad}).status_code == 400
        assert client.get(f"/api/runs/{run_id}/artifacts", query_string={"path": bad}).status_code == 400
    assert client.get(f"/api/runs/{run_id}/artifact").status_code == 400
    assert client.get(f"/api/runs/{run_id}/artifact?path=magnidata/nope.txt").status_code == 404
    assert client.get(f"/api/runs/{run_id}/artifact?path=magnidata").status_code == 404
    assert client.get(f"/api/runs/{'0' * 32}/artifacts").status_code == 404


def test_oversized_artifact_is_refused(client, seeded, monkeypatch):
    monkeypatch.setattr(experiments, "MAX_ARTIFACT_BYTES", 2)
    run_id = seeded[("alpha", 0)]
    assert client.get(f"/api/runs/{run_id}/artifact?path=magnidata/note.txt").status_code == 413


def test_link_sets_tags_and_run_appears_under_dataset(client, mlflow_store):
    run_id = seed_run(mlflow_store, "external")
    assert client.get("/api/runs?dataset=data_user/c.csv").get_json()["runs"] == []
    resp = client.post(f"/api/runs/{run_id}/link", json={"dataset_source": "data_user/c.csv"})
    assert resp.status_code == 200
    assert resp.get_json()["magnidata"][contract.TAG_DATASET_NAME] == "c"
    found = client.get("/api/runs?dataset=data_user/c.csv").get_json()["runs"]
    assert [r["run_id"] for r in found] == [run_id]


@pytest.mark.parametrize("source", ["", "/abs/x.csv", "../x.csv", "a/../../x.csv", "http://host/x.csv"])
def test_link_rejects_bad_sources(client, mlflow_store, source):
    run_id = seed_run(mlflow_store)
    assert client.post(f"/api/runs/{run_id}/link", json={"dataset_source": source}).status_code == 400


def test_link_unknown_run(client):
    assert client.post(f"/api/runs/{'0' * 32}/link", json={"dataset_source": "x.csv"}).status_code == 404


def test_runs_search_is_cached(client, mlflow_store, monkeypatch):
    monkeypatch.setattr(experiments, "RUNS_CACHE_TTL_S", 60)
    seed_run(mlflow_store, "cached")
    calls = []
    original = MlflowStore.search_runs_page

    def counting(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MlflowStore, "search_runs_page", counting)
    assert client.get("/api/runs").status_code == 200
    assert client.get("/api/runs").status_code == 200
    assert len(calls) == 1
    assert client.get("/api/runs?max_results=10").status_code == 200
    assert len(calls) == 2


def test_mlflow_down_returns_503(client):
    store_module.set_store(MlflowStore("http://127.0.0.1:1"))
    for url in ("/api/experiments", "/api/runs", f"/api/runs/{'0' * 32}"):
        resp = client.get(url)
        assert resp.status_code == 503
        assert resp.get_json()["mlflow_unavailable"] is True


@pytest.fixture
def scored_run(mlflow_store, client, tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "root"))
    (tmp_path / "root" / "data_user").mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for epoch, losses in ((1, (0.9, 0.2)), (2, (0.7, 0.3)), (3, (0.5, 0.4))):
        rows = "".join(f"{n},{s},{v}\n" for n, s, v in zip(("a.png", "b.png"), ("train", "val"), losses, strict=True))
        artifacts[contract.per_sample_path(epoch)] = "image_name,split,loss_total\n" + rows
    artifacts[contract.SPLIT_ARTIFACT] = "image_name,split\na.png,train\nb.png,val\n"
    tags = {contract.TAG_PER_SAMPLE_SCHEMA: contract.SCHEMA_VERSION, contract.TAG_SCORED_EPOCHS: "1,2,3"}
    return seed_run(mlflow_store, tags=tags, artifacts=artifacts)


def test_columns_returns_derived_csv(client, scored_run):
    resp = client.get(f"/api/runs/{scored_run}/columns")
    assert resp.status_code == 200
    assert resp.mimetype == "text/csv"
    assert resp.headers["Cache-Control"] == "no-store"
    lines = resp.get_data(as_text=True).splitlines()
    header = lines[0].split(",")
    assert header[0] == "image_name"
    assert {"loss_total_rank", "hardness", "loss_total_mean", "loss_total_slope"} <= set(header)
    row = dict(zip(header, lines[1].split(","), strict=True))
    assert row["image_name"] == "a.png"
    assert float(row["loss_total"]) == pytest.approx(0.5)
    assert float(row["loss_total_delta"]) == pytest.approx(0.4)

    early = client.get(f"/api/runs/{scored_run}/columns?epoch=1").get_data(as_text=True).splitlines()
    assert float(dict(zip(header, early[1].split(","), strict=True))["loss_total"]) == pytest.approx(0.9)


@pytest.fixture
def split_only_run(mlflow_store, client, tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "root"))
    (tmp_path / "root" / "data_user").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(experiments, "RUNS_CACHE_TTL_S", 0)
    return seed_run(
        mlflow_store,
        tags={contract.TAG_DATASET_SOURCE: "data_user/a.csv", contract.TAG_HAS_SPLIT: "true"},
        artifacts={contract.SPLIT_ARTIFACT: "image_name,split\na.png,train\nb.png,val\n"},
        status="RUNNING",
    )


def test_columns_before_first_scored_epoch_returns_split_only(client, split_only_run):
    resp = client.get(f"/api/runs/{split_only_run}/columns")
    assert resp.status_code == 200
    assert resp.mimetype == "text/csv"
    assert resp.headers["X-Magnidata-Scored-Epochs"] == "0"
    assert resp.get_data(as_text=True).splitlines() == ["image_name,split", "a.png,train", "b.png,val"]


def test_columns_with_neither_split_nor_scores_conflicts(client, mlflow_store):
    plain = seed_run(mlflow_store, tags={contract.TAG_HAS_SPLIT: "true"})
    assert client.get(f"/api/runs/{plain}/columns").status_code == 409


def test_run_summary_has_split_and_per_sample_filter(client, mlflow_store, split_only_run, scored_run):
    plain = seed_run(mlflow_store)
    runs = {r["run_id"]: r for r in client.get("/api/runs").get_json()["runs"]}
    assert runs[split_only_run]["has_split"] is True
    assert runs[split_only_run]["has_per_sample"] is False
    assert runs[scored_run]["has_split"] is True
    assert runs[scored_run]["has_per_sample"] is True
    assert runs[plain]["has_split"] is False

    def ids(mode):
        return {r["run_id"] for r in client.get(f"/api/runs?per_sample={mode}").get_json()["runs"]}

    assert {split_only_run, scored_run, plain} <= ids("any")
    assert split_only_run in ids("split")
    assert plain not in ids("split")
    assert ids("scored") == {scored_run}
    assert client.get("/api/runs?per_sample=bogus").status_code == 400


def test_columns_validation(client, scored_run, mlflow_store):
    assert client.get(f"/api/runs/{scored_run}/columns?epoch=9").status_code == 400
    assert client.get(f"/api/runs/{scored_run}/columns?epoch=abc").status_code == 400
    plain = seed_run(mlflow_store)
    resp = client.get(f"/api/runs/{plain}/columns")
    assert resp.status_code == 409
    assert "per-sample" in resp.get_json()["error"]
    assert client.get(f"/api/runs/{'0' * 32}/columns").status_code == 404


def test_columns_are_cached_per_latest_epoch(client, scored_run, mlflow_store):
    experiments.AXES_CACHE.clear()
    client.get(f"/api/runs/{scored_run}/columns")
    assert (scored_run, 3, 3) in experiments.AXES_CACHE
    mlflow_store._client.log_text(
        scored_run, "image_name,split,loss_total\na.png,train,0.1\nb.png,val,0.4\n", contract.per_sample_path(4)
    )
    mlflow_store.set_tag(scored_run, contract.TAG_SCORED_EPOCHS, "1,2,3,4")
    latest = client.get(f"/api/runs/{scored_run}/columns").get_data(as_text=True).splitlines()
    assert latest[1].split(",")[2] == "0.1"
    assert (scored_run, 4, 4) in experiments.AXES_CACHE


def test_match_counts_overlap(client, scored_run, tmp_path):
    (tmp_path / "root" / "data_user" / "ds.csv").write_text(
        "image_path,width\ndata_user/ds/images/a.png,1\ndata_user/ds/images/z.png,1\ndata_user/ds/images/y.png,1\n"
    )
    body = client.get(f"/api/runs/{scored_run}/match?dataset=data_user/ds.csv").get_json()
    assert (body["matched"], body["run_only"], body["dataset_only"]) == (1, 1, 2)
    assert body["run_only_examples"] == ["b.png"]
    assert body["dataset_only_examples"] == ["y.png", "z.png"]


def test_match_errors(client, scored_run, mlflow_store):
    assert client.get(f"/api/runs/{scored_run}/match").status_code == 400
    assert client.get(f"/api/runs/{scored_run}/match?dataset=data_user/missing.csv").status_code == 404
    assert client.get(f"/api/runs/{scored_run}/match?dataset=../x.csv").status_code == 404
    plain = seed_run(mlflow_store)
    assert client.get(f"/api/runs/{plain}/match?dataset=data_user/ds.csv").status_code == 409


def _png_bytes(color, size=(8, 6)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def predicted_run(mlflow_store, client, tmp_path, monkeypatch):
    """A run with 3 scored epochs, a linked dataset on disk, and predictions for a.png at epochs 1 and 3."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "root"))
    folder = tmp_path / "root" / "data_user" / "ds" / "images"
    folder.mkdir(parents=True)
    (folder / "a.png").write_bytes(_png_bytes((10, 120, 10), (40, 30)))
    (folder / "b.png").write_bytes(_png_bytes((120, 10, 10), (40, 30)))
    (tmp_path / "root" / "data_user" / "ds.csv").write_text(
        "image_path\ndata_user/ds/images/a.png\ndata_user/ds/images/b.png\n"
    )
    client_ = mlflow_store._client
    texts = {}
    for epoch, (a, b) in {1: (0.9, 0.2), 2: (0.7, 0.3), 3: (0.5, 0.4)}.items():
        texts[contract.per_sample_path(epoch)] = (
            f"image_name,split,loss_total,loss_dice,iou\na.png,train,{a},{a / 2},0.{epoch}\nb.png,val,{b},{b / 2},0.9\n"
        )
    tags = {
        contract.TAG_PER_SAMPLE_SCHEMA: contract.SCHEMA_VERSION,
        contract.TAG_SCORED_EPOCHS: "1,2,3",
        contract.TAG_DATASET_SOURCE: "data_user/ds.csv",
    }
    run_id = seed_run(mlflow_store, tags=tags, artifacts=texts)
    for epoch in (1, 3):
        local = tmp_path / f"pred{epoch}" / "a.png"
        local.parent.mkdir()
        local.write_bytes(_png_bytes((255, 0, 0), (8, 6)))
        client_.log_artifact(run_id, str(local), f"magnidata/predictions/epoch_{epoch:04d}")
    return run_id


def test_sample_history_is_aligned_with_epochs(client, predicted_run):
    body = client.get(f"/api/runs/{predicted_run}/samples/a.png").get_json()
    assert body["split"] == "train"
    assert body["epochs"] == [1, 2, 3]
    assert body["terms"]["total"] == pytest.approx([0.9, 0.7, 0.5])
    assert body["terms"]["dice"] == pytest.approx([0.45, 0.35, 0.25])
    assert body["iou"] == pytest.approx([0.1, 0.2, 0.3])
    assert body["prediction_epochs"] == [1, 3]
    assert client.get(f"/api/runs/{predicted_run}/samples/b.png").get_json()["prediction_epochs"] == []


def test_sample_history_errors(client, predicted_run, mlflow_store):
    assert client.get(f"/api/runs/{predicted_run}/samples/nope.png").status_code == 404
    assert client.get(f"/api/runs/{predicted_run}/samples/..%2Fx.png").status_code in (400, 404)
    assert client.get(f"/api/runs/{predicted_run}/samples/a..png").status_code == 400
    plain = seed_run(mlflow_store)
    assert client.get(f"/api/runs/{plain}/samples/a.png").status_code == 409


def test_prediction_returns_composited_jpeg(client, predicted_run):
    resp = client.get(f"/api/runs/{predicted_run}/prediction?image=a.png&epoch=3&alpha=1")
    assert resp.status_code == 200
    assert resp.mimetype == "image/jpeg"
    assert resp.headers["X-Magnidata-Epoch"] == "3"
    pixel = [int(v) for v in np.asarray(Image.open(io.BytesIO(resp.data)).convert("RGB"))[5, 5]]
    assert pixel[0] > 200
    assert pixel[1] < 60  # the red prediction covers the green image at full opacity
    assert client.get(f"/api/runs/{predicted_run}/prediction?image=a.png&epoch=latest").status_code == 200


def test_prediction_alpha_blends(client, predicted_run):
    resp = client.get(f"/api/runs/{predicted_run}/prediction?image=a.png&epoch=1&alpha=0.5")
    pixel = [int(v) for v in np.asarray(Image.open(io.BytesIO(resp.data)).convert("RGB"))[5, 5]]
    assert 30 < pixel[1] < 100  # between the green image (120) and the red mask (0)
    assert 80 < pixel[0] < 200


def test_prediction_missing_and_invalid(client, predicted_run):
    assert client.get(f"/api/runs/{predicted_run}/prediction?image=a.png&epoch=2").status_code == 404
    assert client.get(f"/api/runs/{predicted_run}/prediction?image=b.png&epoch=3").status_code == 404
    assert client.get(f"/api/runs/{predicted_run}/prediction?image=../a.png&epoch=3").status_code == 400
    assert client.get(f"/api/runs/{predicted_run}/prediction?image=a.png&epoch=x").status_code == 400
    assert client.get(f"/api/runs/{predicted_run}/prediction?image=a.png&alpha=big").status_code == 400
    assert client.get(f"/api/runs/{predicted_run}/prediction").status_code == 400


def test_inference_request_requires_checkpoint_and_known_image(client, predicted_run):
    assert client.post(f"/api/runs/{predicted_run}/inference", json={"image": "b.png"}).status_code == 409
    assert client.post(f"/api/runs/{predicted_run}/inference", json={"image": "missing.png"}).status_code == 404
    assert client.post(f"/api/runs/{predicted_run}/inference", json={"image": "../b.png"}).status_code == 400


def test_inference_request_is_queued_and_saved_result_becomes_ready(client, predicted_run, mlflow_store, tmp_path):
    mlflow_client = mlflow_store._client
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"checkpoint placeholder")
    mlflow_client.log_artifact(predicted_run, str(checkpoint), "checkpoints")

    queued = client.post(f"/api/runs/{predicted_run}/inference", json={"image": "b.png"})
    assert queued.status_code == 202
    assert queued.get_json() == {"status": "queued", "image": "b.png", "epoch": 3}
    assert client.get(f"/api/runs/{predicted_run}/inference?image=b.png").get_json()["status"] == "queued"

    prediction = tmp_path / "b.png"
    prediction.write_bytes(_png_bytes((0, 0, 255)))
    mlflow_client.log_artifact(predicted_run, str(prediction), "magnidata/predictions/epoch_0003")
    mlflow_client.set_tag(predicted_run, contract.TAG_INFERENCE_STATUS, "ready")

    ready = client.get(f"/api/runs/{predicted_run}/inference?image=b.png").get_json()
    assert ready == {"status": "ready", "image": "b.png", "epoch": 3}
    assert client.get(f"/api/runs/{predicted_run}/samples/b.png").get_json()["prediction_epochs"] == [3]


def _scored(mlflow_store, rows_by_epoch):
    """Seed a run whose per-sample files hold ``{epoch: [(image, split, loss_total), ...]}``."""
    artifacts = {
        contract.per_sample_path(epoch): "image_name,split,loss_total\n" + "".join(f"{n},{s},{v}\n" for n, s, v in rows)
        for epoch, rows in rows_by_epoch.items()
    }
    tags = {
        contract.TAG_PER_SAMPLE_SCHEMA: contract.SCHEMA_VERSION,
        contract.TAG_SCORED_EPOCHS: ",".join(map(str, rows_by_epoch)),
    }
    return seed_run(mlflow_store, tags=tags, artifacts=artifacts)


def _parse(resp):
    return {r["image_name"]: r for r in csv.DictReader(io.StringIO(resp.get_data(as_text=True)))}


def test_compare_samples_outer_join_and_changes(client, mlflow_store):
    run_a = _scored(mlflow_store, {1: [("x.png", "train", 0.9), ("y.png", "train", 0.5), ("z.png", "val", 0.2)]})
    run_b = _scored(mlflow_store, {1: [("x.png", "train", 0.3), ("y.png", "train", 0.8), ("w.png", "val", 0.4)]})
    resp = client.get(f"/api/runs/compare-samples?a={run_a}&b={run_b}")
    assert resp.status_code == 200
    assert resp.mimetype == "text/csv"
    header = resp.get_data(as_text=True).splitlines()[0].split(",")
    assert header == ["image_name", "hardness_a", "hardness_b", "hardness_change", "loss_total_change", "in_a", "in_b"]
    rows = _parse(resp)
    assert set(rows) == {"x.png", "y.png", "z.png", "w.png"}
    assert float(rows["x.png"]["loss_total_change"]) == pytest.approx(-0.6)
    assert float(rows["y.png"]["loss_total_change"]) == pytest.approx(0.3)
    assert (rows["x.png"]["in_a"], rows["x.png"]["in_b"]) == ("1", "1")
    assert (rows["z.png"]["in_a"], rows["z.png"]["in_b"]) == ("1", "0")
    assert (rows["w.png"]["in_a"], rows["w.png"]["in_b"]) == ("0", "1")
    assert rows["z.png"]["hardness_change"] == ""
    assert rows["w.png"]["loss_total_change"] == ""


def test_compare_samples_hardness_change_sign(client, mlflow_store):
    easy = [("a.png", "train", 0.1), ("b.png", "train", 0.9), ("c.png", "train", 0.5)]
    swapped = [("a.png", "train", 0.9), ("b.png", "train", 0.1), ("c.png", "train", 0.5)]
    rows = _parse(
        client.get(
            f"/api/runs/compare-samples?a={_scored(mlflow_store, {1: easy})}&b={_scored(mlflow_store, {1: swapped})}"
        )
    )
    assert float(rows["a.png"]["hardness_change"]) > 0  # got harder
    assert float(rows["b.png"]["hardness_change"]) < 0  # got easier


def test_compare_samples_same_run_has_no_change(client, mlflow_store):
    run = _scored(
        mlflow_store,
        {1: [("a.png", "train", 0.4), ("b.png", "val", 0.6)], 2: [("a.png", "train", 0.3), ("b.png", "val", 0.5)]},
    )
    rows = _parse(client.get(f"/api/runs/compare-samples?a={run}&b={run}"))
    assert all(float(r["loss_total_change"]) == 0 for r in rows.values())
    assert all(float(r["hardness_change"]) == 0 for r in rows.values())


def test_compare_samples_epoch_and_errors(client, mlflow_store):
    a = _scored(mlflow_store, {1: [("a.png", "train", 0.4)], 2: [("a.png", "train", 0.2)]})
    b = _scored(mlflow_store, {1: [("a.png", "train", 0.9)], 2: [("a.png", "train", 0.1)]})
    assert float(
        _parse(client.get(f"/api/runs/compare-samples?a={a}&b={b}&epoch=1"))["a.png"]["loss_total_change"]
    ) == pytest.approx(0.5)
    assert client.get(f"/api/runs/compare-samples?a={a}&b={b}&epoch=9").status_code == 400
    assert client.get(f"/api/runs/compare-samples?a={a}&b={b}&epoch=abc").status_code == 400
    disjoint = _scored(mlflow_store, {1: [("zzz.png", "train", 0.1)]})
    no_overlap = client.get(f"/api/runs/compare-samples?a={a}&b={disjoint}")
    assert no_overlap.status_code == 400
    assert "in common" in no_overlap.get_json()["error"]
    plain = seed_run(mlflow_store)
    assert client.get(f"/api/runs/compare-samples?a={a}&b={plain}").status_code == 409
    assert client.get(f"/api/runs/compare-samples?a=nope&b={b}").status_code == 400
    assert client.get(f"/api/runs/compare-samples?a={a}").status_code == 400


# ── Per-dataset experiment counts ──────────────────────────────────────────────


def _run(status, source=None):
    return {"status": status, "tags": {contract.TAG_DATASET_SOURCE: source} if source else {}}


def test_count_runs_by_dataset_separates_active_from_finished_and_skips_unlinked():
    counts = experiments.count_runs_by_dataset(
        [
            _run("RUNNING", "data_user/a.csv"),
            _run("SCHEDULED", "data_user/a.csv"),
            _run("FINISHED", "data_user/a.csv"),
            _run("FAILED", "data_user/b.csv"),
            _run("RUNNING"),
        ]
    )

    assert counts == {"data_user/a.csv": {"total": 3, "active": 2}, "data_user/b.csv": {"total": 1, "active": 0}}


def test_experiment_counts_endpoint_reads_mlflow_runs(client, seeded, monkeypatch):
    monkeypatch.setattr(experiments, "RUNS_CACHE_TTL_S", 0)

    counts = client.get("/api/datasets/experiment-counts").get_json()

    assert counts == {"data_user/a.csv": {"total": 3, "active": 0}, "data_user/b.csv": {"total": 3, "active": 0}}


def test_experiment_counts_is_empty_when_mlflow_is_down(client):
    store_module.set_store(MlflowStore("http://127.0.0.1:1"))

    resp = client.get("/api/datasets/experiment-counts")

    assert resp.status_code == 200
    assert resp.get_json() == {}


def test_on_demand_inference_is_refused_for_agritune_runs(client, predicted_run, mlflow_store):
    mlflow_store.set_tag(predicted_run, contract.TAG_ENGINE, "agritune")

    resp = client.post(f"/api/runs/{predicted_run}/inference", json={"image": "b.png"})

    assert resp.status_code == 409
    assert "AgriTune" in resp.get_json()["error"]
