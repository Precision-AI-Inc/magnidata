# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MLflow browsing endpoints, against a real sqlite-backed MLflow store.

Run:  python -m pytest precisionai/magnidata/api/test_experiments.py -q
"""

import experiments
import mlflow_store as store_module
import pytest
from conftest import seed_run
from flask import Flask
from mlflow_store import MlflowStore

from precisionai.magnidata.training import contract


@pytest.fixture
def client(mlflow_store):
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
    (tmp_path / "root" / "data_user").mkdir(parents=True)
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
