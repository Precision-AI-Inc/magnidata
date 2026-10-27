# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the demo deployment switch (MAGNIDATA_DEMO_MODE).

Run:  python -m pytest precisionai/magnidata/api/test_demo_mode.py -q
"""

import pytest
from conftest import load_app
from PIL import Image

# Every (method, URL) a demo deployment refuses, with ids filled in. Spelled out here rather than
# imported from app.py, so dropping one from the server's list fails a test.
REFUSED = [
    ("POST", "/api/datasets/build"),
    ("POST", "/api/datasets"),
    ("POST", "/api/datasets/merge"),
    ("POST", "/api/datasets/7/parents"),
    ("DELETE", "/api/datasets/7"),
    ("POST", "/api/notes"),
    ("PUT", "/api/notes/7"),
    ("DELETE", "/api/notes/7"),
    ("POST", "/api/partitions/commit"),
    ("POST", "/api/training/runs"),
    ("POST", "/api/runs/abc/cancel"),
    ("POST", "/api/runs/abc/link"),
    ("POST", "/api/runs/abc/inference"),
    ("GET", "/api/mask/debug"),
]


@pytest.fixture
def default_client(tmp_path, monkeypatch):
    app_module = load_app(tmp_path, monkeypatch, MAGNIDATA_DEMO_MODE=None)
    with app_module.app.test_client() as client:
        yield client, app_module


@pytest.fixture
def demo_client(tmp_path, monkeypatch):
    app_module = load_app(tmp_path, monkeypatch, MAGNIDATA_DEMO_MODE="1")
    with app_module.app.test_client() as client:
        yield client, app_module


def test_config_reports_a_normal_deployment_by_default(default_client):
    client, _ = default_client

    assert client.get("/api/config").get_json() == {"demo_mode": False}


def test_config_reports_a_demo_deployment(demo_client):
    client, _ = demo_client

    assert client.get("/api/config").get_json() == {"demo_mode": True}


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_demo_mode_accepts_the_usual_truthy_spellings(tmp_path, monkeypatch, value):
    app_module = load_app(tmp_path, monkeypatch, MAGNIDATA_DEMO_MODE=value)

    assert app_module.DEMO_MODE is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
def test_demo_mode_stays_off_for_anything_else(tmp_path, monkeypatch, value):
    app_module = load_app(tmp_path, monkeypatch, MAGNIDATA_DEMO_MODE=value)

    assert app_module.DEMO_MODE is False


def test_demo_mode_refuses_a_dataset_build_before_it_starts(demo_client, monkeypatch):
    client, app_module = demo_client

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("a build was started")

    monkeypatch.setattr(app_module.dataset_builder, "start_demo_build", must_not_run)
    monkeypatch.setattr(app_module.dataset_builder, "start_upload_build", must_not_run)

    response = client.post("/api/datasets/build", data={"source_type": "demo", "demo_key": "agristress500"})

    assert response.status_code == 403
    assert response.get_json() == {"error": "Disabled for the Demo"}


def test_demo_mode_refuses_starting_a_training_run(demo_client):
    client, _ = demo_client

    response = client.post("/api/training/runs", json={"dataset": "data_user/a.csv"})

    assert response.status_code == 403
    assert response.get_json() == {"error": "Disabled for the Demo"}


def test_demo_mode_leaves_reading_and_the_rest_alone(demo_client):
    client, _ = demo_client

    assert client.get("/api/health").status_code == 200
    assert client.get("/api/datasets").status_code == 200
    assert client.get("/api/datasets/demos").status_code == 200
    assert client.get("/api/datasets/build/limits").status_code == 200


def test_a_normal_deployment_still_handles_build_requests(default_client):
    client, _ = default_client

    response = client.post("/api/datasets/build", data={"source_type": "nonsense"})

    assert response.status_code == 400
    assert "source_type" in response.get_json()["error"]


def test_a_normal_deployment_still_reaches_the_training_endpoint(default_client):
    client, _ = default_client

    response = client.post("/api/training/runs", json={})

    assert response.status_code != 403


# ── Everything else that changes shared state ────────────────────────────────────


@pytest.mark.parametrize(("method", "url"), REFUSED)
def test_demo_mode_refuses_every_route_that_changes_shared_state(demo_client, method, url):
    client, _ = demo_client

    response = client.open(url, method=method, json={})

    assert response.status_code == 403
    assert response.get_json() == {"error": "Disabled for the Demo"}


@pytest.mark.parametrize(("method", "url"), REFUSED)
def test_a_normal_deployment_still_reaches_each_of_those_routes(default_client, method, url):
    client, _ = default_client

    response = client.open(url, method=method, json={})

    assert response.status_code != 403


def test_demo_mode_keeps_a_visitor_from_adding_a_note(demo_client, monkeypatch):
    client, app_module = demo_client

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("a note was written")

    monkeypatch.setattr(app_module.db, "create_note", must_not_run)

    response = client.post("/api/notes", json={"stem": "a", "image_path": "data_user/a.png", "note": "hello"})

    assert response.status_code == 403


def test_a_normal_deployment_still_adds_a_note(default_client):
    client, _ = default_client

    response = client.post("/api/notes", json={"stem": "a", "image_path": "data_user/a.png", "note": "hello"})

    assert response.status_code == 201
    assert response.get_json()["note"] == "hello"


def test_demo_mode_keeps_a_visitor_from_deleting_a_dataset(demo_client):
    client, app_module = demo_client
    record = app_module.db.create_dataset_record("Mine", "", "data_user/mine.csv", None, None, 3)

    response = client.delete(f"/api/datasets/{record['id']}")

    assert response.status_code == 403
    assert app_module.db.get_created_dataset(record["id"]) is not None


def test_a_normal_deployment_still_deletes_a_dataset(default_client):
    client, app_module = default_client
    record = app_module.db.create_dataset_record("Mine", "", "data_user/mine.csv", None, None, 3)

    response = client.delete(f"/api/datasets/{record['id']}")

    assert response.status_code == 204
    assert app_module.db.get_created_dataset(record["id"]) is None


# ── There is no MLflow server ────────────────────────────────────────────────────


def test_demo_mode_answers_mlflow_questions_at_once_and_quietly(demo_client, caplog):
    client, _ = demo_client

    with caplog.at_level("DEBUG"):
        status = client.get("/api/mlflow/status")
        presets = client.get("/api/training/presets")
        experiments = client.get("/api/experiments")

    assert status.status_code == 200
    assert status.get_json()["reachable"] is False
    assert presets.status_code == 200
    assert presets.get_json()["mlflow"]["reachable"] is False
    assert experiments.status_code == 503
    assert experiments.get_json()["mlflow_unavailable"] is True
    assert not [record for record in caplog.records if "MLflow" in record.getMessage()]


def test_a_normal_deployment_still_uses_the_real_mlflow_store(default_client):
    _, app_module = default_client

    store = app_module.mlflow_store.get_store()

    assert type(store) is app_module.mlflow_store.MlflowStore


# ── Image paths stay inside the data folder ──────────────────────────────────────


def _write_png(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), (10, 120, 10)).save(path)


@pytest.fixture
def open_deployment(tmp_path, monkeypatch):
    """A demo deployment whose DATA_ROOT is ``tmp_path/root``, with an image beside it that is outside it."""
    root = tmp_path / "root"
    app_module = load_app(root, monkeypatch, MAGNIDATA_DEMO_MODE="1")
    monkeypatch.chdir(root)  # image paths in a table are relative to DATA_ROOT, as in the container
    _write_png(root / "data_user" / "ds" / "images" / "a.png")
    _write_png(root / "data_user" / "ds" / "masks" / "a.png")  # the mask the annotation endpoints pair by name
    _write_png(tmp_path / "outside.png")
    with app_module.app.test_client() as client:
        yield client, tmp_path


def test_demo_mode_serves_an_image_inside_the_data_folder(open_deployment):
    client, _ = open_deployment

    for route in ("/api/image", "/api/mask", "/api/annotation/mask", "/api/annotation/overlay"):
        assert client.get(route, query_string={"path": "data_user/ds/images/a.png"}).status_code == 200, route
    overlay = client.get(
        "/api/overlay", query_string={"img": "data_user/ds/images/a.png", "mask": "data_user/ds/masks/a.png"}
    )
    assert overlay.status_code == 200


def test_demo_mode_refuses_an_image_path_outside_the_data_folder(open_deployment):
    client, tmp_path = open_deployment
    outside = [
        str(tmp_path / "outside.png"),
        "../outside.png",
        "data_user/../../outside.png",
        "https://d379glmvb7evte.cloudfront.net/x.png",
    ]

    for route in ("/api/image", "/api/mask", "/api/annotation/mask", "/api/annotation/overlay"):
        for path in outside:
            response = client.get(route, query_string={"path": path})
            assert response.status_code == 400, (route, path)
            assert response.get_json() == {"error": "invalid path"}
    for parameter in ("img", "mask"):
        params = {
            "img": "data_user/ds/images/a.png",
            "mask": "data_user/ds/masks/a.png",
            parameter: str(tmp_path / "outside.png"),
        }
        assert client.get("/api/overlay", query_string=params).status_code == 400


def test_demo_mode_leaves_a_missing_path_parameter_to_the_handler(open_deployment):
    client, _ = open_deployment

    response = client.get("/api/image")

    assert response.status_code == 400
    assert response.get_json() == {"error": "path required"}


def test_a_normal_deployment_still_reads_an_image_by_absolute_path(tmp_path, monkeypatch):
    root = tmp_path / "root"
    app_module = load_app(root, monkeypatch, MAGNIDATA_DEMO_MODE=None)
    _write_png(tmp_path / "outside.png")

    with app_module.app.test_client() as client:
        response = client.get("/api/image", query_string={"path": str(tmp_path / "outside.png")})

    assert response.status_code == 200


# ── The 3D view's analysis jobs ──────────────────────────────────────────────────


def _seed_analysis_jobs(app_module, statuses):
    app_module._ANALYSIS_JOBS.clear()
    for number, status in enumerate(statuses):
        app_module._ANALYSIS_JOBS[f"job{number}"] = {"job_id": f"job{number}", "status": status}


def _start_analysis(client):
    return client.post("/api/embedding/analysis", json={"source": "data_user/a.csv", "k": 3, "reduction": "pca"})


def test_demo_mode_limits_how_many_analyses_run_at_once(demo_client, monkeypatch):
    client, app_module = demo_client
    monkeypatch.setattr(app_module, "_run_analysis_job", lambda *_args, **_kwargs: None)
    _seed_analysis_jobs(app_module, ["running"] * app_module.DEMO_MAX_RUNNING_ANALYSES)

    response = _start_analysis(client)

    assert response.status_code == 429
    assert len(app_module._ANALYSIS_JOBS) == app_module.DEMO_MAX_RUNNING_ANALYSES


def test_demo_mode_keeps_only_the_latest_finished_analyses(demo_client, monkeypatch):
    client, app_module = demo_client
    monkeypatch.setattr(app_module, "_run_analysis_job", lambda *_args, **_kwargs: None)
    kept = app_module.DEMO_MAX_KEPT_ANALYSES
    _seed_analysis_jobs(app_module, ["done"] * (kept + 10))

    response = _start_analysis(client)

    assert response.status_code == 202
    assert len(app_module._ANALYSIS_JOBS) == kept
    assert "job0" not in app_module._ANALYSIS_JOBS
    assert f"job{kept + 9}" in app_module._ANALYSIS_JOBS
    assert response.get_json()["job_id"] in app_module._ANALYSIS_JOBS


def test_a_normal_deployment_never_limits_analyses(default_client, monkeypatch):
    client, app_module = default_client
    monkeypatch.setattr(app_module, "_run_analysis_job", lambda *_args, **_kwargs: None)
    _seed_analysis_jobs(app_module, ["running"] * 20 + ["done"] * 80)

    response = _start_analysis(client)

    assert response.status_code == 202
    assert len(app_module._ANALYSIS_JOBS) == 101
