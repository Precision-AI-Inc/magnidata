# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for serving the built dashboard from the API process (MAGNIDATA_STATIC_DIR).

Run:  python -m pytest precisionai/magnidata/api/test_dashboard_static.py -q
"""

import pytest
from conftest import load_app


@pytest.fixture
def dashboard_client(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>MagniData</title>")
    (dist / "assets" / "app-abc123.js").write_text("console.log('hi')")
    (dist / "logo.png").write_bytes(b"\x89PNG")
    app_module = load_app(tmp_path / "data", monkeypatch, MAGNIDATA_STATIC_DIR=str(dist))
    with app_module.app.test_client() as client:
        yield client


def test_root_serves_index_html_that_is_always_revalidated(dashboard_client):
    response = dashboard_client.get("/")

    assert response.status_code == 200
    assert b"MagniData" in response.data
    assert response.headers["Cache-Control"] == "no-cache"


def test_client_side_route_falls_back_to_index_html(dashboard_client):
    response = dashboard_client.get("/experiments/run-1")

    assert response.status_code == 200
    assert b"MagniData" in response.data
    assert response.headers["Cache-Control"] == "no-cache"


def test_hashed_asset_is_cached_forever(dashboard_client):
    response = dashboard_client.get("/assets/app-abc123.js")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "public, max-age=31536000, immutable"


def test_missing_asset_is_a_404_not_index_html(dashboard_client):
    response = dashboard_client.get("/assets/app-stale.js")

    assert response.status_code == 404


def test_other_static_file_is_served_as_is(dashboard_client):
    response = dashboard_client.get("/logo.png")

    assert response.status_code == 200
    assert response.data == b"\x89PNG"


def test_unknown_api_path_is_a_json_404(dashboard_client):
    response = dashboard_client.get("/api/no-such-endpoint")

    assert response.status_code == 404
    assert response.get_json() == {"error": "not found"}


def test_real_api_routes_still_win(dashboard_client):
    response = dashboard_client.get("/api/health")

    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_path_traversal_does_not_leave_the_static_folder(dashboard_client):
    response = dashboard_client.get("/..%2Fdata%2Fconfi.yaml")

    assert b"datasets" not in response.data


def test_no_dashboard_routes_without_the_env_var(tmp_path, monkeypatch):
    app_module = load_app(tmp_path, monkeypatch, MAGNIDATA_STATIC_DIR=None)

    with app_module.app.test_client() as client:
        assert client.get("/").status_code == 404
        assert client.get("/api/health").status_code == 200
