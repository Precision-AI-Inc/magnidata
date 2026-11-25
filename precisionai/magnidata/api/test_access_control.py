# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for who may reach the API and which files a request may name.

Run:  python -m pytest precisionai/magnidata/api/test_access_control.py -q
"""

import pytest
from conftest import load_app
from PIL import Image

from precisionai.magnidata.tools import features

EVIL = "https://evil.example"
PORTAL = "http://localhost:5175"


def _write_png(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), (10, 120, 10)).save(path)


def _preflight(client, origin, url="/api/datasets/7", method="DELETE"):
    return client.options(url, headers={"Origin": origin, "Access-Control-Request-Method": method})


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    """A normal deployment whose DATA_ROOT holds a dataset, the notes database and confi.yaml."""
    root = tmp_path / "root"
    app_module = load_app(
        root,
        monkeypatch,
        MAGNIDATA_DEMO_MODE=None,
        MAGNIDATA_PORTAL_URL=None,
        MAGNIDATA_CORS_ORIGINS=None,
        MAGNIDATA_ALLOWED_HOSTS=None,
    )
    (root / "data_user" / "ds.csv").write_text("image_path\ndata_user/ds/images/a.png\n")
    _write_png(root / "data_user" / "ds" / "images" / "a.png")
    _write_png(root / "data_user" / "ds" / "masks" / "a.png")
    _write_png(root / "stray.png")
    _write_png(tmp_path / "outside.png")
    with app_module.app.test_client() as client:
        yield client, app_module, tmp_path


# ── CORS ───────────────────────────────────────────────────────────────────────


def test_cors_refuses_a_preflight_from_another_site(deployment):
    client, _, _ = deployment

    response = _preflight(client, EVIL)

    assert "Access-Control-Allow-Origin" not in response.headers


def test_cors_allows_the_portal_origin_by_default(deployment):
    client, _, _ = deployment

    response = _preflight(client, PORTAL)

    assert response.headers.get("Access-Control-Allow-Origin") == PORTAL


def test_cors_follows_the_configured_portal_url(tmp_path, monkeypatch):
    app_module = load_app(
        tmp_path, monkeypatch, MAGNIDATA_PORTAL_URL="http://localhost:6175/", MAGNIDATA_CORS_ORIGINS=None
    )
    with app_module.app.test_client() as client:
        assert _preflight(client, "http://localhost:6175").headers.get("Access-Control-Allow-Origin") == (
            "http://localhost:6175"
        )
        assert "Access-Control-Allow-Origin" not in _preflight(client, PORTAL).headers


def test_cors_origins_can_be_listed_explicitly(tmp_path, monkeypatch):
    app_module = load_app(
        tmp_path, monkeypatch, MAGNIDATA_CORS_ORIGINS="http://a.test, http://b.test", MAGNIDATA_PORTAL_URL=None
    )
    with app_module.app.test_client() as client:
        assert _preflight(client, "http://b.test").headers.get("Access-Control-Allow-Origin") == "http://b.test"
        assert "Access-Control-Allow-Origin" not in _preflight(client, PORTAL).headers
        assert "Access-Control-Allow-Origin" not in _preflight(client, EVIL).headers


def test_a_write_from_another_site_is_refused_even_without_a_preflight(deployment):
    client, _, _ = deployment

    response = client.post("/api/datasets/merge", data={"name": "x"}, headers={"Origin": EVIL})

    assert response.status_code == 403
    assert response.get_json() == {"error": "cross-site request refused"}


@pytest.mark.parametrize(
    "headers", [{}, {"Origin": PORTAL}, {"Origin": "http://localhost:5174"}, {"Origin": "http://localhost"}]
)
def test_a_write_from_the_portal_its_dev_server_its_own_host_or_no_browser_is_let_through(deployment, headers):
    client, _, _ = deployment

    response = client.post("/api/datasets/merge", json={}, headers=headers)

    assert response.status_code == 400  # past the guard, refused by the handler for the empty body


def test_a_read_from_another_site_is_left_to_cors(deployment):
    client, _, _ = deployment

    response = client.get("/api/health", headers={"Origin": EVIL})

    assert response.status_code == 200
    assert "Access-Control-Allow-Origin" not in response.headers


def test_a_demo_deployment_lets_any_page_read_it(tmp_path, monkeypatch):
    app_module = load_app(tmp_path, monkeypatch, MAGNIDATA_DEMO_MODE="1", MAGNIDATA_CORS_ORIGINS=None)
    with app_module.app.test_client() as client:
        response = client.get("/api/health", headers={"Origin": EVIL})

    assert response.headers.get("Access-Control-Allow-Origin") in {"*", EVIL}


# ── Host header ────────────────────────────────────────────────────────────────

REBOUND = "http://evil.example:5175"


def test_a_rebound_page_cannot_write_as_its_own_origin(deployment):
    client, _, _ = deployment

    response = client.delete("/api/datasets/7", base_url=REBOUND, headers={"Origin": REBOUND})

    assert response.status_code == 403
    assert response.get_json() == {"error": "unknown host refused"}


def test_a_read_addressed_to_another_host_is_refused(deployment):
    client, _, _ = deployment

    assert client.get("/api/health", base_url=REBOUND).status_code == 403


@pytest.mark.parametrize(
    "base_url", ["http://localhost:5051", "http://127.0.0.1:5051", "http://[::1]:5051", "http://api:5050", "http://api"]
)
def test_this_machine_and_the_compose_service_name_are_answered(deployment, base_url):
    client, _, _ = deployment

    assert client.get("/api/health", base_url=base_url).status_code == 200


def test_configured_hosts_are_answered(tmp_path, monkeypatch):
    app_module = load_app(
        tmp_path, monkeypatch, MAGNIDATA_DEMO_MODE=None, MAGNIDATA_ALLOWED_HOSTS=" lab-box.local:5175, 10.0.0.5"
    )
    with app_module.app.test_client() as client:
        assert client.get("/api/health", base_url="http://lab-box.local:5051").status_code == 200
        assert client.get("/api/health", base_url="http://10.0.0.5:5175").status_code == 200
        assert client.get("/api/health", base_url=REBOUND).status_code == 403


def test_a_demo_deployment_answers_its_public_host(tmp_path, monkeypatch):
    app_module = load_app(tmp_path, monkeypatch, MAGNIDATA_DEMO_MODE="1", MAGNIDATA_ALLOWED_HOSTS=None)
    with app_module.app.test_client() as client:
        assert client.get("/api/health", base_url="https://someone-magnidata.hf.space").status_code == 200


# ── Dataset paths ──────────────────────────────────────────────────────────────


def test_dataset_csv_serves_a_file_inside_the_data_folders(deployment):
    client, _, _ = deployment

    response = client.get("/api/dataset/csv", query_string={"source": "data_user/ds.csv"})

    assert response.status_code == 200


@pytest.mark.parametrize("source", ["test.db", "confi.yaml", "app.py", "data_user/../confi.yaml", "/etc/passwd"])
def test_dataset_csv_refuses_a_file_outside_the_data_folders(deployment, source):
    client, _, _ = deployment

    response = client.get("/api/dataset/csv", query_string={"source": source})

    assert response.status_code == 400
    assert response.get_json() == {"error": "invalid source path"}


@pytest.mark.parametrize(
    "route", ["/api/dataset/embeddings", "/api/dataset/embeddings/variants", "/api/datasets/summary"]
)
def test_other_dataset_reads_refuse_a_file_outside_the_data_folders(deployment, route):
    client, _, _ = deployment

    assert client.get(route, query_string={"source": "confi.yaml"}).status_code == 400


def test_creating_a_dataset_refuses_a_parent_outside_the_data_folders(deployment):
    client, _, _ = deployment

    response = client.post(
        "/api/datasets", json={"name": "leak", "parent_source": "confi.yaml", "image_names": ["datasets: []"]}
    )

    assert response.status_code == 400


def test_merging_refuses_a_source_outside_the_data_folders(deployment):
    client, _, _ = deployment

    response = client.post("/api/datasets/merge", json={"name": "leak", "sources": ["data_user/ds.csv", "confi.yaml"]})

    assert response.status_code == 400


# ── Image paths ────────────────────────────────────────────────────────────────

IMAGE_ROUTES = ["/api/image", "/api/mask", "/api/mask/debug", "/api/annotation/mask", "/api/annotation/overlay"]


@pytest.mark.parametrize("route", IMAGE_ROUTES)
def test_image_routes_serve_a_file_inside_the_data_folders(deployment, route):
    client, _, _ = deployment

    assert client.get(route, query_string={"path": "data_user/ds/images/a.png"}).status_code == 200


@pytest.mark.parametrize("route", IMAGE_ROUTES)
@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_image_routes_refuse_a_file_outside_the_data_folders(deployment, monkeypatch, route, method):
    client, app_module, tmp_path = deployment

    def must_not_read(_path):
        raise AssertionError("an outside file was read")

    monkeypatch.setattr(app_module.local_files, "read_file", must_not_read)
    for path in (str(tmp_path / "outside.png"), "stray.png", "test.db", "../outside.png", "/dev/zero"):
        response = client.open(route, method=method, query_string={"path": path})
        assert response.status_code == 400, path


def test_overlay_refuses_either_path_outside_the_data_folders(deployment):
    client, _, tmp_path = deployment
    inside = "data_user/ds/images/a.png"

    for params in ({"img": inside, "mask": "stray.png"}, {"img": str(tmp_path / "outside.png"), "mask": inside}):
        assert client.get("/api/overlay", query_string=params).status_code == 400
    assert client.get("/api/overlay", query_string={"img": inside, "mask": inside}).status_code == 200


def test_image_routes_refuse_a_symlink_that_leaves_the_data_folders(deployment):
    client, _, tmp_path = deployment
    link = tmp_path / "root" / "data_user" / "ds" / "images" / "link.png"
    link.symlink_to(tmp_path / "outside.png")

    assert client.get("/api/image", query_string={"path": "data_user/ds/images/link.png"}).status_code == 400


# ── Listening address and image decoding ───────────────────────────────────────


def test_the_api_listens_on_localhost_unless_told_otherwise(tmp_path, monkeypatch):
    app_module = load_app(tmp_path, monkeypatch, API_HOST=None, MAGNIDATA_PORTAL_URL=None, MAGNIDATA_API_URL=None)
    calls = []
    monkeypatch.setattr(app_module.app, "run", lambda **kwargs: calls.append(kwargs))

    app_module.main()
    monkeypatch.setenv("API_HOST", "0.0.0.0")
    app_module.main()

    assert [call["host"] for call in calls] == ["127.0.0.1", "0.0.0.0"]


def test_importing_the_feature_extractor_keeps_pillows_decompression_bomb_guard():
    assert features.Image is Image
    assert Image.MAX_IMAGE_PIXELS is not None
