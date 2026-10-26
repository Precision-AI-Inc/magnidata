# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""HTTP-layer tests for GET /api/datasets/demos, the dataset-build endpoints, and the
extended remove_dataset cleanup. dataset_builder's own orchestration logic is
unit-tested in test_dataset_builder.py — these tests only check request parsing,
status codes, and job_id plumbing, so start_upload_build/start_demo_build are
monkeypatched.

Run:  python -m pytest precisionai/magnidata/api/test_app.py -q
"""

import csv
import importlib
import io
import json
import os
import threading
from pathlib import Path

import mlflow_store
import numpy as np
import pytest
from conftest import seed_run
from PIL import Image

from precisionai.magnidata.training import contract


def _write_feature_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "width", "score"])
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("CONFIG_PATH", str(tmp_path / "confi.yaml"))
    (tmp_path / "confi.yaml").write_text("datasets: []\n")
    os.makedirs(tmp_path / "data_user", exist_ok=True)

    app_module = importlib.import_module("app")
    monkeypatch.setattr(app_module.db, "DB_PATH", str(tmp_path / "test.db"))
    importlib.reload(app_module)
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as client:
        yield client, app_module


# ── Startup banner ───────────────────────────────────────────────────────────────


def test_access_banner_lists_host_urls(app_client, monkeypatch):
    _, app_module = app_client
    monkeypatch.setenv("MAGNIDATA_PORTAL_URL", "http://localhost:5175")
    monkeypatch.setenv("MAGNIDATA_API_URL", "http://localhost:5051/")

    banner = app_module._access_banner()

    assert banner is not None
    assert "portal -> http://localhost:5175" in banner
    assert "api    -> http://localhost:5051/api/health" in banner


def test_access_banner_is_silent_without_host_urls(app_client, monkeypatch):
    _, app_module = app_client
    monkeypatch.delenv("MAGNIDATA_PORTAL_URL", raising=False)
    monkeypatch.delenv("MAGNIDATA_API_URL", raising=False)

    assert app_module._access_banner() is None


# ── GET /api/datasets/demos ──────────────────────────────────────────────────────


def test_list_demos_returns_the_static_registry(app_client):
    client, _ = app_client
    r = client.get("/api/datasets/demos")

    assert r.status_code == 200
    body = r.get_json()
    by_key = {d["key"]: d for d in body}
    assert by_key["coco128"] == {
        "key": "coco128",
        "name": "COCO128",
        "description": "Ultralytics COCO128 sample, built in-app (embeddings precomputed offline and shipped with the tool)",
        "enabled": True,
    }
    assert by_key["agristress500"]["enabled"] is True


def test_build_limits_returns_configured_caps(app_client):
    client, app_module = app_client
    r = client.get("/api/datasets/build/limits")

    assert r.status_code == 200
    assert r.get_json() == {
        "max_images": app_module.dataset_builder.MAX_IMAGES,
        "max_upload_bytes": app_module.dataset_builder.MAX_TOTAL_BYTES,
        "max_embeddings_bytes": app_module.dataset_builder.MAX_EMBEDDINGS_BYTES,
        "max_request_bytes": app_module.dataset_builder.MAX_REQUEST_BYTES,
    }


# ── POST /api/datasets/build ──────────────────────────────────────────────────────


def test_build_rejects_unknown_source_type(app_client):
    client, _ = app_client
    r = client.post("/api/datasets/build", data={"source_type": "bogus"}, content_type="multipart/form-data")
    assert r.status_code == 400


def test_build_upload_returns_job_id(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.setattr(
        app_module.dataset_builder,
        "start_upload_build",
        lambda images, annotations, embeddings_file, name, description: "job-123",
    )

    r = client.post(
        "/api/datasets/build",
        data={
            "source_type": "upload",
            "name": "Test Set",
            "images": (io.BytesIO(b"fake"), "sub/img.png"),
        },
        content_type="multipart/form-data",
    )

    assert r.status_code == 202
    assert r.get_json() == {"job_id": "job-123"}


def test_build_upload_passes_uploaded_files_with_relative_paths(app_client, monkeypatch):
    client, app_module = app_client
    captured = {}

    def fake_start(images, annotations, embeddings_file, name, description):
        captured["images"] = images
        captured["annotations"] = annotations
        captured["embeddings_file"] = embeddings_file
        captured["name"] = name
        return "job-123"

    monkeypatch.setattr(app_module.dataset_builder, "start_upload_build", fake_start)

    client.post(
        "/api/datasets/build",
        data={
            "source_type": "upload",
            "name": "Test Set",
            "images": [
                (io.BytesIO(b"AAAA"), "weeds/a.png"),
                (io.BytesIO(b"BBBB"), "soil/b.png"),
            ],
        },
        content_type="multipart/form-data",
    )

    assert captured["name"] == "Test Set"
    assert {i.relative_path for i in captured["images"]} == {"weeds/a.png", "soil/b.png"}
    assert [Path(i.source_path).read_bytes() for i in captured["images"]] == [b"AAAA", b"BBBB"]
    assert captured["annotations"] == []
    assert captured["embeddings_file"] is None


def test_build_upload_passes_annotation_files(app_client, monkeypatch):
    client, app_module = app_client
    captured = {}

    def fake_start(images, annotations, embeddings_file, name, description):
        captured["images"] = images
        captured["annotations"] = annotations
        return "job-123"

    monkeypatch.setattr(app_module.dataset_builder, "start_upload_build", fake_start)

    client.post(
        "/api/datasets/build",
        data={
            "source_type": "upload",
            "name": "Test Set",
            "images": (io.BytesIO(b"AAAA"), "weeds/a.png"),
            "annotations": (io.BytesIO(b'{"a": 1}'), "weeds/a.json"),
        },
        content_type="multipart/form-data",
    )

    assert len(captured["annotations"]) == 1
    ann = captured["annotations"][0]
    assert ann.relative_path == "weeds/a.json"
    assert ann.data is None
    assert Path(ann.source_path).read_bytes() == b'{"a": 1}'


def test_build_upload_passes_embeddings_file(app_client, monkeypatch):
    client, app_module = app_client
    captured = {}

    def fake_start(images, annotations, embeddings_file, name, description):
        captured["embeddings_file"] = embeddings_file
        return "job-123"

    monkeypatch.setattr(app_module.dataset_builder, "start_upload_build", fake_start)

    client.post(
        "/api/datasets/build",
        data={
            "source_type": "upload",
            "name": "Test Set",
            "images": (io.BytesIO(b"AAAA"), "a.png"),
            "embeddings": (io.BytesIO(b'{"embeddings": {}}'), "embeddings.json"),
        },
        content_type="multipart/form-data",
    )

    embeddings_file = captured["embeddings_file"]
    assert embeddings_file.relative_path == "embeddings.json"
    assert Path(embeddings_file.source_path).read_bytes() == b'{"embeddings": {}}'


def test_build_demo_passes_demo_key(app_client, monkeypatch):
    client, app_module = app_client
    captured = {}

    def fake_start(demo_key):
        captured["demo_key"] = demo_key
        return "job-456"

    monkeypatch.setattr(app_module.dataset_builder, "start_demo_build", fake_start)

    r = client.post(
        "/api/datasets/build",
        data={
            "source_type": "demo",
            "demo_key": "coco128",
        },
        content_type="multipart/form-data",
    )

    assert r.status_code == 202
    assert r.get_json() == {"job_id": "job-456"}
    assert captured == {"demo_key": "coco128"}


def test_build_returns_400_on_validation_error(app_client, monkeypatch):
    client, app_module = app_client

    def fake_start(images, annotations, embeddings_file, name, description):
        raise app_module.dataset_builder.BuildValidationError("name is required")

    monkeypatch.setattr(app_module.dataset_builder, "start_upload_build", fake_start)

    r = client.post(
        "/api/datasets/build",
        data={
            "source_type": "upload",
            "name": "",
            "images": (io.BytesIO(b"x"), "a.png"),
        },
        content_type="multipart/form-data",
    )
    assert r.status_code == 400
    incoming = Path(app_module.DATA_ROOT) / "data_user" / app_module.dataset_builder.INCOMING_UPLOAD_DIRNAME
    assert not incoming.exists() or not any(incoming.iterdir())


def test_build_returns_json_413_for_oversized_request(app_client):
    client, app_module = app_client
    app_module.app.config["MAX_CONTENT_LENGTH"] = 128

    r = client.post(
        "/api/datasets/build",
        data={
            "source_type": "upload",
            "name": "Too Big",
            "images": (io.BytesIO(b"x" * 512), "a.png"),
        },
        content_type="multipart/form-data",
    )

    assert r.status_code == 413
    assert "request body too large" in r.get_json()["error"]


def test_build_returns_409_when_a_build_is_in_progress(app_client, monkeypatch):
    client, app_module = app_client

    def fake_start(demo_key):
        raise app_module.dataset_builder.BuildInProgressError("busy")

    monkeypatch.setattr(app_module.dataset_builder, "start_demo_build", fake_start)

    r = client.post(
        "/api/datasets/build", data={"source_type": "demo", "demo_key": "coco128"}, content_type="multipart/form-data"
    )
    assert r.status_code == 409


# ── GET /api/datasets/build/<job_id> ──────────────────────────────────────────────


def test_get_build_job_returns_job_status(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.setattr(
        app_module.dataset_builder,
        "get_job",
        lambda job_id: {"status": "done", "percent": 100, "message": "Done", "dataset": {"id": 1}, "error": None},
    )

    r = client.get("/api/datasets/build/job-1")

    assert r.status_code == 200
    assert r.get_json()["status"] == "done"


def test_get_build_job_404_for_unknown_id(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.setattr(app_module.dataset_builder, "get_job", lambda job_id: None)

    r = client.get("/api/datasets/build/does-not-exist")
    assert r.status_code == 404
    assert r.get_json() == {"error": "job not found"}


# ── DELETE /api/datasets/<id> cleanup ──────────────────────────────────────────────


def test_cloud_union_lineage_and_complement_derivation(app_client):
    client, app_module = app_client
    root = Path(app_module.DATA_ROOT)
    mini1 = "data_user/agristress500-mini1.csv"
    mini2 = "data_user/agristress500-mini2.csv"
    _write_feature_csv(
        root / mini1,
        [
            {"image_path": "images/a.png", "width": "256", "score": "0"},
            {"image_path": "images/b.png", "width": "256", "score": "10"},
        ],
    )
    _write_feature_csv(
        root / mini2,
        [
            {"image_path": "images/c.png", "width": "256", "score": "1"},
            {"image_path": "images/d.png", "width": "256", "score": "9"},
        ],
    )
    app_module.db.create_dataset_record("Mini 1", "", mini1, None, None, 2)
    app_module.db.create_dataset_record("Mini 2", "", mini2, None, None, 2)

    catalog = client.get("/api/datasets").get_json()
    by_source = {dataset["source"]: dataset for dataset in catalog}
    cloud = by_source[app_module.dataset_curation.CLOUD_SOURCE]
    assert cloud["name"] == "Cloud Dataset"
    assert cloud["row_count"] == 4
    assert cloud["deletable"] is False
    assert by_source[mini1]["parent_sources"] == [app_module.dataset_curation.CLOUD_SOURCE]
    assert by_source[mini2]["parent_sources"] == [app_module.dataset_curation.CLOUD_SOURCE]

    response = client.post(
        "/api/datasets",
        json={
            "name": "Mini 1 complemented",
            "parent_source": mini1,
            "image_names": ["a.png", "b.png"],
            "query_image_names": ["a.png"],
            "complement_top_k": 1,
        },
    )
    assert response.status_code == 201
    derived = response.get_json()
    assert derived["complement_count"] == 1
    assert derived["row_count"] == 3
    assert derived["parent_sources"] == [mini1, app_module.dataset_curation.CLOUD_SOURCE]
    _, rows = app_module.dataset_curation.read_rows(root / derived["source"])
    assert [Path(row["image_path"]).name for row in rows] == ["a.png", "b.png", "c.png"]


def test_removal_only_dataset_has_current_dataset_as_sole_parent(app_client):
    client, app_module = app_client
    source = "data_user/current.csv"
    _write_feature_csv(
        Path(app_module.DATA_ROOT) / source,
        [
            {"image_path": "images/a.png", "width": "256", "score": "1"},
            {"image_path": "images/b.png", "width": "256", "score": "2"},
        ],
    )
    app_module.db.create_dataset_record("Current", "", source, None, None, 2)

    response = client.post(
        "/api/datasets",
        json={
            "name": "Removed B",
            "parent_source": source,
            "image_names": ["a.png"],
        },
    )

    assert response.status_code == 201
    assert response.get_json()["parent_sources"] == [source]


def test_remove_dataset_cleans_up_built_dataset_files(app_client):
    client, app_module = app_client
    db = app_module.db
    data_root = app_module.DATA_ROOT
    csv_path = os.path.join(data_root, "data_user", "myset.csv")
    json_path = os.path.join(data_root, "data_user", "myset.json")
    images_dir = os.path.join(data_root, "data_user", "myset_images")
    os.makedirs(images_dir)
    open(csv_path, "w").close()
    open(json_path, "w").close()
    open(csv_path + ".provenance.json", "w").close()
    with open(os.path.join(images_dir, "a.png"), "w") as f:
        f.write("x")
    rec = db.create_dataset_record("My Set", "", "data_user/myset.csv", None, "data_user/myset.json", 1)

    r = client.delete(f"/api/datasets/{rec['id']}")

    assert r.status_code == 204
    assert not os.path.exists(csv_path)
    assert not os.path.exists(json_path)
    assert not os.path.exists(csv_path + ".provenance.json")
    assert not os.path.isdir(images_dir)


def test_remove_dataset_child_only_removes_its_own_csv(app_client):
    client, app_module = app_client
    db = app_module.db
    data_root = app_module.DATA_ROOT
    parent_json = os.path.join(data_root, "data_user", "parent.json")
    child_csv = os.path.join(data_root, "data_user", "child.csv")
    os.makedirs(os.path.join(data_root, "data_user"), exist_ok=True)
    open(parent_json, "w").close()
    open(child_csv, "w").close()
    rec = db.create_dataset_record("Child", "", "data_user/child.csv", "data/parent.csv", "data/parent.csv", 1)

    r = client.delete(f"/api/datasets/{rec['id']}")

    assert r.status_code == 204
    assert not os.path.exists(child_csv)
    assert os.path.exists(parent_json)  # the parent's embeddings must survive


# ── GET /api/annotation/mask + /api/annotation/overlay ─────────────────────────────


def _write_labeled_image(tmp_path, with_label=True):
    """<tmp_path>/data/ds1/images/a.png (40x40) + a COCO label covering its left half.
    Returns the DATA_ROOT-relative image_path."""
    images_dir = tmp_path / "data" / "ds1" / "images"
    labels_dir = tmp_path / "data" / "ds1" / "labels"
    images_dir.mkdir(parents=True)
    Image.fromarray(np.full((40, 40, 3), 255, np.uint8), "RGB").save(images_dir / "a.png")
    if with_label:
        labels_dir.mkdir(parents=True)
        doc = {
            "images": [{"id": 1, "file_name": "a.png", "width": 40, "height": 40}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": [[0, 0, 20, 0, 20, 40, 0, 40]],  # left half
                }
            ],
            "categories": [{"id": 0, "name": "background"}, {"id": 1, "name": "weed"}],
        }
        (labels_dir / "a.json").write_text(json.dumps(doc))
    return "data/ds1/images/a.png"


def test_annotation_mask_404_without_label_file(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.chdir(app_module.DATA_ROOT)
    path = _write_labeled_image(Path(app_module.DATA_ROOT), with_label=False)

    r = client.get(f"/api/annotation/mask?path={path}")

    assert r.status_code == 404
    assert r.get_json() == {"error": "no annotations for this image"}


def test_annotation_mask_returns_colored_png(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.chdir(app_module.DATA_ROOT)
    path = _write_labeled_image(Path(app_module.DATA_ROOT))

    r = client.get(f"/api/annotation/mask?path={path}")

    assert r.status_code == 200
    assert r.content_type == "image/png"
    mask = Image.open(io.BytesIO(r.data))
    arr = np.array(mask)
    assert arr.shape == (40, 40, 4)
    # Left half (annotated) is opaque and colored; right half (unlabeled) stays transparent.
    assert arr[20, 5, 3] == 255
    assert tuple(arr[20, 5, :3]) != (0, 0, 0)
    assert arr[20, 35, 3] == 0


def test_annotation_overlay_404_without_label_file(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.chdir(app_module.DATA_ROOT)
    path = _write_labeled_image(Path(app_module.DATA_ROOT), with_label=False)

    r = client.get(f"/api/annotation/overlay?path={path}")

    assert r.status_code == 404


def test_annotation_overlay_alpha_zero_keeps_image_unchanged(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.chdir(app_module.DATA_ROOT)
    path = _write_labeled_image(Path(app_module.DATA_ROOT))

    r = client.get(f"/api/annotation/overlay?path={path}&alpha=0")

    assert r.status_code == 200
    assert r.content_type == "image/jpeg"
    result = np.array(Image.open(io.BytesIO(r.data)).convert("RGB"))
    # Source image was solid white — alpha=0 means the mask contributes nothing.
    assert result[20, 5].tolist() == [255, 255, 255]


def test_annotation_overlay_alpha_one_shows_full_mask_color(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.chdir(app_module.DATA_ROOT)
    path = _write_labeled_image(Path(app_module.DATA_ROOT))

    r = client.get(f"/api/annotation/overlay?path={path}&alpha=1")

    assert r.status_code == 200
    result = np.array(Image.open(io.BytesIO(r.data)).convert("RGB"))
    # Fully opaque mask over the annotated region should no longer be white.
    assert result[20, 5].tolist() != [255, 255, 255]
    # The unlabeled region is untouched regardless of alpha.
    assert result[20, 35].tolist() == [255, 255, 255]


# ── GET /api/annotation/mask + overlay — raster-mask fallback ──────────────────────
# For a dataset with no COCO labels but a pre-colored masks/<stem>.png sibling (e.g.
# AgriStress-500) — the same "left half colored, right half black background" fixture
# as the COCO tests above, but as a raster mask file instead of a polygon label.


def _write_image_with_raster_mask(tmp_path):
    images_dir = tmp_path / "data" / "ds2" / "images"
    masks_dir = tmp_path / "data" / "ds2" / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    Image.fromarray(np.full((40, 40, 3), 255, np.uint8), "RGB").save(images_dir / "a.png")
    mask = np.zeros((40, 40, 3), np.uint8)
    mask[:, :20] = [90, 60, 200]  # left half colored (foreground), right half black (bg)
    Image.fromarray(mask, "RGB").save(masks_dir / "a.png")
    return "data/ds2/images/a.png"


def test_annotation_mask_falls_back_to_raster_mask_file(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.chdir(app_module.DATA_ROOT)
    path = _write_image_with_raster_mask(Path(app_module.DATA_ROOT))

    r = client.get(f"/api/annotation/mask?path={path}")

    assert r.status_code == 200
    assert r.content_type == "image/png"
    arr = np.array(Image.open(io.BytesIO(r.data)))
    assert arr.shape == (40, 40, 4)
    assert arr[20, 5, 3] == 255  # colored left half is opaque
    assert tuple(arr[20, 5, :3]) == (90, 60, 200)  # ...and keeps its original color
    assert arr[20, 35, 3] == 0  # black right half is transparent


def test_annotation_overlay_falls_back_to_raster_mask_file(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.chdir(app_module.DATA_ROOT)
    path = _write_image_with_raster_mask(Path(app_module.DATA_ROOT))

    r = client.get(f"/api/annotation/overlay?path={path}&alpha=1")

    assert r.status_code == 200
    result = np.array(Image.open(io.BytesIO(r.data)).convert("RGB"))
    assert result[20, 5].tolist() != [255, 255, 255]  # masked region shows the color
    assert result[20, 35].tolist() == [255, 255, 255]  # unmasked region stays white


def test_annotation_mask_prefers_coco_labels_over_raster_mask(app_client, monkeypatch):
    """When a dataset somehow has both, the exact per-instance COCO polygons win over
    the coarser raster mask (see _load_annotation_mask's priority order)."""
    client, app_module = app_client
    monkeypatch.chdir(app_module.DATA_ROOT)
    path = _write_labeled_image(Path(app_module.DATA_ROOT))  # COCO label, left-half square
    # Also drop a raster mask claiming the *right* half — if this wins, the test below fails.
    masks_dir = Path(app_module.DATA_ROOT) / "data" / "ds1" / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((40, 40, 3), np.uint8)
    mask[:, 20:] = [1, 2, 3]
    Image.fromarray(mask, "RGB").save(masks_dir / "a.png")

    r = client.get(f"/api/annotation/mask?path={path}")

    arr = np.array(Image.open(io.BytesIO(r.data)))
    assert arr[20, 5, 3] == 255  # COCO's left-half annotation still wins
    assert arr[20, 35, 3] == 0  # not the raster mask's right half


# ── GET /api/datasets/build/local-folders ─────────────────────────────────────────


def test_list_local_folders_returns_the_index(app_client, monkeypatch):
    client, app_module = app_client
    index = [{"folder": "MMDE-POC", "image_count": 201, "label_count": 201, "built": False}]
    monkeypatch.setattr(app_module.dataset_builder, "list_local_folders", lambda: index)

    r = client.get("/api/datasets/build/local-folders")

    assert r.status_code == 200
    assert r.get_json() == index


def test_list_local_folders_returns_empty_list_when_none_present(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.setattr(app_module.dataset_builder, "list_local_folders", lambda: [])

    r = client.get("/api/datasets/build/local-folders")

    assert r.status_code == 200
    assert r.get_json() == []


# ── POST /api/datasets/build (source_type=local) ──────────────────────────────────


def test_build_local_returns_job_id(app_client, monkeypatch):
    client, app_module = app_client
    seen = {}

    def fake_start(folder, name, description):
        seen.update(folder=folder, name=name, description=description)
        return "job-local"

    monkeypatch.setattr(app_module.dataset_builder, "start_local_build", fake_start)

    r = client.post(
        "/api/datasets/build",
        data={"source_type": "local", "folder": "MMDE-POC", "name": "MMDE POC", "description": "soybeans"},
        content_type="multipart/form-data",
    )

    assert r.status_code == 202
    assert r.get_json() == {"job_id": "job-local"}
    assert seen == {"folder": "MMDE-POC", "name": "MMDE POC", "description": "soybeans"}


def test_build_local_defaults_name_and_description_to_blank(app_client, monkeypatch):
    client, app_module = app_client
    seen = {}

    def fake_start(folder, name, description):
        seen.update(folder=folder, name=name, description=description)
        return "job-local"

    monkeypatch.setattr(app_module.dataset_builder, "start_local_build", fake_start)

    r = client.post(
        "/api/datasets/build",
        data={"source_type": "local", "folder": "MMDE-POC"},
        content_type="multipart/form-data",
    )

    assert r.status_code == 202
    assert seen == {"folder": "MMDE-POC", "name": "", "description": ""}


def test_build_local_returns_400_for_unknown_folder(app_client, monkeypatch):
    client, app_module = app_client

    def fake_start(folder, name, description):
        raise app_module.dataset_builder.BuildValidationError("image set 'nope' not found under image_sets/")

    monkeypatch.setattr(app_module.dataset_builder, "start_local_build", fake_start)

    r = client.post(
        "/api/datasets/build",
        data={"source_type": "local", "folder": "nope"},
        content_type="multipart/form-data",
    )

    assert r.status_code == 400
    assert "not found" in r.get_json()["error"]


def test_build_local_returns_409_when_a_build_is_in_progress(app_client, monkeypatch):
    client, app_module = app_client

    def fake_start(folder, name, description):
        raise app_module.dataset_builder.BuildInProgressError("busy")

    monkeypatch.setattr(app_module.dataset_builder, "start_local_build", fake_start)

    r = client.post(
        "/api/datasets/build",
        data={"source_type": "local", "folder": "MMDE-POC"},
        content_type="multipart/form-data",
    )

    assert r.status_code == 409


# ── Base matrix ──────────────────────────────────────────────────────────────────


def _write_dataset(tmp_path, stem, rows, payload):
    """Write a minimal dataset CSV plus its embeddings JSON and return the CSV source path."""
    csv_path = tmp_path / "data_user" / f"{stem}.csv"
    csv_path.write_text("image_path\n" + "".join(f"images/{r}\n" for r in rows))
    (tmp_path / "data_user" / f"{stem}.json").write_text(json.dumps(payload))
    return f"data_user/{stem}.csv"


def test_base_matrix_aligns_vectors_to_csv_row_order(app_client, tmp_path):
    _, app_module = app_client
    source = _write_dataset(
        tmp_path, "aligned", ["b.jpg", "a.jpg"], {"embeddings": {"a.jpg": [1.0, 2.0], "b.jpg": [3.0, 4.0]}}
    )

    matrix = app_module._base_matrix(source, None)

    np.testing.assert_array_equal(matrix, np.array([[3.0, 4.0], [1.0, 2.0]], dtype=np.float32))


def test_base_matrix_zero_fills_rows_without_an_embedding(app_client, tmp_path):
    _, app_module = app_client
    source = _write_dataset(tmp_path, "sparse", ["a.jpg", "missing.jpg"], {"embeddings": {"a.jpg": [1.0, 2.0]}})

    matrix = app_module._base_matrix(source, None)

    np.testing.assert_array_equal(matrix, np.array([[1.0, 2.0], [0.0, 0.0]], dtype=np.float32))


def test_base_matrix_zero_fills_wrong_length_vectors(app_client, tmp_path):
    _, app_module = app_client
    source = _write_dataset(
        tmp_path, "ragged", ["a.jpg", "b.jpg"], {"embeddings": {"a.jpg": [1.0, 2.0], "b.jpg": [9.0]}}
    )

    matrix = app_module._base_matrix(source, None)

    np.testing.assert_array_equal(matrix, np.array([[1.0, 2.0], [0.0, 0.0]], dtype=np.float32))


def test_base_matrix_repeats_a_vector_across_duplicate_image_names(app_client, tmp_path):
    _, app_module = app_client
    source = _write_dataset(tmp_path, "dupes", ["a.jpg", "a.jpg"], {"embeddings": {"a.jpg": [1.0, 2.0]}})

    matrix = app_module._base_matrix(source, None)

    np.testing.assert_array_equal(matrix, np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32))


def test_base_matrix_parses_the_file_once_for_concurrent_callers(app_client, tmp_path):
    """Concurrent requests must share one parse — otherwise peak memory scales with how
    many overlap, which is what OOM-kills the API on a multi-gigabyte embeddings file."""
    _, app_module = app_client
    source = _write_dataset(tmp_path, "shared", ["a.jpg", "b.jpg"], {"embeddings": {"a.jpg": [1.0], "b.jpg": [2.0]}})

    parses = []
    original = app_module.embeddings_io.iter_entries
    started = threading.Event()

    def counting_iter(path, **kwargs):
        parses.append(path)
        started.set()
        yield from original(path, **kwargs)

    app_module.embeddings_io.iter_entries = counting_iter
    try:
        results = []
        threads = [
            threading.Thread(target=lambda: results.append(app_module._base_matrix(source, None))) for _ in range(6)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
    finally:
        app_module.embeddings_io.iter_entries = original

    assert len(parses) == 1
    assert len(results) == 6
    for matrix in results:
        np.testing.assert_array_equal(matrix, np.array([[1.0], [2.0]], dtype=np.float32))


# ── Precomputed projections ──────────────────────────────────────────────────────


def _write_projection_sidecar(tmp_path, stem, row_count, positions_by_method):
    (tmp_path / "data_user" / f"{stem}_projections.json").write_text(
        json.dumps({"row_count": row_count, "projections": positions_by_method})
    )


def test_projection_serves_precomputed_positions(app_client, tmp_path, monkeypatch):
    """The whole point of the sidecar: the view renders stored points, nothing is reduced."""
    client, app_module = app_client
    source = _write_dataset(
        tmp_path,
        "stored",
        ["a.jpg", "b.jpg", "c.jpg"],
        {"embeddings": {"a.jpg": [1.0], "b.jpg": [2.0], "c.jpg": [3.0]}},
    )
    stored = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    _write_projection_sidecar(tmp_path, "stored", 3, {"pca": stored})

    def fail_if_called(*args, **kwargs):
        raise AssertionError("projection was computed despite a usable sidecar")

    monkeypatch.setattr(app_module.projections, "project", fail_if_called)

    resp = client.get(f"/api/embedding/projection?source={source}&method=pca")

    assert resp.status_code == 200
    assert resp.get_json()["positions"] == stored


def test_projection_ignores_a_sidecar_with_a_stale_row_count(app_client, tmp_path):
    client, _ = app_client
    source = _write_dataset(
        tmp_path, "stale", ["a.jpg", "b.jpg", "c.jpg"], {"embeddings": {"a.jpg": [1.0], "b.jpg": [2.0], "c.jpg": [3.0]}}
    )
    _write_projection_sidecar(tmp_path, "stale", 99, {"pca": [[0.0, 0.0, 0.0]]})

    resp = client.get(f"/api/embedding/projection?source={source}&method=pca")

    assert resp.status_code == 200
    assert len(resp.get_json()["positions"]) == 3  # computed, not the one stale position


def test_projection_ignores_the_sidecar_for_a_variant(app_client, tmp_path):
    """A variant selects another model's vectors; the base dataset's sidecar does not describe them."""
    client, _ = app_client
    source = _write_dataset(
        tmp_path, "var", ["a.jpg", "b.jpg", "c.jpg"], {"embeddings": {"a.jpg": [1.0], "b.jpg": [2.0], "c.jpg": [3.0]}}
    )
    (tmp_path / "data_user" / "var_other.json").write_text(
        json.dumps({"embeddings": {"a.jpg": [9.0], "b.jpg": [8.0], "c.jpg": [7.0]}})
    )
    stored = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    _write_projection_sidecar(tmp_path, "var", 3, {"pca": stored})

    resp = client.get(f"/api/embedding/projection?source={source}&method=pca&variant=other")

    assert resp.status_code == 200
    assert resp.get_json()["positions"] != stored


def test_projection_falls_back_to_computing_without_a_sidecar(app_client, tmp_path):
    client, _ = app_client
    source = _write_dataset(
        tmp_path,
        "nosidecar",
        ["a.jpg", "b.jpg", "c.jpg"],
        {"embeddings": {"a.jpg": [1.0, 0.0], "b.jpg": [0.0, 1.0], "c.jpg": [1.0, 1.0]}},
    )

    resp = client.get(f"/api/embedding/projection?source={source}&method=pca")

    assert resp.status_code == 200
    assert len(resp.get_json()["positions"]) == 3


def test_projection_rejects_an_unknown_method(app_client, tmp_path):
    client, _ = app_client
    source = _write_dataset(
        tmp_path,
        "unknown",
        ["a.jpg", "b.jpg", "c.jpg"],
        {"embeddings": {"a.jpg": [1.0], "b.jpg": [2.0], "c.jpg": [3.0]}},
    )

    resp = client.get(f"/api/embedding/projection?source={source}&method=umap")

    assert resp.status_code == 400


def test_embedding_routes_reject_traversal_and_invalid_variants(app_client):
    client, _ = app_client

    assert client.get("/api/dataset/csv?source=../secret.csv").status_code == 400
    assert client.get("/api/embedding/projection?source=data_user/demo.csv&variant=../secret").status_code == 400
    assert client.get("/api/embedding/clusters?source=data_user/demo.csv&variant=secret%2Fother").status_code == 400


def test_data_path_rejects_symlink_escape(app_client, tmp_path):
    _, app_module = app_client
    outside = tmp_path.parent / f"{tmp_path.name}-outside.csv"
    outside.write_text("not served\n")
    link = tmp_path / "data_user" / "escape.csv"
    link.symlink_to(outside)

    with pytest.raises(ValueError, match="invalid data path"):
        app_module._data_path("data_user/escape.csv")


# ── Dataset CSV/embeddings caching ─────────────────────────────────────────────────


def test_dataset_csv_is_not_cacheable(app_client, tmp_path):
    """A dataset's CSV can be rebuilt in place at the same path, so its response must
    never be cached or resumed by the client against bytes that no longer match."""
    client, _ = app_client
    (tmp_path / "data_user" / "plain.csv").write_text("image_path\nimages/a.jpg\n")

    resp = client.get("/api/dataset/csv?source=data_user/plain.csv")

    assert resp.status_code == 200
    # no-store is what actually matters: the browser must never keep this body around to
    # revalidate or range-resume against later, regardless of what conditional headers
    # Werkzeug still attaches to the response.
    assert resp.headers["Cache-Control"] == "no-store"


def test_dataset_csv_ignores_a_range_header(app_client, tmp_path):
    """Range support is what lets a client resume a stale cached body; it is turned off
    for a file that can be rebuilt in place."""
    client, _ = app_client
    (tmp_path / "data_user" / "ranged.csv").write_text("image_path\nimages/a.jpg\n")

    resp = client.get("/api/dataset/csv?source=data_user/ranged.csv", headers={"Range": "bytes=0-2"})

    assert resp.status_code == 200  # not 206 — the whole, current file every time
    assert resp.data == b"image_path\nimages/a.jpg\n"


def test_dataset_embeddings_is_not_cacheable(app_client, tmp_path):
    client, _ = app_client
    (tmp_path / "data_user" / "plain.csv").write_text("image_path\nimages/a.jpg\n")
    (tmp_path / "data_user" / "plain.json").write_text('{"embeddings": {"a.jpg": [1.0]}}')

    resp = client.get("/api/dataset/embeddings?source=data_user/plain.csv")

    assert resp.status_code == 200
    assert resp.headers["Cache-Control"] == "no-store"


def test_mlflow_status_reachable_and_unreachable(app_client, tmp_path):
    client, _ = app_client
    try:
        mlflow_store.set_store(mlflow_store.MlflowStore(f"sqlite:///{tmp_path / 'mlflow.db'}"))
        resp = client.get("/api/mlflow/status")
        assert resp.status_code == 200
        assert resp.get_json()["reachable"] is True

        mlflow_store.set_store(mlflow_store.MlflowStore("http://127.0.0.1:1"))
        resp = client.get("/api/mlflow/status")
        assert resp.status_code == 200
        assert resp.get_json()["reachable"] is False
    finally:
        mlflow_store.set_store(None)


def test_experiment_routes_are_registered(app_client):
    _, app_module = app_client
    rules = {rule.rule for rule in app_module.app.url_map.iter_rules()}
    assert {"/api/experiments", "/api/runs", "/api/runs/<run_id>/metrics", "/api/runs/<run_id>/link"} <= rules


@pytest.fixture
def masked_image(app_client, tmp_path, monkeypatch):
    """A 20x10 green image with a red mask over its left half, in the <dataset>/images + masks layout."""
    monkeypatch.chdir(tmp_path)  # these endpoints open paths relative to the working directory (/app in Docker)
    folder = tmp_path / "ds"
    (folder / "images").mkdir(parents=True)
    (folder / "masks").mkdir()
    Image.new("RGB", (20, 10), (10, 160, 10)).save(folder / "images" / "x.png")
    mask = np.zeros((10, 20, 3), dtype=np.uint8)
    mask[:, :10] = (230, 20, 20)
    Image.fromarray(mask).save(folder / "masks" / "x.png")
    return "ds/images/x.png", "ds/masks/x.png"


def _pixel(resp, xy):
    x, y = xy
    return tuple(int(v) for v in np.asarray(Image.open(io.BytesIO(resp.data)).convert("RGB"))[y, x])


def test_overlay_composites_mask_over_image(app_client, masked_image):
    client, _ = app_client
    img, mask = masked_image
    resp = client.get(f"/api/overlay?img={img}&mask={mask}")
    assert resp.status_code == 200
    assert resp.mimetype == "image/jpeg"
    assert _pixel(resp, (3, 5))[0] > 180  # masked half is red
    assert _pixel(resp, (15, 5))[1] > 120  # background half keeps the green image


def test_annotation_overlay_honours_alpha(app_client, masked_image):
    client, _ = app_client
    img, _ = masked_image
    full = client.get(f"/api/annotation/overlay?path={img}&alpha=1")
    assert full.status_code == 200
    assert _pixel(full, (3, 5))[0] > 180
    none = client.get(f"/api/annotation/overlay?path={img}&alpha=0")
    assert _pixel(none, (3, 5))[1] > 120
    half = _pixel(client.get(f"/api/annotation/overlay?path={img}&alpha=0.5"), (3, 5))
    assert 80 < half[0] < 200
    assert 40 < half[1] < 140


def test_annotation_overlay_without_mask_is_404(app_client, tmp_path, monkeypatch):
    client, _ = app_client
    monkeypatch.chdir(tmp_path)
    (tmp_path / "plain" / "images").mkdir(parents=True)
    Image.new("RGB", (4, 4), (1, 2, 3)).save(tmp_path / "plain" / "images" / "y.png")
    assert client.get("/api/annotation/overlay?path=plain/images/y.png").status_code == 404
    assert client.get("/api/annotation/overlay?path=../x.png").status_code == 400


def test_mask_endpoint_makes_background_transparent(app_client, masked_image):
    client, _ = app_client
    _, mask = masked_image
    resp = client.get(f"/api/mask?path={mask}")
    assert resp.mimetype == "image/png"
    alpha = np.asarray(Image.open(io.BytesIO(resp.data)).convert("RGBA"))[:, :, 3]
    assert alpha[5, 3] == 255
    assert alpha[5, 15] == 0


# ── Dataset summary ────────────────────────────────────────────────────────────


def test_dataset_summary_reports_stats_and_samples(app_client, tmp_path):
    client, _ = app_client
    rows = [{"image_path": f"images/{i}.png", "width": str(100 + i), "score": "0.5"} for i in range(30)]
    _write_feature_csv(tmp_path / "data_user" / "s.csv", rows)

    res = client.get("/api/datasets/summary?source=data_user/s.csv")

    assert res.status_code == 200
    body = res.get_json()
    assert body["row_count"] == 30
    assert body["column_count"] == 3
    assert body["image_column"] == "image_path"
    assert len(body["sample_images"]) == 8
    assert body["mask_coverage"] == {"checked": 30, "with_masks": 0}
    names = [c["name"] for c in body["numeric_columns"]]
    assert names == ["width", "score"]
    assert sum(body["numeric_columns"][0]["histogram"]) == 30


def test_dataset_summary_rejects_bad_sources(app_client):
    client, _ = app_client

    assert client.get("/api/datasets/summary").status_code == 400
    assert client.get("/api/datasets/summary?source=../etc/passwd").status_code == 400
    assert client.get("/api/datasets/summary?source=data_user/missing.csv").status_code == 404


def test_confi_datasets_report_row_count(app_client, tmp_path):
    client, _ = app_client
    _write_feature_csv(tmp_path / "data_user" / "c.csv", [{"image_path": "a.png", "width": "1", "score": "2"}])
    (tmp_path / "confi.yaml").write_text("datasets:\n  - name: C\n    source: data_user/c.csv\n")

    listed = client.get("/api/datasets").get_json()

    assert next(d for d in listed if d["source"] == "data_user/c.csv")["row_count"] == 1


# ── Dataset merge ──────────────────────────────────────────────────────────────


def _merge(client, **body):
    return client.post("/api/datasets/merge", json=body)


def _two_sources(tmp_path):
    _write_feature_csv(
        tmp_path / "data_user" / "m1.csv",
        [
            {"image_path": "images/a.png", "width": "1", "score": "1"},
            {"image_path": "images/b.png", "width": "2", "score": "2"},
        ],
    )
    _write_feature_csv(
        tmp_path / "data_user" / "m2.csv",
        [
            {"image_path": "images/b.png", "width": "2", "score": "2"},
            {"image_path": "images/c.png", "width": "3", "score": "3"},
        ],
    )
    return ["data_user/m1.csv", "data_user/m2.csv"]


def test_merge_writes_deduplicated_union_with_every_parent(app_client, tmp_path):
    client, _ = app_client
    sources = _two_sources(tmp_path)

    res = _merge(client, name="Both", sources=sources)

    assert res.status_code == 201
    body = res.get_json()
    assert body["row_count"] == 3
    assert body["parent_sources"] == sources
    assert body["deletable"] is True
    with open(tmp_path / body["source"]) as handle:
        names = [row["image_path"] for row in csv.DictReader(handle)]
    assert names == ["images/a.png", "images/b.png", "images/c.png"]
    listed = next(d for d in client.get("/api/datasets").get_json() if d["source"] == body["source"])
    assert listed["parent_sources"] == sources


@pytest.mark.parametrize(
    ("body", "status", "message"),
    [
        ({"sources": ["a", "b"]}, 400, "invalid merge request"),
        ({"name": "x", "sources": ["data_user/m1.csv"]}, 400, "invalid merge request"),
        ({"name": "x", "sources": ["data_user/m1.csv", "data_user/m1.csv"]}, 400, "invalid merge request"),
        ({"name": "x", "sources": ["data_user/m1.csv", "../etc/passwd"]}, 400, "invalid merge request"),
        ({"name": "x", "sources": "data_user/m1.csv"}, 400, "invalid merge request"),
        ({"name": "x", "sources": ["data_user/m1.csv", "data_user/missing.csv"]}, 404, "not found"),
    ],
)
def test_merge_rejects_bad_requests(app_client, tmp_path, body, status, message):
    client, _ = app_client
    _two_sources(tmp_path)

    res = _merge(client, **body)

    assert res.status_code == status
    assert message in res.get_json()["error"]


def test_merge_rejects_sources_with_different_columns(app_client, tmp_path):
    client, _ = app_client
    sources = _two_sources(tmp_path)
    (tmp_path / "data_user" / "odd.csv").write_text("image_path,other\nimages/z.png,1\n")

    res = _merge(client, name="Odd", sources=[sources[0], "data_user/odd.csv"])

    assert res.status_code == 400
    assert "same CSV columns" in res.get_json()["error"]


def test_merge_name_collisions_get_unique_files(app_client, tmp_path):
    client, _ = app_client
    sources = _two_sources(tmp_path)

    first = _merge(client, name="Same", sources=sources).get_json()
    second = _merge(client, name="Same", sources=sources).get_json()

    assert first["source"] != second["source"]


# ── Ground truth and prediction are different sources ──────────────────────────


def _solid_png(color, size=(8, 6)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _center_pixel(data):
    return tuple(int(v) for v in np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))[3, 4])


def test_annotation_overlay_and_run_prediction_show_different_masks(app_client, tmp_path, monkeypatch):
    client, _ = app_client
    monkeypatch.chdir(tmp_path)  # the annotation endpoints read relative image paths from the working directory
    dataset = tmp_path / "data_user" / "ds"
    (dataset / "images").mkdir(parents=True)
    (dataset / "masks").mkdir()
    (dataset / "images" / "a.png").write_bytes(_solid_png((10, 120, 10)))
    (dataset / "masks" / "a.png").write_bytes(_solid_png((0, 0, 255)))  # ground truth: blue
    (tmp_path / "data_user" / "ds.csv").write_text("image_path\ndata_user/ds/images/a.png\n")

    store = mlflow_store.MlflowStore(f"sqlite:///{tmp_path / 'mlflow.db'}", artifact_root=(tmp_path / "art").as_uri())
    mlflow_store.set_store(store)
    try:
        run_id = seed_run(
            store,
            tags={
                contract.TAG_PER_SAMPLE_SCHEMA: contract.SCHEMA_VERSION,
                contract.TAG_SCORED_EPOCHS: "1",
                contract.TAG_DATASET_SOURCE: "data_user/ds.csv",
            },
            artifacts={contract.per_sample_path(1): "image_name,split,loss_total\na.png,train,0.5\n"},
        )
        prediction = tmp_path / "pred" / "a.png"
        prediction.parent.mkdir()
        prediction.write_bytes(_solid_png((255, 0, 0)))  # prediction: red
        store._client.log_artifact(run_id, str(prediction), "magnidata/predictions/epoch_0001")

        truth = client.get("/api/annotation/overlay?path=data_user/ds/images/a.png&alpha=1")
        predicted = client.get(f"/api/runs/{run_id}/prediction?image=a.png&epoch=1&alpha=1")
        truth_mask = client.get("/api/annotation/mask?path=data_user/ds/images/a.png")

        assert truth.status_code == predicted.status_code == truth_mask.status_code == 200
        blue, red = _center_pixel(truth.data), _center_pixel(predicted.data)
        assert blue[2] > 200
        assert blue[0] < 60
        assert red[0] > 200
        assert red[2] < 60
        assert blue != red
        assert _center_pixel(truth_mask.data) == (0, 0, 255)

        # An image without a saved prediction has none: the prediction endpoint never falls back to the ground truth.
        (dataset / "images" / "b.png").write_bytes(_solid_png((10, 120, 10)))
        (dataset / "masks" / "b.png").write_bytes(_solid_png((0, 0, 255)))
        assert client.get("/api/annotation/overlay?path=data_user/ds/images/b.png&alpha=1").status_code == 200
        assert client.get(f"/api/runs/{run_id}/prediction?image=b.png&epoch=1&alpha=1").status_code == 404
    finally:
        mlflow_store.set_store(None)


# ── Lineage updates ────────────────────────────────────────────────────────────


def _register(client, name, tmp_path, parents=None):
    sources = _two_sources(tmp_path)
    return client.post("/api/datasets/merge", json={"name": name, "sources": sources}).get_json(), sources


def test_set_lineage_replaces_parents_and_appears_in_listing(app_client, tmp_path):
    client, _ = app_client
    merged, sources = _register(client, "Child", tmp_path)

    res = client.post(f"/api/datasets/{merged['id']}/parents", json={"parent_sources": [sources[1]]})

    assert res.status_code == 200
    assert res.get_json()["parent_sources"] == [sources[1]]
    listed = next(d for d in client.get("/api/datasets").get_json() if d["source"] == merged["source"])
    assert listed["parent_sources"] == [sources[1]]


@pytest.mark.parametrize(
    ("body", "status", "message"),
    [
        ({}, 400, "list"),
        ({"parent_sources": []}, 400, "at least one"),
        ({"parent_sources": ["../x.csv"]}, 400, "invalid"),
        ({"parent_sources": ["data_user/missing.csv"]}, 404, "not found"),
    ],
)
def test_set_lineage_rejects_bad_requests(app_client, tmp_path, body, status, message):
    client, _ = app_client
    merged, _ = _register(client, "Child", tmp_path)

    res = client.post(f"/api/datasets/{merged['id']}/parents", json=body)

    assert res.status_code == status
    assert message in res.get_json()["error"]


def test_set_lineage_rejects_self_cycles_and_unknown_datasets(app_client, tmp_path):
    client, _ = app_client
    merged, _ = _register(client, "Child", tmp_path)

    own = client.post(f"/api/datasets/{merged['id']}/parents", json={"parent_sources": [merged["source"]]})
    grandchild = client.post(
        "/api/datasets/merge", json={"name": "Grandchild", "sources": [merged["source"], "data_user/m1.csv"]}
    ).get_json()
    cycle = client.post(f"/api/datasets/{merged['id']}/parents", json={"parent_sources": [grandchild["source"]]})

    assert own.status_code == 400
    assert cycle.status_code == 400
    assert "cycle" in cycle.get_json()["error"]
    assert client.post("/api/datasets/9999/parents", json={"parent_sources": ["x"]}).status_code == 404


# ── Embeddings beside a created dataset ────────────────────────────────────────


def test_created_dataset_picks_up_an_embeddings_file_placed_beside_its_csv(app_client, tmp_path):
    client, _ = app_client
    merged, _ = _register(client, "Embedded Later", tmp_path)
    assert merged["has_embeddings"] is False
    stem = merged["source"].removesuffix(".csv")

    (tmp_path / f"{stem}.json").write_text(json.dumps({"embeddings": {"a.png": [0.1, 0.2]}}))

    listed = next(d for d in client.get("/api/datasets").get_json() if d["source"] == merged["source"])
    assert listed["has_embeddings"] is True
    assert client.get(f"/api/dataset/embeddings?source={merged['source']}").status_code == 200


def test_cloud_union_is_built_from_tile_parts_and_keeps_their_parent(app_client):
    client, app_module = app_client
    root = Path(app_module.DATA_ROOT)
    curation = app_module.dataset_curation
    full = "data_user/agristress500.csv"
    part1, part2 = curation.TILE_SOURCES
    _write_feature_csv(root / full, [{"image_path": "images/a.png", "width": "256", "score": "0"}])
    _write_feature_csv(root / part1, [{"image_path": "images/a.png", "width": "256", "score": "0"}])
    _write_feature_csv(root / part2, [{"image_path": "images/b.png", "width": "256", "score": "5"}])
    app_module.db.create_dataset_record("Full", "", full, None, None, 1)
    app_module.db.create_dataset_record("Part 1", "", part1, full, None, 1)
    app_module.db.create_dataset_record("Part 2", "", part2, full, None, 1)

    by_source = {dataset["source"]: dataset for dataset in client.get("/api/datasets").get_json()}

    assert by_source[curation.CLOUD_SOURCE]["row_count"] == 2
    assert by_source[part1]["parent_sources"] == [full, curation.CLOUD_SOURCE]
    assert by_source[part2]["parent_sources"] == [full, curation.CLOUD_SOURCE]
