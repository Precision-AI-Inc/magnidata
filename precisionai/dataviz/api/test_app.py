# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""HTTP-layer tests for GET /api/datasets/demos, the dataset-build endpoints, and the
extended remove_dataset cleanup. dataset_builder's own orchestration logic is
unit-tested in test_dataset_builder.py — these tests only check request parsing,
status codes, and job_id plumbing, so start_upload_build/start_demo_build are
monkeypatched.

Run:  python -m pytest precisionai/dataviz/api/test_app.py -q
"""

import importlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("CONFIG_PATH", str(tmp_path / "confi.yaml"))
    (tmp_path / "confi.yaml").write_text("datasets: []\n")
    os.makedirs(tmp_path / "data_user", exist_ok=True)

    app_module = importlib.import_module("app")

    importlib.reload(app_module)
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as client:
        yield client, app_module


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
    assert ann.data == b'{"a": 1}'


def test_build_upload_passes_embeddings_file_bytes(app_client, monkeypatch):
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

    assert captured["embeddings_file"] == b'{"embeddings": {}}'


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
