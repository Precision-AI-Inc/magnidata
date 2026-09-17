# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""HTTP-layer tests for GET /api/datasets/demos, the dataset-build endpoints, and the
extended remove_dataset cleanup. dataset_builder's own orchestration logic is
unit-tested in test_dataset_builder.py — these tests only check request parsing,
status codes, and job_id plumbing, so start_upload_build/start_demo_build are
monkeypatched.

Run:  python -m pytest precisionai/magnidata/api/test_app.py -q
"""

import importlib
import io
import json
import os
import threading
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
