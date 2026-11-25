# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the COCO128 download, staging + pipeline wiring. No internet access — the
download tests use a local HTTP server, coco_root is a tiny synthetic fixture, and
features/embeddings.run are monkeypatched to fast fakes so no torch/timm/pyiqa is required.

Run:  python -m pytest precisionai/magnidata/scripts/test_prepare_coco128_dashboard.py -q
"""

import argparse
import contextlib
import csv
import hashlib
import http.server
import json
import sys
import threading
import time
import types
import zipfile

import numpy as np
import pytest
from PIL import Image

from precisionai.magnidata.training.dataset_check import has_masks

from . import prepare_coco128_dashboard as mod

# A 0.4x0.4 (normalized) square polygon, YOLO-segment format: class_id x1 y1 x2 y2 ... xn yn.
SQUARE_POLYGON_LABEL = "0 0.3 0.3 0.7 0.3 0.7 0.7 0.3 0.7\n"


def _make_coco_root(root, n=2):
    coco_root = root / "coco128_src"
    images_dir = coco_root / "images" / "train2017"
    labels_dir = coco_root / "labels" / "train2017"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    for i in range(n):
        Image.fromarray(np.full((10, 10, 3), 100, np.uint8), "RGB").save(images_dir / f"img{i}.jpg")
        (labels_dir / f"img{i}.txt").write_text(SQUARE_POLYGON_LABEL)
    return coco_root


def _args(**overrides):
    defaults = dict(
        dataset_stem="coco128_test",
        limit=None,
        stage_mode="copy",
        cluster_by="dominant-class",
        device="cpu",
        with_nima=False,
        no_embeddings=False,
        embedding_model="fake",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ── YOLO-segment parsing + COCO polygon writing ──────────────────────────────


def test_read_yolo_seg_labels_parses_polygon_points(tmp_path):
    label_path = tmp_path / "a.txt"
    label_path.write_text(SQUARE_POLYGON_LABEL)

    instances = mod.read_yolo_seg_labels(label_path)

    assert len(instances) == 1
    assert instances[0].class_id == 0
    assert instances[0].points == ((0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7))


def test_read_yolo_seg_labels_skips_malformed_rows(tmp_path):
    label_path = tmp_path / "a.txt"
    # Too few points (a pair), an even-length row, and a valid triangle.
    label_path.write_text("0 0.1 0.1 0.2 0.2\n1 0.1 0.1 0.2 0.2 0.3\n2 0.1 0.1 0.5 0.1 0.3 0.5\n")

    instances = mod.read_yolo_seg_labels(label_path)

    assert len(instances) == 1
    assert instances[0].class_id == 2


def test_read_yolo_seg_labels_missing_file_returns_empty(tmp_path):
    assert mod.read_yolo_seg_labels(tmp_path / "missing.txt") == []


def test_write_coco_json_computes_real_polygon_bbox_and_area(tmp_path):
    out = tmp_path / "a.json"
    instance = mod.YoloInstance(class_id=0, points=((0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)))

    mod.write_coco_json(out, "a.jpg", width=100, height=100, instances=[instance])

    doc = json.loads(out.read_text())
    ann = doc["annotations"][0]
    assert ann["category_id"] == 1  # class 0 ("person") + 1 (background reserves 0)
    assert ann["segmentation"] == [[30.0, 30.0, 70.0, 30.0, 70.0, 70.0, 30.0, 70.0]]
    assert ann["bbox"] == [30.0, 30.0, 40.0, 40.0]
    assert ann["area"] == 1600.0  # 40x40 square


def test_write_coco_json_no_instances_writes_empty_annotations(tmp_path):
    out = tmp_path / "a.json"
    mod.write_coco_json(out, "a.jpg", width=100, height=100, instances=[])
    doc = json.loads(out.read_text())
    assert doc["annotations"] == []
    assert doc["images"] == [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}]


def test_dominant_class_picks_the_largest_polygon():
    small = mod.YoloInstance(class_id=0, points=((0.0, 0.0), (0.1, 0.0), (0.1, 0.1)))
    large = mod.YoloInstance(class_id=1, points=((0.0, 0.0), (0.9, 0.0), (0.9, 0.9), (0.0, 0.9)))

    assert mod.dominant_class([small, large]) == mod.COCO80[1]


def test_dominant_class_empty_is_background():
    assert mod.dominant_class([]) == "background"


def test_stage_coco128_default_root_unchanged(tmp_path):
    root = tmp_path
    coco_root = _make_coco_root(root)

    manifest, stats = mod.stage_coco128(_args(), root, coco_root)

    assert manifest == root / "image_sets" / "coco128_test" / "coco128_test_input.csv"
    assert stats["rows"] == 2


def test_stage_coco128_custom_stage_root(tmp_path):
    root = tmp_path
    coco_root = _make_coco_root(root)
    custom_root = root / "data_user" / "coco128_test_images"

    manifest, _stats = mod.stage_coco128(_args(), root, coco_root, stage_root=custom_root)

    assert manifest == custom_root / "coco128_test_input.csv"
    assert len(list((custom_root / "images").glob("*.jpg"))) == 2
    with open(manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["image_path"].startswith("data_user/coco128_test_images/")


def test_stage_coco128_writes_masks_the_trainer_can_find(tmp_path):
    root = tmp_path
    coco_root = _make_coco_root(root)
    custom_root = root / "data_user" / "coco128_test_images"

    manifest, stats = mod.stage_coco128(_args(), root, coco_root, stage_root=custom_root)

    assert stats["masks"] == 2
    assert has_masks(manifest, root)
    with Image.open(custom_root / "masks" / "img0.png") as im:
        mask = np.asarray(im.convert("RGB"))
    assert mask.shape == (10, 10, 3)
    assert mask[0, 0].tolist() == [0, 0, 0]
    assert mask[5, 5].tolist() == list(mod._class_colour(0))


def _write_labels(stage_labels, name, instances, size=100):
    mod.write_coco_json(stage_labels / f"{name}.json", f"{name}.jpg", size, size, instances)


def _box(class_id, x1, y1, x2, y2):
    return mod.YoloInstance(class_id, ((x1, y1), (x2, y1), (x2, y2), (x1, y2)))


def test_write_class_masks_folds_classes_beyond_the_limit_into_other(tmp_path):
    labels = tmp_path / "labels"
    big, medium, small = _box(0, 0.0, 0.0, 0.5, 0.5), _box(1, 0.6, 0.0, 0.9, 0.3), _box(2, 0.6, 0.6, 0.7, 0.7)
    _write_labels(labels, "a", [small, big, medium])

    written = mod.write_class_masks(labels, tmp_path / "masks", class_limit=1)

    assert written == 1
    with Image.open(tmp_path / "masks" / "a.png") as im:
        mask = np.asarray(im.convert("RGB"))
    assert mask[25, 25].tolist() == list(mod._class_colour(0))
    assert mask[15, 75].tolist() == list(mod.OTHER_MASK_COLOUR)
    assert mask[65, 65].tolist() == list(mod.OTHER_MASK_COLOUR)
    assert mask[95, 5].tolist() == [0, 0, 0]


def test_write_class_masks_draws_small_objects_over_large_ones(tmp_path):
    labels = tmp_path / "labels"
    _write_labels(labels, "a", [_box(0, 0.0, 0.0, 1.0, 1.0), _box(1, 0.4, 0.4, 0.6, 0.6)])

    mod.write_class_masks(labels, tmp_path / "masks")

    with Image.open(tmp_path / "masks" / "a.png") as im:
        mask = np.asarray(im.convert("RGB"))
    assert mask[50, 50].tolist() == list(mod._class_colour(1))
    assert mask[10, 10].tolist() == list(mod._class_colour(0))


def test_write_class_masks_without_instances_is_all_background(tmp_path):
    labels = tmp_path / "labels"
    _write_labels(labels, "empty", [])

    mod.write_class_masks(labels, tmp_path / "masks")

    with Image.open(tmp_path / "masks" / "empty.png") as im:
        assert not np.asarray(im.convert("RGB")).any()


def _fake_features_embeddings(monkeypatch):
    # run_pipeline() import-checks torch/timm itself before calling embeddings.run
    # (so a missing install fails with a clear message) — stub them in sys.modules
    # too, so that preflight check passes without a real torch/timm install.
    for module_name in ("torch", "timm"):
        monkeypatch.setitem(sys.modules, module_name, types.ModuleType(module_name))

    calls = {"features": [], "embeddings": []}

    def fake_features_run(input_csv, output_csv, **kwargs):
        with open(input_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        with open(output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["image_path"])
            w.writeheader()
            for r in rows:
                w.writerow({"image_path": r["image_path"]})
        calls["features"].append((input_csv, output_csv))
        return len(rows)

    def fake_embeddings_run(input_csv, output_json, **kwargs):
        with open(output_json, "w") as f:
            json.dump({"embeddings": {}}, f)
        calls["embeddings"].append((input_csv, output_json))
        return 0

    monkeypatch.setattr("precisionai.magnidata.tools.features.run", fake_features_run)
    monkeypatch.setattr("precisionai.magnidata.tools.embeddings.run", fake_embeddings_run)
    return calls


def test_run_pipeline_default_data_dir_unchanged(tmp_path, monkeypatch):
    root = tmp_path
    coco_root = _make_coco_root(root)
    manifest, _ = mod.stage_coco128(_args(), root, coco_root)
    calls = _fake_features_embeddings(monkeypatch)
    monkeypatch.chdir(root)

    feature_csv, embeddings_json = mod.run_pipeline(_args(), root, manifest)

    assert feature_csv == root / "data" / "coco128_test.csv"
    assert embeddings_json == root / "data" / "coco128_test.json"
    assert feature_csv.is_file()
    assert embeddings_json is not None
    assert embeddings_json.is_file()
    assert len(calls["features"]) == 1


def test_run_pipeline_custom_data_dir(tmp_path, monkeypatch):
    root = tmp_path
    coco_root = _make_coco_root(root)
    manifest, _ = mod.stage_coco128(_args(), root, coco_root)
    _fake_features_embeddings(monkeypatch)
    monkeypatch.chdir(root)
    custom_data_dir = root / "data_user"
    custom_data_dir.mkdir()

    feature_csv, embeddings_json = mod.run_pipeline(_args(), root, manifest, data_dir=custom_data_dir)

    assert feature_csv == custom_data_dir / "coco128_test.csv"
    assert embeddings_json == custom_data_dir / "coco128_test.json"


# ── Download + extraction ─────────────────────────────────────────────────────


def _make_zip(path):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("coco128/images/train2017/img0.jpg", b"jpeg bytes")
        zf.writestr("coco128/labels/train2017/img0.txt", SQUARE_POLYGON_LABEL)
    return path.read_bytes()


@contextlib.contextmanager
def _serve(body, declared_length=None, stall=0.0):
    """Serve ``body`` at any path on localhost; yield the base URL and a request counter."""
    hits = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", str(declared_length if declared_length is not None else len(body)))
            self.end_headers()
            if stall:
                time.sleep(stall)
            with contextlib.suppress(OSError):
                self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/coco128-seg.zip", hits
    finally:
        server.shutdown()
        server.server_close()


def test_download_replaces_a_truncated_cached_zip(tmp_path):
    body = _make_zip(tmp_path / "good.zip")
    sha = hashlib.sha256(body).hexdigest()
    dest = tmp_path / "cache" / "coco128-seg.zip"
    dest.parent.mkdir()
    dest.write_bytes(body[: len(body) // 2])
    with _serve(body) as (url, hits):
        assert mod.download(url, dest, sha256=sha) is True
    assert dest.read_bytes() == body
    assert len(hits) == 1
    assert not dest.with_name(dest.name + ".part").exists()


def test_download_without_checksum_still_rejects_a_truncated_cached_zip(tmp_path):
    body = _make_zip(tmp_path / "good.zip")
    dest = tmp_path / "coco128-seg.zip"
    dest.write_bytes(body[:-10])
    with _serve(body) as (url, hits):
        assert mod.download(url, dest) is True
    assert dest.read_bytes() == body
    assert len(hits) == 1


def test_download_reuses_an_intact_cached_zip_without_a_request(tmp_path):
    body = _make_zip(tmp_path / "good.zip")
    dest = tmp_path / "coco128-seg.zip"
    dest.write_bytes(body)
    with _serve(body) as (url, hits):
        assert mod.download(url, dest, sha256=hashlib.sha256(body).hexdigest()) is False
    assert hits == []


def test_download_force_fetches_even_when_cached(tmp_path):
    body = _make_zip(tmp_path / "good.zip")
    dest = tmp_path / "coco128-seg.zip"
    dest.write_bytes(body)
    with _serve(body) as (url, hits):
        assert mod.download(url, dest, force=True) is True
    assert len(hits) == 1


def test_download_with_wrong_checksum_keeps_nothing(tmp_path):
    body = _make_zip(tmp_path / "good.zip")
    dest = tmp_path / "coco128-seg.zip"
    with _serve(body) as (url, _hits), pytest.raises(OSError, match="does not match the expected SHA-256"):
        mod.download(url, dest, sha256="0" * 64)
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()


def test_download_that_ends_early_keeps_nothing(tmp_path):
    body = _make_zip(tmp_path / "good.zip")
    dest = tmp_path / "coco128-seg.zip"
    with _serve(body, declared_length=len(body) + 100) as (url, _hits), pytest.raises(OSError, match="stopped after"):
        mod.download(url, dest)
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()


def test_download_that_stalls_times_out(tmp_path):
    body = _make_zip(tmp_path / "good.zip")
    dest = tmp_path / "coco128-seg.zip"
    with _serve(body, stall=2.0) as (url, _hits), pytest.raises(OSError, match="timed out"):
        mod.download(url, dest, timeout=0.3)
    assert not dest.exists()


def test_known_sha256_only_applies_to_the_default_url():
    assert mod.known_sha256(mod.COCO128_URL) == mod.COCO128_SHA256
    assert mod.known_sha256("https://example.com/other.zip") is None


def test_extract_ignores_an_interrupted_extraction(tmp_path):
    zip_path = tmp_path / "coco128-seg.zip"
    _make_zip(zip_path)
    extract_dir = tmp_path / "extracted"
    leftover = tmp_path / "extracted.partial" / "coco128" / "images" / "train2017"
    leftover.mkdir(parents=True)
    coco_root = mod.extract(zip_path, extract_dir)
    assert coco_root == extract_dir / "coco128"
    assert (coco_root / "labels" / "train2017" / "img0.txt").is_file()
    assert not (tmp_path / "extracted.partial").exists()


def test_extract_reuses_a_complete_extraction_unless_forced(tmp_path):
    zip_path = tmp_path / "coco128-seg.zip"
    _make_zip(zip_path)
    extract_dir = tmp_path / "extracted"
    coco_root = mod.extract(zip_path, extract_dir)
    marker = coco_root / "marker"
    marker.write_text("kept")
    assert mod.extract(zip_path, extract_dir) == coco_root
    assert marker.exists()
    assert mod.extract(zip_path, extract_dir, force=True) == coco_root
    assert not marker.exists()


def test_extract_without_a_coco_tree_leaves_no_folder(tmp_path):
    zip_path = tmp_path / "other.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("readme.txt", "nothing here")
    with pytest.raises(FileNotFoundError, match="could not find extracted coco128"):
        mod.extract(zip_path, tmp_path / "extracted")
    assert not (tmp_path / "extracted").exists()
    assert not (tmp_path / "extracted.partial").exists()
