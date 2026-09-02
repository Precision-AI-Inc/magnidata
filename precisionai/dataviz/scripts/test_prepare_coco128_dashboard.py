# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the COCO128 staging + pipeline wiring. No network — coco_root is a
tiny synthetic fixture, and features/embeddings.run are monkeypatched to fast fakes so
no torch/timm/pyiqa is required.

Run:  python -m pytest precisionai/agriviz/scripts/test_prepare_coco128_dashboard.py -q
"""
import argparse
import csv
import json

import numpy as np
from PIL import Image

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
        dataset_stem="coco128_test", limit=None, stage_mode="copy", cluster_by="dominant-class",
        device="cpu", with_nima=False, no_embeddings=False, embedding_model="fake",
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
    assert ann["category_id"] == 1   # class 0 ("person") + 1 (background reserves 0)
    assert ann["segmentation"] == [[30.0, 30.0, 70.0, 30.0, 70.0, 70.0, 30.0, 70.0]]
    assert ann["bbox"] == [30.0, 30.0, 40.0, 40.0]
    assert ann["area"] == 1600.0   # 40x40 square


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

    manifest, stats = mod.stage_coco128(_args(), root, coco_root, stage_root=custom_root)

    assert manifest == custom_root / "coco128_test_input.csv"
    assert len(list((custom_root / "images").glob("*.jpg"))) == 2
    with open(manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["image_path"].startswith("data_user/coco128_test_images/")


def _fake_features_embeddings(monkeypatch):
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

    monkeypatch.setattr("precisionai.agriviz.tools.features.run", fake_features_run)
    monkeypatch.setattr("precisionai.agriviz.tools.embeddings.run", fake_embeddings_run)
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
    assert feature_csv.is_file() and embeddings_json.is_file()
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
