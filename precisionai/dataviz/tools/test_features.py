# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the feature extractor. Synthetic in-memory COCO + PNG only —
no disk datalake, no network, no torch/pyiqa (NIMA is skipped).

Run:  .venv-agriviz/bin/python -m pytest precisionai/agriviz/tools/test_features.py -q
"""
import csv
import json
import os

import numpy as np
import pytest
from PIL import Image

from . import coco_labels, image_source, pixel_features, quality_features
from .features import process_row
from .schema import OUTPUT_COLUMNS


def _rect(x0, y0, x1, y1):
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def make_dataset(root, stem, rgb, annotations, categories, metadata=None, write_label=True):
    """Lay out <root>/<ds>/{images,labels,metadata} and return the image path."""
    ds = os.path.join(str(root), "ds1")
    for sub in ("images", "labels", "metadata"):
        os.makedirs(os.path.join(ds, sub), exist_ok=True)
    img_path = os.path.join(ds, "images", stem + ".png")
    Image.fromarray(rgb, "RGB").save(img_path)
    h, w = rgb.shape[:2]
    if write_label:
        coco = {
            "images": [{"id": 1, "file_name": stem + ".png", "height": h, "width": w}],
            "annotations": [{"id": i + 1, "image_id": 1, "category_id": c, "segmentation": [seg]}
                            for i, (c, seg) in enumerate(annotations)],
            "categories": categories,
        }
        with open(os.path.join(ds, "labels", stem + ".json"), "w") as f:
            json.dump(coco, f)
    if metadata is not None:
        mpath = os.path.join(ds, "metadata", "images_metadata.csv")
        with open(mpath, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(metadata.keys()))
            wr.writeheader()
            wr.writerow(metadata)
    return img_path


CATS = [{"id": 0, "name": "background"}, {"id": 1, "name": "cropA"}, {"id": 2, "name": "cropB"}]


def pr(*a, **k):
    """process_row returning just the row dict (drops the error string)."""
    row, _err = process_row(*a, **k)
    return row


def test_coverage_half(tmp_path):
    rgb = np.zeros((100, 100, 3), np.uint8)
    img = make_dataset(tmp_path, "s", rgb, [(1, _rect(0, 0, 50, 100))], CATS)
    row = pr(img)
    assert row["green_annotation_ratio"] == pytest.approx(0.5, abs=0.03)
    assert row["annotated_px_count"] > 0


def test_three_disjoint_instances(tmp_path):
    rgb = np.zeros((100, 100, 3), np.uint8)
    anns = [(1, _rect(0, 0, 10, 10)), (1, _rect(40, 40, 60, 60)), (1, _rect(80, 80, 95, 95))]
    img = make_dataset(tmp_path, "s", rgb, anns, CATS)
    row = pr(img)
    assert row["instance_count"] == 3


def test_overlap_ratio(tmp_path):
    rgb = np.zeros((100, 100, 3), np.uint8)
    overlapping = [(1, _rect(0, 0, 60, 60)), (2, _rect(40, 40, 100, 100))]
    img = make_dataset(tmp_path, "s", rgb, overlapping, CATS)
    assert pr(img)["overlap_ratio"] > 0

    disjoint = [(1, _rect(0, 0, 20, 20)), (2, _rect(70, 70, 90, 90))]
    img2 = make_dataset(tmp_path / "b", "s", rgb, disjoint, CATS)
    assert pr(img2)["overlap_ratio"] == 0


def test_interclass_color_sim_identical(tmp_path):
    rgb = np.full((100, 100, 3), 120, np.uint8)            # uniform colour
    anns = [(1, _rect(0, 0, 50, 100)), (2, _rect(50, 0, 100, 100))]
    img = make_dataset(tmp_path, "s", rgb, anns, CATS)
    row = pr(img)
    assert row["class_count"] == 2
    assert row["interclass_color_sim"] == pytest.approx(1.0, abs=0.02)


def test_overexpose(tmp_path):
    rgb = np.full((40, 40, 3), 255, np.uint8)
    img = make_dataset(tmp_path, "s", rgb, [(1, _rect(0, 0, 40, 40))], CATS)
    assert pr(img)["overexpose_ratio"] == 1.0


def test_blur_sharp_vs_flat():
    flat = np.full((64, 64, 3), 128, np.uint8)
    cb = np.indices((64, 64)).sum(axis=0) % 2
    checker = (cb[:, :, None] * 255).astype(np.uint8).repeat(3, axis=2)
    sharp = pixel_features.focus(pixel_features.to_luma(checker))["blur_laplacian"]
    dull = pixel_features.focus(pixel_features.to_luma(flat))["blur_laplacian"]
    assert sharp > dull
    assert dull == pytest.approx(0.0, abs=1e-9)


def test_white_balance_neutral():
    gray = np.full((32, 32, 3), 100, np.uint8)
    wb = pixel_features.white_balance(gray)
    assert wb["wb_r_gain"] == pytest.approx(1.0, abs=1e-3)
    assert wb["wb_b_gain"] == pytest.approx(1.0, abs=1e-3)
    assert wb["wb_rb_ratio"] == pytest.approx(1.0, abs=1e-3)


def test_determinism(tmp_path):
    rng = np.linspace(0, 255, 100 * 100 * 3).reshape(100, 100, 3).astype(np.uint8)
    img = make_dataset(tmp_path, "s", rng, [(1, _rect(10, 10, 80, 80))], CATS)
    assert process_row(img) == process_row(img)             # (row, err) tuple equality


def test_missing_label_continues(tmp_path):
    rgb = np.full((40, 40, 3), 200, np.uint8)
    img = make_dataset(tmp_path, "s", rgb, [], CATS, write_label=False)
    row, err = process_row(img)                             # must not raise
    assert "label_not_found" in err                         # error reported out-of-band
    assert row["green_annotation_ratio"] == ""              # COCO column blank
    assert row["width"] == 40                               # RGB-only features still set


def test_metadata_join_and_angle(tmp_path):
    rgb = np.zeros((40, 40, 3), np.uint8)
    meta = {"id": "s", "gsd": "0.0151", "angle": "-1.0",
            "crop_name": "canola", "camera_view": "oblique", "weed_density": "high"}
    img = make_dataset(tmp_path, "s", rgb, [(1, _rect(0, 0, 20, 40))], CATS, metadata=meta)
    row = pr(img)
    assert row["gsd"] == pytest.approx(0.0151)
    assert row["camera_angle"] == "oriented"                # camera_view=oblique -> oriented
    assert row["weed_density"] == "high"
    assert "crop_stage" not in row                           # dropped from the schema


def test_camera_angle_categories():
    from .metadata_join import camera_angle_category as cat
    assert cat("", "nadir") == "nadir"
    assert cat("60.0", "oblique") == "oriented"
    assert cat("", "none") == "missing"
    assert cat("3.0", "") == "nadir"                        # numeric fallback, near 0
    assert cat("45.0", "") == "oriented"


def test_noise_sigma_clean_vs_noisy():
    rng = np.random.default_rng(0)
    flat = np.full((128, 128), 128.0)
    noisy = np.clip(flat + rng.normal(0, 20, flat.shape), 0, 255)
    assert quality_features.noise_sigma(flat) == pytest.approx(0.0, abs=1e-6)
    # estimator should recover roughly the injected sigma (~20) on a flat field
    assert quality_features.noise_sigma(noisy) == pytest.approx(20.0, rel=0.25)


def test_colorfulness_gray_vs_vivid():
    gray = np.full((32, 32, 3), 128, np.uint8)
    vivid = np.zeros((32, 32, 3), np.uint8)
    vivid[:, :16, 0] = 255          # red | blue split -> highly colorful
    vivid[:, 16:, 2] = 255
    assert quality_features.colorfulness(gray) == pytest.approx(0.0, abs=1e-6)
    assert quality_features.colorfulness(vivid) > 50.0


def test_contrast_and_dynamic_range():
    flat = np.full((32, 32), 100.0)
    assert quality_features.rms_contrast(flat) == pytest.approx(0.0, abs=1e-9)
    assert quality_features.dynamic_range(flat) == pytest.approx(0.0, abs=1e-9)
    ramp = np.tile(np.linspace(0, 255, 32), (32, 1))
    assert quality_features.rms_contrast(ramp) > 0.2
    assert quality_features.dynamic_range(ramp) > 0.9


def test_quality_columns_populated(tmp_path):
    rgb = (np.indices((40, 40)).sum(0) % 2 * 255).astype(np.uint8)[:, :, None].repeat(3, 2)
    img = make_dataset(tmp_path, "s", rgb, [(1, _rect(0, 0, 20, 40))], CATS)
    row = pr(img)
    for col in ("noise_sigma", "tenengrad", "rms_contrast", "dynamic_range",
                "luma_entropy", "colorfulness"):
        assert isinstance(row[col], float)


def test_image_source_resolution(tmp_path, monkeypatch):
    dl = tmp_path / "dl"
    th = tmp_path / "th720"
    rel = "/ds/images/x"
    (dl / "ds/images").mkdir(parents=True)
    (th / "ds/images").mkdir(parents=True)
    (dl / "ds/images/x.png").write_bytes(b"full")
    (th / "ds/images/x.jpg").write_bytes(b"thumb")     # thumbnail is .jpg
    monkeypatch.setattr(image_source, "DATALAKE_ROOT", str(dl))
    monkeypatch.setattr(image_source, "THUMB_ROOTS", [("thumb720", str(th))])
    full = str(dl) + rel + ".png"
    assert image_source.resolve(full, "auto") == (full, "fullres")
    assert image_source.resolve(full, "thumbnail") == (str(th) + rel + ".jpg", "thumb720")
    # full-res missing -> auto falls back to the .jpg thumbnail
    (dl / "ds/images/x.png").unlink()
    assert image_source.resolve(full, "auto") == (str(th) + rel + ".jpg", "thumb720")
    assert image_source.resolve(str(dl) + "/ds/images/none.png", "auto")[1] == "missing"


def test_coco_rasterize_scales_to_thumbnail(tmp_path):
    # full-res label is 100x100; coverage must survive rasterizing at half-res
    rgb = np.zeros((100, 100, 3), np.uint8)
    img = make_dataset(tmp_path, "s", rgb, [(1, _rect(0, 0, 50, 100))], CATS)
    labels = coco_labels.parse(str(tmp_path / "ds1" / "labels" / "s.json"))
    full = coco_labels.rasterize_class_map(labels)
    half = coco_labels.rasterize_class_map(labels, target_wh=(50, 50))
    assert full.shape == (100, 100) and half.shape == (50, 50)
    fg_full = coco_labels.foreground_from_class_map(full).mean()
    fg_half = coco_labels.foreground_from_class_map(half).mean()
    assert fg_half == pytest.approx(fg_full, abs=0.03)   # coverage ratio preserved


def test_schema_complete(tmp_path):
    rgb = np.zeros((20, 20, 3), np.uint8)
    img = make_dataset(tmp_path, "s", rgb, [(1, _rect(0, 0, 10, 20))], CATS)
    row = pr(img)
    assert list(row.keys()) == OUTPUT_COLUMNS                # exact column set + order
