# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the feature extractor. Synthetic in-memory COCO + PNG only —
no disk datalake, no network, no torch/pyiqa (NIMA is skipped).

Run:  .venv-magnidata/bin/python -m pytest precisionai/magnidata/tools/test_features.py -q
"""

import csv
import io
import json
import logging
import os

import numpy as np
import pytest
from PIL import Image

from . import coco_labels, image_source, mask_labels, pixel_features, quality_features
from .features import extract, process_row, run
from .metadata_join import camera_angle_category
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
            "annotations": [
                {"id": i + 1, "image_id": 1, "category_id": c, "segmentation": [seg]}
                for i, (c, seg) in enumerate(annotations)
            ],
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
    assert row["annotation_ratio"] == pytest.approx(0.5, abs=0.03)
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
    rgb = np.full((100, 100, 3), 120, np.uint8)  # uniform colour
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
    assert process_row(img) == process_row(img)  # (row, err) tuple equality


def test_missing_label_continues(tmp_path):
    rgb = np.full((40, 40, 3), 200, np.uint8)
    img = make_dataset(tmp_path, "s", rgb, [], CATS, write_label=False)
    row, err = process_row(img)  # must not raise
    assert "label_not_found" in err  # error reported out-of-band
    assert row["annotation_ratio"] == ""  # COCO column blank
    assert row["width"] == 40  # RGB-only features still set


def test_metadata_join_and_angle(tmp_path):
    rgb = np.zeros((40, 40, 3), np.uint8)
    meta = {
        "id": "s",
        "gsd": "0.0151",
        "angle": "-1.0",
        "crop_name": "canola",
        "camera_view": "oblique",
        "domain_metric": "high",
    }
    img = make_dataset(tmp_path, "s", rgb, [(1, _rect(0, 0, 20, 40))], CATS, metadata=meta)
    row = pr(img)
    assert row["gsd"] == pytest.approx(0.0151)
    assert row["camera_angle"] == "oriented"  # camera_view=oblique -> oriented
    assert row["domain_metric"] == "high"
    assert "crop_stage" not in row  # dropped from the schema


def test_camera_angle_categories():
    assert camera_angle_category("", "nadir") == "nadir"
    assert camera_angle_category("60.0", "oblique") == "oriented"
    assert camera_angle_category("", "none") == "missing"
    assert camera_angle_category("3.0", "") == "nadir"  # numeric fallback, near 0
    assert camera_angle_category("45.0", "") == "oriented"


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
    vivid[:, :16, 0] = 255  # red | blue split -> highly colorful
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
    for col in ("noise_sigma", "tenengrad", "rms_contrast", "dynamic_range", "luma_entropy", "colorfulness"):
        assert isinstance(row[col], float)


def test_image_source_resolution(tmp_path, monkeypatch):
    dl = tmp_path / "dl"
    th = tmp_path / "th720"
    rel = "/ds/images/x"
    (dl / "ds/images").mkdir(parents=True)
    (th / "ds/images").mkdir(parents=True)
    (dl / "ds/images/x.png").write_bytes(b"full")
    (th / "ds/images/x.jpg").write_bytes(b"thumb")  # thumbnail is .jpg
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
    make_dataset(tmp_path, "s", rgb, [(1, _rect(0, 0, 50, 100))], CATS)
    labels = coco_labels.parse(str(tmp_path / "ds1" / "labels" / "s.json"))
    full = coco_labels.rasterize_class_map(labels)
    half = coco_labels.rasterize_class_map(labels, target_wh=(50, 50))
    assert full.shape == (100, 100)
    assert half.shape == (50, 50)
    fg_full = coco_labels.foreground_from_class_map(full).mean()
    fg_half = coco_labels.foreground_from_class_map(half).mean()
    assert fg_half == pytest.approx(fg_full, abs=0.03)  # coverage ratio preserved


def test_schema_complete(tmp_path):
    rgb = np.zeros((20, 20, 3), np.uint8)
    img = make_dataset(tmp_path, "s", rgb, [(1, _rect(0, 0, 10, 20))], CATS)
    row = pr(img)
    assert list(row.keys()) == OUTPUT_COLUMNS  # exact column set + order


def test_parallel_run_matches_serial_run(tmp_path):
    images = [
        make_dataset(tmp_path, f"s{i}", np.full((60, 60, 3), 40 * i, np.uint8), [(1, _rect(0, 0, 30, 60))], CATS)
        for i in range(5)
    ]
    source = tmp_path / "input.csv"
    source.write_text("image_path\n" + "\n".join(images) + "\n")
    serial, parallel = tmp_path / "serial.csv", tmp_path / "parallel.csv"

    run(str(source), str(serial), skip_nima=True, workers=1)
    run(str(source), str(parallel), skip_nima=True, workers=3)

    assert serial.read_text() == parallel.read_text()


def _write_coco(path, w, h, annotations):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    coco = {
        "images": [{"id": 1, "file_name": "x.png", "height": h, "width": w}],
        "annotations": [
            {"id": i + 1, "image_id": 1, "category_id": c, "segmentation": [seg]}
            for i, (c, seg) in enumerate(annotations)
        ],
        "categories": CATS,
    }
    with open(path, "w") as f:
        json.dump(coco, f)


def _save_png(path, rgb):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(rgb, "RGB").save(path)
    return path


def test_nested_images_find_mirrored_label(tmp_path):
    ds = tmp_path / "ds1"
    img = _save_png(str(ds / "images" / "siteA" / "row2" / "s.png"), np.zeros((40, 40, 3), np.uint8))
    _write_coco(str(ds / "labels" / "siteA" / "row2" / "s.json"), 40, 40, [(1, _rect(0, 0, 20, 40))])
    row, err = process_row(img)
    assert err == ""
    assert row["annotation_ratio"] == pytest.approx(0.5, abs=0.05)
    assert row["instance_count"] == 1


def test_nested_images_fall_back_to_flat_label_and_metadata(tmp_path):
    ds = tmp_path / "ds1"
    img = _save_png(str(ds / "images" / "siteA" / "s.png"), np.zeros((40, 40, 3), np.uint8))
    _write_coco(str(ds / "labels" / "s.json"), 40, 40, [(1, _rect(0, 0, 40, 10))])
    os.makedirs(ds / "metadata")
    (ds / "metadata" / "images_metadata.csv").write_text("id,gsd\ns,0.5\n")
    row, err = process_row(img)
    assert err == ""
    assert row["annotation_ratio"] == pytest.approx(0.25, abs=0.05)
    assert row["gsd"] == 0.5  # metadata joined from the dataset root, not images/


def test_nested_images_ignore_label_inside_images_tree(tmp_path):
    ds = tmp_path / "ds1"
    img = _save_png(str(ds / "images" / "siteA" / "s.png"), np.zeros((40, 40, 3), np.uint8))
    _write_coco(str(ds / "images" / "labels" / "s.json"), 40, 40, [(1, _rect(0, 0, 20, 40))])
    row, err = process_row(img)
    assert err == "coco:label_not_found"
    assert row["annotation_ratio"] == ""


def test_non_images_folder_keeps_grandparent_label(tmp_path):
    ds = tmp_path / "ds1"
    img = _save_png(str(ds / "raw" / "s.png"), np.zeros((40, 40, 3), np.uint8))
    _write_coco(str(ds / "labels" / "s.json"), 40, 40, [(1, _rect(0, 0, 20, 40))])
    row, err = process_row(img)
    assert err == ""
    assert row["annotation_ratio"] == pytest.approx(0.5, abs=0.05)


def test_images_folder_far_up_the_path_keeps_grandparent_label_and_metadata(tmp_path):
    ds = tmp_path / "images" / "ds1"
    img = _save_png(str(ds / "raw" / "s.png"), np.zeros((40, 40, 3), np.uint8))
    _write_coco(str(ds / "labels" / "s.json"), 40, 40, [(1, _rect(0, 0, 20, 40))])
    os.makedirs(ds / "metadata")
    (ds / "metadata" / "images_metadata.csv").write_text("id,gsd\ns,0.5\n")
    row, err = process_row(img)
    assert err == ""
    assert row["annotation_ratio"] == pytest.approx(0.5, abs=0.05)
    assert row["gsd"] == 0.5


def _two_class_mask(h, w):
    """Left half class A (green), top-right quarter class B (yellow), rest near-black."""
    mask = np.full((h, w, 3), 5, np.uint8)  # near-black, not pure black: still background
    mask[:, : w // 2] = (49, 140, 52)
    mask[: h // 2, w // 2 :] = (154, 178, 54)
    return mask


def test_mask_only_dataset_gets_coverage_and_class_columns(tmp_path):
    ds = tmp_path / "ds1"
    img = _save_png(str(ds / "images" / "s.png"), np.full((40, 40, 3), 120, np.uint8))
    _save_png(str(ds / "masks" / "s.png"), _two_class_mask(40, 40))
    row, err = process_row(img)
    assert err == ""
    assert row["annotation_ratio"] == pytest.approx(0.75)
    assert row["annotated_px_count"] == 1200
    assert row["bg_coverage"] == pytest.approx(0.25)
    assert row["fg_green_mean"] == pytest.approx(120.0)
    assert row["class_count"] == 2
    assert row["smallest_class_ratio"] == pytest.approx(1 / 3, abs=1e-5)
    assert row["class_entropy"] == pytest.approx(0.918296, abs=1e-4)
    assert row["mean_pairwise_color_dist"] == 0.0  # both classes cover the same grey pixels
    assert row["interclass_color_sim"] == 1.0
    assert row["instance_count"] == ""  # a class-coloured mask has no instances


def test_mask_in_nested_folder_is_resized_to_the_image(tmp_path):
    ds = tmp_path / "ds1"
    img = _save_png(str(ds / "images" / "siteA" / "s.png"), np.zeros((20, 20, 3), np.uint8))
    _save_png(str(ds / "masks" / "siteA" / "s.png"), _two_class_mask(80, 80))  # e.g. image read as a thumbnail
    row, err = process_row(img)
    assert err == ""
    assert row["annotated_px_count"] == 300
    assert row["annotation_ratio"] == pytest.approx(0.75)
    assert row["class_count"] == 2


def test_coco_label_wins_over_mask(tmp_path):
    img = make_dataset(tmp_path, "s", np.zeros((40, 40, 3), np.uint8), [(1, _rect(0, 0, 20, 40))], CATS)
    _save_png(str(tmp_path / "ds1" / "masks" / "s.png"), _two_class_mask(40, 40))
    row = pr(img)
    assert row["annotation_ratio"] == pytest.approx(0.5, abs=0.05)
    assert row["class_count"] == 1
    assert row["class_entropy"] == 0.0  # a single class has no mix, never a negative entropy
    assert row["instance_count"] == 1


def test_unreadable_mask_is_reported(tmp_path):
    ds = tmp_path / "ds1"
    img = _save_png(str(ds / "images" / "s.png"), np.zeros((20, 20, 3), np.uint8))
    os.makedirs(ds / "masks")
    (ds / "masks" / "s.png").write_bytes(b"not a png")
    row, err = process_row(img)
    assert err.startswith("mask:")
    assert row["annotation_ratio"] == ""
    assert row["width"] == 20


def test_mask_class_map_is_deterministic(tmp_path):
    path = _save_png(str(tmp_path / "m.png"), _two_class_mask(10, 10))
    class_map = mask_labels.load_class_map(path)
    bg = coco_labels.BACKGROUND_CATEGORY_ID + coco_labels.CLASS_MAP_OFFSET
    assert class_map[9, 9] == bg
    assert class_map[0, 0] == bg + 1  # (49, 140, 52) packs below (154, 178, 54)
    assert class_map[0, 9] == bg + 2
    assert mask_labels.load_class_map(path, target_wh=(5, 5)).shape == (5, 5)


def _save_grey(path, values):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(values).save(path)
    return path


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_single_channel_mask_is_read_as_class_ids(tmp_path, dtype):
    ids = np.zeros((40, 40), dtype)
    ids[:, 10:20], ids[:, 20:30], ids[:, 30:] = 1, 2, 3
    ds = tmp_path / "ds1"
    img = _save_png(str(ds / "images" / "s.png"), np.full((40, 40, 3), 120, np.uint8))
    _save_grey(str(ds / "masks" / "s.png"), ids)
    row, err = process_row(img)
    assert err == ""
    assert row["annotation_ratio"] == pytest.approx(0.75)
    assert row["class_count"] == 3


def test_mask_with_too_many_colours_is_reported(tmp_path):
    noisy = np.zeros((40, 40, 3), np.uint8)
    noisy[..., 0] = 100 + np.arange(40, dtype=np.uint8)  # 40 distinct colours, e.g. anti-aliased edges
    ds = tmp_path / "ds1"
    img = _save_png(str(ds / "images" / "s.png"), np.zeros((40, 40, 3), np.uint8))
    _save_png(str(ds / "masks" / "s.png"), noisy)
    row, err = process_row(img)
    assert err.startswith("mask:too many mask colours: 41 classes")
    assert row["class_count"] == ""
    assert row["annotation_ratio"] == ""


def _provenance_of(tmp_path, image_mode):
    img = make_dataset(tmp_path, "s", np.zeros((20, 20, 3), np.uint8), [(1, _rect(0, 0, 10, 20))], CATS)
    source = tmp_path / "input.csv"
    source.write_text(f"image_path\n{img}\n")
    out = tmp_path / "out.csv"
    run(str(source), str(out), skip_nima=True, image_mode=image_mode)
    with open(str(out) + ".provenance.json") as f:
        return json.load(f)


def test_provenance_records_full_resolution_source(tmp_path):
    prov = _provenance_of(tmp_path, "fullres")
    assert prov["params"]["image_sources"] == {"fullres": 1}
    assert prov["determinism_notes"][0] == "Features computed on full-resolution images (no resize/thumbnail)."


def test_provenance_records_thumbnail_source(tmp_path, monkeypatch):
    thumbs = tmp_path / "th720"
    monkeypatch.setattr(image_source, "DATALAKE_ROOT", str(tmp_path))
    monkeypatch.setattr(image_source, "THUMB_ROOTS", [("thumb720", str(thumbs))])
    _save_png(str(thumbs / "ds1" / "images" / "s.png"), np.zeros((10, 10, 3), np.uint8))
    prov = _provenance_of(tmp_path, "thumbnail")
    assert prov["params"]["image_sources"] == {"thumb720": 1}
    note = prov["determinism_notes"][0]
    assert "1 720 px thumbnail" in note
    assert "full-resolution images (no resize" not in " ".join(prov["determinism_notes"])


def _broken_inputs(tmp_path):
    """Return an input CSV listing one good image (no label), a text file named .png, and a truncated JPEG."""
    good = make_dataset(tmp_path, "good", np.full((24, 32, 3), 120, np.uint8), [], CATS, write_label=False)
    images_dir = os.path.dirname(good)
    fake = os.path.join(images_dir, "notes.png")
    with open(fake, "w") as f:
        f.write("this is not an image")
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(buf, "JPEG")
    truncated = os.path.join(images_dir, "cut.jpg")
    with open(truncated, "wb") as f:
        f.write(buf.getvalue()[: len(buf.getvalue()) // 2])
    source = tmp_path / "in.csv"
    with open(source, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path"])
        writer.writeheader()
        writer.writerows({"image_path": p} for p in (good, fake, truncated))
    return source, good, fake, truncated


def test_extract_drops_and_reports_unreadable_images(tmp_path):
    source, good, fake, truncated = _broken_inputs(tmp_path)
    out = tmp_path / "out.csv"

    report = extract(str(source), str(out), skip_nima=True, drop_unreadable=True)

    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["image_path"] for r in rows] == [good]
    assert report.rows == 1
    assert report.unreadable[0] == (fake, "not a recognised image format")
    assert report.unreadable[1][0] == truncated
    assert "truncated" in report.unreadable[1][1]
    assert report.errors == []
    assert report.missing_labels == 1


def test_run_keeps_unreadable_rows_by_default(tmp_path):
    source, *_ = _broken_inputs(tmp_path)
    out = tmp_path / "out.csv"

    assert run(str(source), str(out), skip_nima=True) == 3


def test_missing_labels_are_logged_as_information_not_errors(tmp_path, caplog):
    rgb = np.full((40, 40, 3), 200, np.uint8)
    img = make_dataset(tmp_path, "s", rgb, [], CATS, write_label=False)
    source = tmp_path / "in.csv"
    with open(source, "w", newline="") as f:
        f.write(f"image_path\n{img}\n")

    with caplog.at_level(logging.INFO, logger="precisionai.magnidata.tools.features"):
        report = extract(str(source), str(tmp_path / "out.csv"), skip_nima=True)

    assert report.missing_labels == 1
    assert report.errors == []
    assert [r.levelno for r in caplog.records] == [logging.INFO]
    assert "no label file" in caplog.records[0].getMessage()


def test_unreadable_images_are_logged_as_warnings(tmp_path, caplog):
    source, *_ = _broken_inputs(tmp_path)

    with caplog.at_level(logging.INFO, logger="precisionai.magnidata.tools.features"):
        extract(str(source), str(tmp_path / "out.csv"), skip_nima=True)

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings[0] == "2/3 images could not be read:"
