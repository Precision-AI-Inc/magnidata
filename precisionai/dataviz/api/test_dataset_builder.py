# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the dataset-build orchestrator. Network-free: the feature tool and the
COCO128 script's download/stage/pipeline calls are all monkeypatched to fast fakes. This
pipeline never computes embeddings itself (no model/torch surface to fake here at all) —
COCO128's embeddings come from a shipped precomputed asset (also faked, see
_patch_fake_coco128_pipeline) and AgriStress-500's from the CDN, bring-your-own.

Run:  python -m pytest precisionai/agriviz/api/test_dataset_builder.py -q
(from precisionai/agriviz/api/, or from the repo root — dataset_builder.py resolves the
precisionai package path from its own file location, not from cwd/DATA_ROOT)
"""
import csv
import io
import json
import os
import time

import dataset_builder
import db
import numpy as np
import pytest
from PIL import Image

UploadedImage = dataset_builder.staging_mod.UploadedImage

FAKE_EMBEDDINGS = json.dumps({"embeddings": {"a.png": [0.1, 0.2]}}).encode()


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    db.init_db()
    monkeypatch.setattr(dataset_builder, "DATA_ROOT", str(tmp_path))
    os.makedirs(tmp_path / "data_user", exist_ok=True)
    monkeypatch.chdir(tmp_path)
    dataset_builder._jobs.clear()
    if dataset_builder._build_lock.locked():
        dataset_builder._build_lock.release()
    yield


def _fake_features_run(input_csv, output_csv, **kwargs):
    with open(input_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path"])
        w.writeheader()
        for r in rows:
            w.writerow({"image_path": r["image_path"]})


def _wait_for_job(job_id, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = dataset_builder.get_job(job_id)
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.02)
    raise TimeoutError(f"job {job_id} did not finish: {dataset_builder.get_job(job_id)}")


# ── Upload build: validation ─────────────────────────────────────────────────

def test_start_upload_build_rejects_empty_name():
    with pytest.raises(dataset_builder.BuildValidationError):
        dataset_builder.start_upload_build([UploadedImage("a.png", b"x")], [], None, "", "")


def test_start_upload_build_rejects_no_images():
    with pytest.raises(dataset_builder.BuildValidationError):
        dataset_builder.start_upload_build([], [], None, "Name", "")


def test_start_upload_build_rejects_too_many_images():
    images = [UploadedImage(f"{i}.png", b"x") for i in range(dataset_builder.MAX_IMAGES + 1)]
    with pytest.raises(dataset_builder.BuildValidationError):
        dataset_builder.start_upload_build(images, [], None, "Name", "")


def test_start_upload_build_rejects_oversized_upload():
    big = b"x" * (dataset_builder.MAX_TOTAL_BYTES + 1)
    with pytest.raises(dataset_builder.BuildValidationError):
        dataset_builder.start_upload_build([UploadedImage("a.png", big)], [], None, "Name", "")


def test_start_upload_build_rejects_oversized_when_annotations_push_over_cap():
    images = [UploadedImage("a.png", b"x" * (dataset_builder.MAX_TOTAL_BYTES - 10))]
    annotations = [UploadedImage("a.json", b"x" * 20)]
    with pytest.raises(dataset_builder.BuildValidationError):
        dataset_builder.start_upload_build(images, annotations, None, "Name", "")


def test_start_upload_build_rejects_malformed_embeddings_json():
    with pytest.raises(dataset_builder.BuildValidationError):
        dataset_builder.start_upload_build([UploadedImage("a.png", b"x")], [], b"not json",
                                            "Name", "")


def test_start_upload_build_rejects_embeddings_with_wrong_shape():
    bad = json.dumps({"vectors": {}}).encode()   # missing the "embeddings" key
    with pytest.raises(dataset_builder.BuildValidationError):
        dataset_builder.start_upload_build([UploadedImage("a.png", b"x")], [], bad, "Name", "")


def test_start_upload_build_rejects_oversized_embeddings_file():
    big = b'{"embeddings": {}}' + b" " * dataset_builder.MAX_EMBEDDINGS_BYTES
    with pytest.raises(dataset_builder.BuildValidationError):
        dataset_builder.start_upload_build([UploadedImage("a.png", b"x")], [], big, "Name", "")


# ── Upload build: end to end ──────────────────────────────────────────────────

def test_upload_build_end_to_end_without_embeddings(monkeypatch):
    monkeypatch.setattr(dataset_builder.features_mod, "run", _fake_features_run)
    images = [UploadedImage("weeds/a.png", b"AAAA"), UploadedImage("soil/b.png", b"BBBB")]

    job_id = dataset_builder.start_upload_build(images, [], None, "My Set", "a test set")
    job = _wait_for_job(job_id)

    assert job["status"] == "done"
    assert job["percent"] == 100
    assert job["dataset"]["name"] == "My Set"
    assert job["dataset"]["has_embeddings"] is False
    rec = db.get_created_dataset_by_source(job["dataset"]["source"])
    assert rec is not None
    assert rec["parent_source"] is None
    assert rec["emb_source"] is None
    assert os.path.isfile(rec["source"])
    assert os.path.isdir(rec["source"].replace(".csv", "_images"))
    assert not os.path.isfile(rec["source"].replace(".csv", ".json"))


def test_upload_build_stages_provided_embeddings_file(monkeypatch):
    monkeypatch.setattr(dataset_builder.features_mod, "run", _fake_features_run)
    images = [UploadedImage("a.png", b"AAAA")]

    job_id = dataset_builder.start_upload_build(images, [], FAKE_EMBEDDINGS, "Embedded Set", "")
    job = _wait_for_job(job_id)

    assert job["status"] == "done"
    assert job["dataset"]["has_embeddings"] is True
    rec = db.get_created_dataset_by_source(job["dataset"]["source"])
    assert rec["emb_source"] == rec["source"].replace(".csv", ".json")
    with open(rec["emb_source"], "rb") as f:
        assert f.read() == FAKE_EMBEDDINGS


def test_upload_build_with_annotations_stages_matching_labels(monkeypatch):
    monkeypatch.setattr(dataset_builder.features_mod, "run", _fake_features_run)
    images = [UploadedImage("weeds/a.png", b"AAAA"), UploadedImage("soil/b.png", b"BBBB")]
    annotations = [UploadedImage("weeds/a.json", b'{"ok": true}')]

    job_id = dataset_builder.start_upload_build(images, annotations, None, "Labeled Set", "")
    job = _wait_for_job(job_id)

    assert job["status"] == "done"
    images_dir = os.path.join(dataset_builder.USER_DATA_REL, "labeled_set_images", "images")
    labels_dir = os.path.join(dataset_builder.USER_DATA_REL, "labeled_set_images", "labels")
    staged_a = next(f for f in os.listdir(images_dir) if f.startswith("a-"))
    stem = os.path.splitext(staged_a)[0]
    with open(os.path.join(labels_dir, f"{stem}.json"), "rb") as f:
        assert f.read() == b'{"ok": true}'
    # b.png had no matching annotation — no label file for it.
    staged_b = next(f for f in os.listdir(images_dir) if f.startswith("b-"))
    b_stem = os.path.splitext(staged_b)[0]
    assert not os.path.isfile(os.path.join(labels_dir, f"{b_stem}.json"))


def test_upload_build_rejects_concurrent_build(monkeypatch):
    monkeypatch.setattr(dataset_builder.features_mod, "run", _fake_features_run)
    images = [UploadedImage("a.png", b"AAAA")]
    dataset_builder.start_upload_build(images, [], None, "First", "")

    with pytest.raises(dataset_builder.BuildInProgressError):
        dataset_builder.start_upload_build(images, [], None, "Second", "")


def test_upload_build_failure_cleans_up_and_reports_error(monkeypatch):
    def failing_features_run(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(dataset_builder.features_mod, "run", failing_features_run)
    images = [UploadedImage("a.png", b"AAAA")]

    job_id = dataset_builder.start_upload_build(images, [], None, "Broken Set", "")
    job = _wait_for_job(job_id)

    assert job["status"] == "error"
    assert "boom" in job["error"]
    assert db.get_created_dataset_by_source("data_user/broken_set.csv") is None
    assert not os.path.isdir("data_user/broken_set_images")


# ── Demo registry ──────────────────────────────────────────────────────────────

def test_demos_registry_has_both_demos_enabled():
    by_key = {d["key"]: d for d in dataset_builder.DEMOS}
    assert by_key["coco128"]["enabled"] is True
    assert by_key["agristress500"]["enabled"] is True


def test_start_demo_build_rejects_unknown_demo_key():
    with pytest.raises(dataset_builder.BuildValidationError):
        dataset_builder.start_demo_build("not-a-demo")


def test_start_demo_build_rejects_disabled_demo_key(monkeypatch):
    monkeypatch.setattr(dataset_builder, "DEMOS", dataset_builder.DEMOS + [
        {"key": "future-demo", "name": "Future Demo", "description": "not yet", "enabled": False},
    ])
    with pytest.raises(dataset_builder.BuildValidationError):
        dataset_builder.start_demo_build("future-demo")


def _patch_fake_coco128_pipeline(monkeypatch, tmp_path):
    def fake_download(url, dest, force=False):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake-zip")

    def fake_extract(zip_path, extract_dir, force=False):
        coco_root = extract_dir / "coco128"
        (coco_root / "images" / "train2017").mkdir(parents=True, exist_ok=True)
        (coco_root / "labels" / "train2017").mkdir(parents=True, exist_ok=True)
        return coco_root

    def fake_stage_coco128(args, root, coco_root, stage_root=None):
        stage_root = stage_root or (root / "image_sets" / args.dataset_stem)
        (stage_root / "images").mkdir(parents=True, exist_ok=True)
        manifest = stage_root / f"{args.dataset_stem}_input.csv"
        with open(manifest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["image_path", "cluster", "cluster_l2"])
            w.writeheader()
            w.writerow({"image_path": "image_sets/coco128/images/x.jpg",
                        "cluster": "all", "cluster_l2": "all"})
        return manifest, {"rows": 1}

    def fake_run_pipeline(args, root, manifest, data_dir=None):
        # Mirrors the real run_pipeline's no_embeddings short-circuit — demo builds
        # always pass no_embeddings=True, so this never produces a .json sibling
        # (COCO128's embeddings come from the shipped asset, attached separately).
        data_dir = data_dir or (root / "data")
        data_dir.mkdir(exist_ok=True)
        feature_csv = data_dir / f"{args.dataset_stem}.csv"
        _fake_features_run(str(manifest), str(feature_csv))
        if args.no_embeddings:
            return feature_csv, None
        embeddings_json = data_dir / f"{args.dataset_stem}.json"
        embeddings_json.write_text('{"embeddings": {}}')
        return feature_csv, embeddings_json

    monkeypatch.setattr(dataset_builder.coco128, "download", fake_download)
    monkeypatch.setattr(dataset_builder.coco128, "extract", fake_extract)
    monkeypatch.setattr(dataset_builder.coco128, "stage_coco128", fake_stage_coco128)
    monkeypatch.setattr(dataset_builder.coco128, "run_pipeline", fake_run_pipeline)

    # Stand in for the real shipped scripts/coco128_embeddings.json — keyed to match
    # fake_stage_coco128's one fake row (basename x.jpg) rather than real COCO128 images.
    fake_asset = tmp_path / "fake_coco128_embeddings.json"
    fake_asset.write_text(json.dumps({"embeddings": {"x.jpg": [0.1, 0.2, 0.3]}}))
    monkeypatch.setattr(dataset_builder, "COCO128_EMBEDDINGS_ASSET", fake_asset)


def test_start_demo_build_end_to_end(monkeypatch, tmp_path):
    _patch_fake_coco128_pipeline(monkeypatch, tmp_path)

    job_id = dataset_builder.start_demo_build("coco128")
    job = _wait_for_job(job_id)

    assert job["status"] == "done"
    assert job["dataset"]["name"] == dataset_builder.COCO128_NAME
    assert job["dataset"]["has_embeddings"] is True
    rec = db.get_created_dataset_by_source(f"{dataset_builder.USER_DATA_REL}/{dataset_builder.COCO128_STEM}.csv")
    assert rec is not None
    assert rec["emb_source"] == f"{dataset_builder.USER_DATA_REL}/{dataset_builder.COCO128_STEM}.json"
    with open(rec["emb_source"]) as f:
        assert json.load(f) == {"embeddings": {"x.jpg": [0.1, 0.2, 0.3]}}


def test_start_demo_build_fails_if_shipped_embeddings_dont_match_images(monkeypatch, tmp_path):
    """The shipped embeddings asset must cover exactly the built dataset's images —
    a mismatch (e.g. a future COCO128 source change) fails the build loudly rather
    than silently registering incomplete or wrong embeddings."""
    _patch_fake_coco128_pipeline(monkeypatch, tmp_path)
    mismatched_asset = tmp_path / "mismatched_coco128_embeddings.json"
    mismatched_asset.write_text(json.dumps({"embeddings": {"not-x.jpg": [0.1]}}))
    monkeypatch.setattr(dataset_builder, "COCO128_EMBEDDINGS_ASSET", mismatched_asset)

    job_id = dataset_builder.start_demo_build("coco128")
    job = _wait_for_job(job_id)

    assert job["status"] == "error"
    assert db.get_created_dataset_by_source(f"{dataset_builder.USER_DATA_REL}/{dataset_builder.COCO128_STEM}.csv") is None


def test_start_demo_build_rejects_if_already_built():
    db.create_dataset_record(dataset_builder.COCO128_NAME, "",
                              f"data_user/{dataset_builder.COCO128_STEM}.csv",
                              None, None, 1)
    with pytest.raises(dataset_builder.BuildValidationError):
        dataset_builder.start_demo_build("coco128")


def test_start_demo_build_rejects_concurrent_build(monkeypatch, tmp_path):
    _patch_fake_coco128_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(dataset_builder.features_mod, "run", _fake_features_run)
    images = [UploadedImage("a.png", b"AAAA")]
    dataset_builder.start_upload_build(images, [], None, "First", "")

    with pytest.raises(dataset_builder.BuildInProgressError):
        dataset_builder.start_demo_build("coco128")


# ── AgriStress-500 demo build (Hugging Face images+masks, features computed locally,
#    embeddings still brought in from the CDN) ──────────────────────────────────────

def _tiny_png_bytes(color=(200, 50, 50)):
    buf = io.BytesIO()
    Image.fromarray(np.full((8, 8, 3), color, np.uint8), "RGB").save(buf, format="PNG")
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_urlopen_agristress(monkeypatch, basenames=("pai-aaa.png", "pai-bbb.png")):
    """Fakes the HF repo listing (two image/mask pairs under different label folders,
    plus an unrelated instances/ entry that must be ignored), the per-file downloads,
    and the CDN embeddings download."""
    siblings = [{"rfilename": "README.md"}]
    for i, b in enumerate(basenames):
        label = f"L{i}"
        siblings.append({"rfilename": f"images/{label}/{b}"})
        siblings.append({"rfilename": f"masks/{label}/{b}"})
        siblings.append({"rfilename": f"instances/{label}/{b.replace('.png', '-1.png')}"})
    hf_info = json.dumps({"siblings": siblings}).encode()

    image_bytes = {b: _tiny_png_bytes((10 * i, 20, 30)) for i, b in enumerate(basenames)}
    mask_bytes = {b: b"not-a-real-image-but-never-decoded-during-build" for b in basenames}
    emb_bytes = json.dumps({"embeddings": {b: [0.1, 0.2] for b in basenames}}).encode()

    def fake_urlopen(url, timeout=None):
        if url == dataset_builder.AGRISTRESS_HF_API_URL:
            return _FakeResponse(hf_info)
        if url == dataset_builder.AGRISTRESS_EMBEDDINGS_URL:
            return _FakeResponse(emb_bytes)
        for b in basenames:
            if url.endswith(f"/images/L{basenames.index(b)}/{b}"):
                return _FakeResponse(image_bytes[b])
            if url.endswith(f"/masks/L{basenames.index(b)}/{b}"):
                return _FakeResponse(mask_bytes[b])
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(dataset_builder.urllib.request, "urlopen", fake_urlopen)
    return image_bytes, mask_bytes, emb_bytes


def test_agristress_file_pairs_matches_images_to_masks_and_skips_instances(monkeypatch):
    _fake_urlopen_agristress(monkeypatch)

    pairs = dataset_builder._agristress_file_pairs()

    assert {p[0] for p in pairs} == {"pai-aaa.png", "pai-bbb.png"}
    for basename, image_rel, mask_rel in pairs:
        assert image_rel.startswith("images/")
        assert mask_rel.startswith("masks/")
        assert image_rel.endswith(basename)
        assert mask_rel.endswith(basename)


def test_start_agristress_build_end_to_end(monkeypatch):
    image_bytes, mask_bytes, emb_bytes = _fake_urlopen_agristress(monkeypatch)
    monkeypatch.setattr(dataset_builder.features_mod, "run", _fake_features_run)

    job_id = dataset_builder.start_demo_build("agristress500")
    job = _wait_for_job(job_id)

    assert job["status"] == "done"
    assert job["dataset"]["name"] == dataset_builder.AGRISTRESS_NAME
    assert job["dataset"]["has_embeddings"] is True
    assert job["dataset"]["row_count"] == 2

    rec = db.get_created_dataset_by_source(f"{dataset_builder.USER_DATA_REL}/{dataset_builder.AGRISTRESS_STEM}.csv")
    assert rec is not None
    with open(rec["emb_source"], "rb") as f:
        assert f.read() == emb_bytes

    # Images, masks, and locally-generated thumbnails are all staged on disk.
    stage_root = f"{dataset_builder.USER_DATA_REL}/{dataset_builder.AGRISTRESS_STEM}_images"
    for basename, data in image_bytes.items():
        with open(f"{stage_root}/images/{basename}", "rb") as f:
            assert f.read() == data
        with open(f"{stage_root}/masks/{basename}", "rb") as f:
            assert f.read() == mask_bytes[basename]
        thumb_path = f"{stage_root}/thumbnails/{basename.replace('.png', '.jpg')}"
        assert os.path.isfile(thumb_path)
        assert Image.open(thumb_path).size[0] <= dataset_builder.AGRISTRESS_THUMB_MAX

    # thumbnail_image survives features.run()'s cluster-only passthrough (re-attached
    # by row index afterwards).
    with open(rec["source"], newline="") as f:
        feature_rows = list(csv.DictReader(f))
    assert len(feature_rows) == 2
    assert all(r["thumbnail_image"].startswith(f"{stage_root}/thumbnails/") for r in feature_rows)


def test_start_agristress_build_rejects_if_already_built():
    db.create_dataset_record(dataset_builder.AGRISTRESS_NAME, "",
                              f"data_user/{dataset_builder.AGRISTRESS_STEM}.csv",
                              None, f"data_user/{dataset_builder.AGRISTRESS_STEM}.json", 1)
    with pytest.raises(dataset_builder.BuildValidationError):
        dataset_builder.start_demo_build("agristress500")


def test_start_agristress_build_cleans_up_on_listing_failure(monkeypatch):
    def failing_urlopen(url, timeout=None):
        raise OSError("network down")

    monkeypatch.setattr(dataset_builder.urllib.request, "urlopen", failing_urlopen)

    job_id = dataset_builder.start_demo_build("agristress500")
    job = _wait_for_job(job_id)

    assert job["status"] == "error"
    assert db.get_created_dataset_by_source(f"data_user/{dataset_builder.AGRISTRESS_STEM}.csv") is None
    assert not os.path.isdir(f"data_user/{dataset_builder.AGRISTRESS_STEM}_images")
