# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the upload-staging helper. No network, no torch.

Run:  python -m pytest precisionai/magnidata/tools/test_staging.py -q
"""

import csv
import hashlib
import os

from .staging import UploadedImage, stage_images


def test_stage_images_writes_files_and_manifest(tmp_path):
    images = [
        UploadedImage(relative_path="a.png", data=b"AAAA"),
        UploadedImage(relative_path="b.png", data=b"BBBB"),
    ]
    stage_root = os.path.join(str(tmp_path), "myset_images")

    manifest_path = stage_images(images, stage_root)

    assert manifest_path == os.path.join(stage_root, "manifest.csv")
    staged_files = sorted(os.listdir(os.path.join(stage_root, "images")))
    assert len(staged_files) == 2
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["image_path"].startswith(os.path.join(stage_root, "images"))


def test_flat_upload_gets_uncategorized_cluster(tmp_path):
    images = [UploadedImage(relative_path="a.png", data=b"AAAA")]
    manifest_path = stage_images(images, os.path.join(str(tmp_path), "s"))

    with open(manifest_path, newline="") as f:
        row = next(csv.DictReader(f))
    assert row["cluster"] == "uncategorized"
    assert row["cluster_l2"] == "uncategorized"


def test_folder_upload_uses_relative_path_as_cluster(tmp_path):
    images = [
        UploadedImage(relative_path="weeds/patch1/a.png", data=b"AAAA"),
        UploadedImage(relative_path="soil/b.png", data=b"BBBB"),
    ]
    manifest_path = stage_images(images, os.path.join(str(tmp_path), "s"))

    with open(manifest_path, newline="") as f:
        by_cluster = {row["cluster"]: row["cluster_l2"] for row in csv.DictReader(f)}
    assert by_cluster["weeds/patch1"] == "weeds"
    assert by_cluster["soil"] == "soil"


def test_staged_filenames_are_slugified_and_hash_suffixed(tmp_path):
    images = [UploadedImage(relative_path="My Photo #1.JPG", data=b"CONTENT")]
    stage_images(images, os.path.join(str(tmp_path), "s"))

    staged = os.listdir(os.path.join(str(tmp_path), "s", "images"))
    digest = hashlib.sha256(b"CONTENT").hexdigest()[:8]
    assert staged == [f"my_photo_1-{digest}.jpg"]


def test_same_basename_in_different_folders_does_not_collide(tmp_path):
    images = [
        UploadedImage(relative_path="weeds/img.png", data=b"ONE"),
        UploadedImage(relative_path="soil/img.png", data=b"TWO"),
    ]
    stage_images(images, os.path.join(str(tmp_path), "s"))

    staged = os.listdir(os.path.join(str(tmp_path), "s", "images"))
    assert len(staged) == 2
    assert len(set(staged)) == 2


def test_checksums_file_has_one_line_per_image(tmp_path):
    images = [
        UploadedImage(relative_path="a.png", data=b"AAAA"),
        UploadedImage(relative_path="b.png", data=b"BBBB"),
    ]
    stage_images(images, os.path.join(str(tmp_path), "s"))

    with open(os.path.join(str(tmp_path), "s", "checksums.sha256")) as f:
        lines = [line for line in f.read().splitlines() if line.strip()]
    assert len(lines) == 2
    assert lines[0].split()[0] == hashlib.sha256(b"AAAA").hexdigest()


def test_stage_images_accepts_source_backed_files(tmp_path):
    source = tmp_path / "upload.bin"
    source.write_bytes(b"CONTENT")
    images = [UploadedImage(relative_path="My Photo #1.JPG", source_path=str(source))]
    stage_root = os.path.join(str(tmp_path), "s")

    stage_images(images, stage_root)

    digest = hashlib.sha256(b"CONTENT").hexdigest()
    staged_path = os.path.join(stage_root, "images", f"my_photo_1-{digest[:8]}.jpg")
    assert os.path.isfile(staged_path)
    with open(staged_path, "rb") as f:
        assert f.read() == b"CONTENT"


# ── Annotations ───────────────────────────────────────────────────────────────


def test_no_annotations_means_no_labels_dir(tmp_path):
    images = [UploadedImage(relative_path="a.png", data=b"AAAA")]
    stage_root = os.path.join(str(tmp_path), "s")

    stage_images(images, stage_root)

    assert not os.path.isdir(os.path.join(stage_root, "labels"))


def test_matched_annotation_written_to_staged_image_stem(tmp_path):
    images = [UploadedImage(relative_path="weeds/patch1/img.png", data=b"AAAA")]
    annotations = [UploadedImage(relative_path="weeds/patch1/img.json", data=b'{"a": 1}')]
    stage_root = os.path.join(str(tmp_path), "s")

    stage_images(images, stage_root, annotations=annotations)

    staged_image = os.listdir(os.path.join(stage_root, "images"))[0]
    staged_stem = os.path.splitext(staged_image)[0]
    label_path = os.path.join(stage_root, "labels", f"{staged_stem}.json")
    assert os.path.isfile(label_path)
    with open(label_path, "rb") as f:
        assert f.read() == b'{"a": 1}'


def test_unmatched_annotation_is_not_an_error(tmp_path):
    images = [UploadedImage(relative_path="img.png", data=b"AAAA")]
    annotations = [UploadedImage(relative_path="other/nope.json", data=b"{}")]
    stage_root = os.path.join(str(tmp_path), "s")

    manifest_path = stage_images(images, stage_root, annotations=annotations)

    assert os.path.isfile(manifest_path)
    labels_dir = os.path.join(stage_root, "labels")
    assert not os.path.isdir(labels_dir) or not os.listdir(labels_dir)


def test_unmatched_image_has_no_label_file(tmp_path):
    images = [
        UploadedImage(relative_path="a.png", data=b"AAAA"),
        UploadedImage(relative_path="b.png", data=b"BBBB"),
    ]
    annotations = [UploadedImage(relative_path="a.json", data=b"{}")]
    stage_root = os.path.join(str(tmp_path), "s")

    stage_images(images, stage_root, annotations=annotations)

    labels = os.listdir(os.path.join(stage_root, "labels"))
    assert len(labels) == 1


def test_annotation_match_is_by_basename_stem_not_full_path(tmp_path):
    """The original relative FOLDER doesn't have to match — only the basename stem."""
    images = [UploadedImage(relative_path="images/sub/img.png", data=b"AAAA")]
    annotations = [UploadedImage(relative_path="labels/elsewhere/img.json", data=b"{}")]
    stage_root = os.path.join(str(tmp_path), "s")

    stage_images(images, stage_root, annotations=annotations)

    assert len(os.listdir(os.path.join(stage_root, "labels"))) == 1


def test_annotation_matching_is_case_sensitive(tmp_path):
    images = [UploadedImage(relative_path="IMG.png", data=b"AAAA")]
    annotations = [UploadedImage(relative_path="img.json", data=b"{}")]
    stage_root = os.path.join(str(tmp_path), "s")

    stage_images(images, stage_root, annotations=annotations)

    labels_dir = os.path.join(stage_root, "labels")
    assert not os.path.isdir(labels_dir) or not os.listdir(labels_dir)


def test_empty_annotations_list_behaves_like_none(tmp_path):
    images = [UploadedImage(relative_path="a.png", data=b"AAAA")]
    stage_root = os.path.join(str(tmp_path), "s")

    stage_images(images, stage_root, annotations=[])

    assert not os.path.isdir(os.path.join(stage_root, "labels"))
