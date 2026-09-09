# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Stage uploaded images into a dataset-ready layout: <stage_root>/images/<name>, a manifest CSV (image_path, cluster, cluster_l2), and a checksums file.

Optionally also stages uploaded COCO-format annotation JSON files into
<stage_root>/labels/<staged-image-stem>.json — the exact sibling path
tools/features.py::_derive_paths looks for next to <stage_root>/images/<stem>.ext.
An annotation is matched to an image by comparing the annotation's *original*
relative path's stem against the image's *original* stem (exact, case-sensitive).
An annotation with no matching image, or an image with no matching annotation, is
fine (not an error) — annotations are entirely optional, per-image.

No metadata CSV is written here — tools/features.py::process_row already handles
a missing labels/<stem>.json and metadata/images_metadata.csv gracefully (blank
COCO/metadata columns, RGB-only columns still populate — see
test_missing_label_continues in test_features.py and metadata_join.py's
isfile-guarded cache).
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
from dataclasses import dataclass

CHUNK_SIZE = 1024 * 1024


@dataclass
class UploadedImage:
    """One uploaded file, either in memory or backed by a temporary source path."""

    relative_path: str  # as uploaded, e.g. "field_rows/IMG_002.jpg" or "IMG_002.jpg"
    data: bytes | None = None
    source_path: str | None = None


def _slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_").lower()
    return s or "image"


def _staged_filename(relative_path: str, digest: str) -> str:
    stem, ext = os.path.splitext(os.path.basename(relative_path))
    return f"{_slugify(stem)}-{digest[:8]}{ext.lower()}"


def _sha256(upload: UploadedImage) -> str:
    h = hashlib.sha256()
    if upload.data is not None:
        h.update(upload.data)
        return h.hexdigest()
    if upload.source_path:
        with open(upload.source_path, "rb") as f:
            while chunk := f.read(CHUNK_SIZE):
                h.update(chunk)
        return h.hexdigest()
    raise ValueError(f"uploaded file {upload.relative_path!r} has no data or source_path")


def _write_upload(upload: UploadedImage, dest: str) -> None:
    if upload.data is not None:
        with open(dest, "wb") as out:
            out.write(upload.data)
        return

    source_path = upload.source_path
    if source_path is None:
        raise ValueError(f"uploaded file {upload.relative_path!r} has no data or source_path")
    with open(dest, "wb") as out, open(source_path, "rb") as src:
        while chunk := src.read(CHUNK_SIZE):
            out.write(chunk)


def cluster_for(relative_path: str) -> tuple[str, str]:
    """(cluster, cluster_l2) from the relative folder path; a flat layout (no subfolder) gets 'uncategorized' for both."""
    parts = relative_path.replace("\\", "/").split("/")[:-1]
    if not parts:
        return "uncategorized", "uncategorized"
    return "/".join(parts), parts[0]


def _original_stem(relative_path: str) -> str:
    return os.path.splitext(os.path.basename(relative_path))[0]


def stage_images(images: list[UploadedImage], stage_root: str, annotations: list[UploadedImage] | None = None) -> str:
    """Write each image under <stage_root>/images/, plus a manifest CSV (<stage_root>/manifest.csv) and a checksums file (<stage_root>/checksums.sha256), and return the manifest CSV path.

    If `annotations` is given, each one whose original stem
    (os.path.splitext(os.path.basename(relative_path))[0]) matches an uploaded
    image's original stem is written as-is to
    <stage_root>/labels/<staged-image-stem>.json (the raw bytes, already valid
    JSON from the client — not re-parsed here). If more than one uploaded image
    shares the same original stem, an annotation matching that stem is written
    for all of them (there is no other information to disambiguate). Unmatched
    annotations and unmatched images are both fine — annotations are optional.

    `stage_root` is used exactly as given (via os.path.join) — a caller that needs
    DATA_ROOT-relative `image_path` values in the manifest (as every other dataset in
    this app has) should pass a DATA_ROOT-relative `stage_root` while the process's
    current working directory is DATA_ROOT (the convention
    scripts/prepare_coco128_dashboard.py already uses).
    """
    images_dir = os.path.join(stage_root, "images")
    os.makedirs(images_dir, exist_ok=True)

    rows = []
    checksum_lines = []
    stem_map: dict[str, list[str]] = {}
    for img in images:
        digest = _sha256(img)
        staged_name = _staged_filename(img.relative_path, digest)
        staged_path = os.path.join(images_dir, staged_name)
        _write_upload(img, staged_path)

        cluster, cluster_l2 = cluster_for(img.relative_path)
        rows.append(
            {
                "image_path": os.path.join(images_dir, staged_name),
                "cluster": cluster,
                "cluster_l2": cluster_l2,
            }
        )
        checksum_lines.append(f"{digest}  {staged_name}")

        staged_stem = os.path.splitext(staged_name)[0]
        stem_map.setdefault(_original_stem(img.relative_path), []).append(staged_stem)

    manifest_path = os.path.join(stage_root, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "cluster", "cluster_l2"])
        writer.writeheader()
        writer.writerows(rows)

    with open(os.path.join(stage_root, "checksums.sha256"), "w") as f:
        f.write("\n".join(checksum_lines) + ("\n" if checksum_lines else ""))

    if annotations:
        _stage_annotations(annotations, stem_map, stage_root)

    return manifest_path


def _stage_annotations(annotations: list[UploadedImage], stem_map: dict[str, list[str]], stage_root: str) -> None:
    labels_dir = os.path.join(stage_root, "labels")
    for ann in annotations:
        staged_stems = stem_map.get(_original_stem(ann.relative_path))
        if not staged_stems:
            continue
        os.makedirs(labels_dir, exist_ok=True)
        for staged_stem in staged_stems:
            _write_upload(ann, os.path.join(labels_dir, f"{staged_stem}.json"))
