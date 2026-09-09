# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Resolve an image_path to a readable file, with thumbnail fallback.

The CSV holds full-resolution paths under DATALAKE_ROOT
(``<root>/<DATASET>/images/<STEM>.png``). Pre-generated thumbnails live at a parallel
root with the SAME relative layout but a ``.jpg`` extension
(``<thumb_root>/<DATASET>/images/<STEM>.jpg``). When the full-res image is not present
on this machine, fall back to a thumbnail so the run still produces features.

Roots are env-overridable (matching magnidata/api/local_files.py):
  DATALAKE_ROOT, THUMB_ROOT_720, THUMB_ROOT_64

NOTE: thumbnails are downscaled JPEGs, so image-derived features (blur, noise,
illumination, NIMA) computed from them are NOT comparable to full-res values. The
chosen source is recorded per row in the ``image_source`` column and in provenance.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

DATALAKE_ROOT = os.environ.get("DATALAKE_ROOT", "/mnt/disks/ssd-raid/datalake-clone")
# (label, root) in preference order — 720 first (higher resolution than 64).
THUMB_ROOTS = [
    ("thumb720", os.environ.get("THUMB_ROOT_720", "/mnt/disks/ssd-raid/datalake-clone-thumbnails/720")),
    ("thumb64", os.environ.get("THUMB_ROOT_64", "/mnt/disks/ssd-raid/datalake-clone-thumbnails/64")),
]
_THUMB_EXTS = (".jpg", ".jpeg", ".png")


def _thumb_candidates(path: str) -> Iterator[tuple[str, str]]:
    if not path.startswith(DATALAKE_ROOT):
        return
    rel_base = os.path.splitext(path[len(DATALAKE_ROOT) :])[0]
    for label, root in THUMB_ROOTS:
        for ext in _THUMB_EXTS:
            yield label, root + rel_base + ext


def resolve(image_path: str, mode: str = "auto") -> tuple[str, str]:
    """Return (readable_path, source_label).

    mode: 'auto' (full-res, fall back to thumbnail) | 'fullres' (full-res only) |
          'thumbnail' (thumbnail, fall back to full-res). source_label is one of
          'fullres' | 'thumb720' | 'thumb64' | 'missing'.
    """
    full_ok = bool(image_path) and os.path.isfile(image_path)

    if mode == "fullres":
        return (image_path, "fullres") if full_ok else (image_path, "missing")

    if mode == "auto" and full_ok:
        return image_path, "fullres"

    for label, cand in _thumb_candidates(image_path):
        if os.path.isfile(cand):
            return cand, label

    # thumbnail not found: in 'thumbnail' mode still try full-res before giving up
    if full_ok:
        return image_path, "fullres"
    return image_path, "missing"


def find_path_column(fieldnames: list[str]) -> str:
    """Best-effort image-path column detection.

    Prefers an exact/likely 'image_path' name, else any column with 'path' in
    its name, else the first column. Shared by features.py and embeddings.py
    so both tools apply the same convention to their input CSV.
    """
    for c in fieldnames:
        lc = c.lower()
        if lc == "image_path" or ("image" in lc and "path" in lc):
            return c
    for c in fieldnames:
        if "path" in c.lower():
            return c
    return fieldnames[0]
