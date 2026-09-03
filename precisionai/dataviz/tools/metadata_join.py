# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Join ``<DATASET>/metadata/images_metadata.csv`` onto each image by stem.

Trimmed to what the curated schema keeps: GSD, a derived camera-angle CATEGORY, and a
few optional domain-specific passthroughs. The camera angle is translated to {nadir,
oriented, missing} using ``camera_view`` (authoritative: nadir / oblique / none) with
the numeric ``angle`` (only ever ~35/60° for oblique, empty for nadir in this data) as
a fallback.
"""

from __future__ import annotations

import csv
import os

# off-nadir tilt (deg) at/below which a capture counts as nadir, when only the
# numeric angle is available (camera_view absent).
NADIR_DEG_TOL = 10.0

# straight metadata -> output passthroughs (no transform)
_PASSTHROUGH = [
    ("gsd", "gsd"),
    ("domain_metric", "domain_metric"),
]
_NUMERIC = {"gsd"}

_KEY_COLS = ("id", "original_name", "uuid")
_CACHE: dict[str, dict[str, dict]] = {}


def _index_csv(csv_path: str) -> dict[str, dict]:
    idx: dict[str, dict] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            for kc in _KEY_COLS:
                v = (row.get(kc) or "").strip()
                if v:
                    idx.setdefault(v.lower(), row)
    return idx


def _to_float(s: str) -> float | str:
    try:
        return float(s)
    except (TypeError, ValueError):
        return ""


def camera_angle_category(angle_raw: str | None, view_raw: str | None) -> str:
    """{nadir, oriented, missing} from camera_view (primary) + numeric angle (fallback)."""
    view = (view_raw or "").strip().lower()
    if view == "nadir":
        return "nadir"
    if view == "oblique":
        return "oriented"
    a = _to_float((angle_raw or "").strip())
    if isinstance(a, str):
        return "missing"
    return "nadir" if abs(a) <= NADIR_DEG_TOL else "oriented"


def metadata_csv_path(dataset_dir: str) -> str:
    """Return the path to `dataset_dir`'s images_metadata.csv sidecar."""
    return os.path.join(dataset_dir, "metadata", "images_metadata.csv")


def lookup(dataset_dir: str, stem: str) -> tuple[dict, bool]:
    """Return (mapped_columns, matched) using OUTPUT_COLUMNS names."""
    path = metadata_csv_path(dataset_dir)
    if path not in _CACHE:
        _CACHE[path] = _index_csv(path) if os.path.isfile(path) else {}
    row = _CACHE[path].get(stem.lower())
    if row is None:
        return {"camera_angle": "missing"}, False

    out: dict = {"camera_angle": camera_angle_category(row.get("angle"), row.get("camera_view"))}
    for meta_key, out_col in _PASSTHROUGH:
        raw = (row.get(meta_key) or "").strip()
        if raw == "":
            continue
        out[out_col] = _to_float(raw) if out_col in _NUMERIC else raw
    return out, True
