# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Precision AI — CSV Dashboard API.

Serves images/masks/overlays from local disk, stores notes in SQLite.
Also exposes dataset listing and CSV serving from local disk.
"""

import colorsys
import contextlib
import csv
import glob
import io
import json
import os
import re
import shutil
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import yaml
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file
from flask.typing import ResponseReturnValue
from flask_cors import CORS
from PIL import Image

# Imported at module load (not lazily inside the /api/embedding/* handlers that use them) so
# the one-off cost of importing sklearn's compiled extensions is paid once at server startup
# rather than stalling — and occasionally timing out on the client, stranding the 3D view mid
# render — whichever request happens to be the first to hit one of these endpoints.
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

load_dotenv()

CONFIG_PATH = os.environ.get("CONFIG_PATH", "confi.yaml")
DATA_ROOT = os.environ.get("DATA_ROOT", ".")
# Writable location for user-created (child) datasets — kept separate from the read-only `data/`.
USER_DATA_REL = "data_user"
USER_DATA_DIR = os.path.join(DATA_ROOT, USER_DATA_REL)
os.makedirs(USER_DATA_DIR, exist_ok=True)

# dataset_builder must be imported before precisionai.dataviz.tools below — importing it
# inserts the precisionai package root onto sys.path (see its own _find_package_root),
# which that import relies on.
import dataset_builder
import db
import local_files

from precisionai.dataviz.tools import coco_labels as coco_labels_mod

db.init_db()

app = Flask(__name__)
CORS(app)

# Background threshold: pixels whose R+G+B sum is below this are treated as
# background and rendered transparent in the mask/overlay views.
BG_THRESHOLD = 30
LFS_POINTER_VERSION = "version https://git-lfs.github.com/spec/v1"


def _lfs_pointer_info(path: str) -> dict | None:
    """Return Git LFS pointer metadata when `path` is a pointer stub."""
    try:
        with open(path, "rb") as f:
            head = f.read(512)
    except OSError:
        return None
    try:
        text = head.decode("utf-8")
    except UnicodeDecodeError:
        return None

    lines = text.splitlines()
    if not lines or lines[0].strip() != LFS_POINTER_VERSION:
        return None

    info: dict = {}
    for line in lines[1:]:
        if line.startswith("oid "):
            info["oid"] = line[4:].strip()
        elif line.startswith("size "):
            with contextlib.suppress(ValueError):
                info["size"] = int(line[5:].strip())
    return info


def _img_from_bytes(data: bytes, max_size: int = 1600) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    if max(img.size) > max_size:
        img = img.copy()
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return img


def _mask_as_rgba(mask_data: bytes, max_size: int = 0) -> Image.Image:
    """Load an RGB(A) mask PNG and return an RGBA image where black/near-black background pixels are fully transparent and coloured foreground pixels keep their original colour at full (255) alpha.

    The masks in this dataset are pre-coloured: each segmentation class already
    has its own RGB colour; black (0,0,0) means background.
    """
    mask = Image.open(io.BytesIO(mask_data))

    # Normalise to RGBA
    if mask.mode != "RGBA":
        mask = mask.convert("RGBA")

    arr = np.array(mask, dtype=np.uint8)  # (H, W, 4)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Background = near-black in the original RGB
    is_bg = (r.astype(np.uint16) + g + b) < BG_THRESHOLD
    arr[:, :, 3] = np.where(is_bg, 0, 255)  # transparent bg, opaque fg

    result = Image.fromarray(arr, "RGBA")

    if max_size and max(result.size) > max_size:
        result.thumbnail((max_size, max_size), Image.Resampling.NEAREST)

    return result


def _to_jpeg(img: Image.Image, quality: int = 88) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health() -> ResponseReturnValue:
    """Return service status and local-filesystem/datalake accessibility."""
    return jsonify({"status": "ok", "local": local_files.local_status()})


# ── Images ─────────────────────────────────────────────────────────────────────
@app.get("/api/image")
def get_image() -> ResponseReturnValue:
    """Return the image at `path` as a size-capped JPEG."""
    path = request.args.get("path", "").strip()
    max_size = int(request.args.get("max_size", 1600))
    if not path:
        return jsonify({"error": "path required"}), 400
    try:
        data = local_files.read_thumbnail(path, max_size) or local_files.read_file(path)
        img = _img_from_bytes(data, max_size)
        buf = io.BytesIO(_to_jpeg(img))
        buf.seek(0)
        return send_file(buf, mimetype="image/jpeg")
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/mask")
def get_mask() -> ResponseReturnValue:
    """Return the mask as a transparent-background PNG.

    Coloured foreground pixels keep their original RGB colours; black pixels
    become fully transparent so the mask can be viewed on any background.
    """
    path = request.args.get("path", "").strip()
    max_size = int(request.args.get("max_size", 1600))
    if not path:
        return jsonify({"error": "path required"}), 400
    try:
        data = local_files.read_thumbnail(path, max_size) or local_files.read_file(path)
        colored = _mask_as_rgba(data, max_size)
        buf = io.BytesIO(_to_png(colored))
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/overlay")
def get_overlay() -> ResponseReturnValue:
    """Alpha-composite the pre-coloured mask directly over the source image.

    Background (near-black) mask pixels are transparent so the image shows
    through; coloured foreground pixels are fully opaque on top of the image.
    """
    img_path = request.args.get("img", "").strip()
    mask_path = request.args.get("mask", "").strip()
    max_size = int(request.args.get("max_size", 1600))

    if not img_path or not mask_path:
        return jsonify({"error": "img and mask paths required"}), 400
    try:
        img_data = local_files.read_thumbnail(img_path, max_size) or local_files.read_file(img_path)
        mask_data = local_files.read_thumbnail(mask_path, max_size) or local_files.read_file(mask_path)

        base = _img_from_bytes(img_data, max_size).convert("RGBA")
        overlay = _mask_as_rgba(mask_data)

        # Resize mask to match image if they differ (preserves colour regions)
        if overlay.size != base.size:
            overlay = overlay.resize(base.size, Image.Resampling.NEAREST)

        result = Image.alpha_composite(base, overlay)
        buf = io.BytesIO(_to_jpeg(result))
        buf.seek(0)
        return send_file(buf, mimetype="image/jpeg")
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/mask/debug")
def debug_mask() -> ResponseReturnValue:
    """Inspect raw mask properties — helps diagnose rendering issues."""
    path = request.args.get("path", "").strip()
    if not path:
        return jsonify({"error": "path required"}), 400
    try:
        data = local_files.read_file(path)
        mask = Image.open(io.BytesIO(data))
        raw = np.array(mask)
        rgba = _mask_as_rgba(data)
        fg = np.array(rgba)[:, :, 3] > 0
        return jsonify(
            {
                "path": path,
                "pil_mode": mask.mode,
                "pil_size": list(mask.size),
                "raw_shape": list(raw.shape),
                "raw_dtype": str(raw.dtype),
                "raw_min": int(raw.min()),
                "raw_max": int(raw.max()),
                "fg_pixels": int(fg.sum()),
                "total_pixels": int(fg.size),
                "coverage_pct": round(100 * fg.sum() / fg.size, 2),
                "bg_threshold": BG_THRESHOLD,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Annotation mask / overlay (from whichever source the image has) ────────────
# Two mask sources, tried in order:
#   1. COCO polygon labels — labels/<stem>.json sibling to images/<stem>.ext (same
#      convention tools/features.py::_derive_paths reads for instance features).
#      Rasterized on the fly, one distinct color per category.
#   2. A pre-colored raster mask — masks/<stem>.<ext> sibling to images/<stem>.ext
#      (what /api/mask + /api/overlay above expect too, but nothing ever populated
#      a mask_path column for them; datasets built with real masks, e.g.
#      AgriStress-500, use this path instead of a CSV column at all).
# 404 when the image has neither — the common case for most images/datasets, not a
# server error.


def _category_color(category_id: int) -> tuple[int, int, int]:
    """Deterministic, well-spread color per category id (golden-ratio hue step)."""
    hue = (category_id * 0.6180339887498949) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def _annotation_label_path(image_path: str) -> str:
    """<DATASET>/labels/<STEM>.json for <DATASET>/images/<STEM>.ext."""
    images_dir = os.path.dirname(image_path)
    dataset_dir = os.path.dirname(images_dir)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(dataset_dir, "labels", stem + ".json")


def _rasterize_annotation_mask(image_path: str, target_wh: tuple[int, int]) -> Image.Image | None:
    """Colored RGBA mask rasterized from the image's COCO label file — transparent where unlabeled/background, a distinct color per category elsewhere.

    None if the image has no label file.
    """
    label_abs = os.path.normpath(os.path.join(DATA_ROOT, _annotation_label_path(image_path)))
    if not os.path.isfile(label_abs):
        return None
    labels = coco_labels_mod.parse(label_abs)
    class_map = coco_labels_mod.rasterize_class_map(labels, target_wh=target_wh)

    rgba = np.zeros((*class_map.shape, 4), dtype=np.uint8)
    bg_val = coco_labels_mod.BACKGROUND_CATEGORY_ID + coco_labels_mod.CLASS_MAP_OFFSET
    for cid in np.unique(class_map):
        if cid in (0, bg_val):
            continue
        color = _category_color(int(cid) - coco_labels_mod.CLASS_MAP_OFFSET)
        m = class_map == cid
        rgba[m, 0], rgba[m, 1], rgba[m, 2], rgba[m, 3] = color[0], color[1], color[2], 255
    return Image.fromarray(rgba, "RGBA")


def _sibling_raster_mask_bytes(image_path: str) -> bytes | None:
    """Bytes of a pre-colored raster mask sibling to the image — <DATASET>/masks/<STEM>.ext next to <DATASET>/images/<STEM>.ext.

    This is the convention datasets like AgriStress-500 ship: one PNG per image,
    already colored per class, black = background — exactly _mask_as_rgba's format.
    Tries local paths and, for a remote image_path, the same allowlisted hosts
    /api/image already supports. None if no candidate is readable.
    """
    images_dir = image_path.rsplit("/", 1)[0] if "/" in image_path else ""
    dataset_dir = images_dir.rsplit("/", 1)[0] if "/" in images_dir else ""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = f"{dataset_dir}/masks/{stem}{ext}"
        resolved = (
            candidate if local_files.is_remote_url(image_path) else os.path.normpath(os.path.join(DATA_ROOT, candidate))
        )
        try:
            return local_files.read_file(resolved)
        except (OSError, PermissionError):
            continue
    return None


def _load_annotation_mask(image_path: str, target_wh: tuple[int, int]) -> Image.Image | None:
    """Return the image's mask, from whichever source it has.

    Real COCO polygon labels (exact per-instance shapes) take priority, falling back
    to a pre-colored raster mask file when there are no COCO labels. None if the image
    has neither.
    """
    coco_mask = _rasterize_annotation_mask(image_path, target_wh)
    if coco_mask is not None:
        return coco_mask
    raster_bytes = _sibling_raster_mask_bytes(image_path)
    if raster_bytes is None:
        return None
    colored = _mask_as_rgba(raster_bytes)
    if colored.size != target_wh:
        colored = colored.resize(target_wh, Image.Resampling.NEAREST)
    return colored


@app.get("/api/annotation/mask")
def get_annotation_mask() -> ResponseReturnValue:
    """Return the colorized segmentation mask alone, as a transparent PNG."""
    path = request.args.get("path", "").strip()
    max_size = int(request.args.get("max_size", 1600))
    if not path:
        return jsonify({"error": "path required"}), 400
    if os.path.isabs(path) or ".." in path:
        return jsonify({"error": "invalid path"}), 400
    try:
        image_data = local_files.read_thumbnail(path, max_size) or local_files.read_file(path)
        img = _img_from_bytes(image_data, max_size)
        mask = _load_annotation_mask(path, target_wh=img.size)
        if mask is None:
            return jsonify({"error": "no annotations for this image"}), 404
        buf = io.BytesIO(_to_png(mask))
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/annotation/overlay")
def get_annotation_overlay() -> ResponseReturnValue:
    """Return the image with its segmentation mask composited on top at a caller-chosen opacity.

    Opacity (0..1, default 0.5) backs the preview modal's Overlay transparency slider.
    """
    path = request.args.get("path", "").strip()
    max_size = int(request.args.get("max_size", 1600))
    try:
        alpha = max(0.0, min(1.0, float(request.args.get("alpha", 0.5))))
    except ValueError:
        alpha = 0.5
    if not path:
        return jsonify({"error": "path required"}), 400
    if os.path.isabs(path) or ".." in path:
        return jsonify({"error": "invalid path"}), 400
    try:
        image_data = local_files.read_thumbnail(path, max_size) or local_files.read_file(path)
        base = _img_from_bytes(image_data, max_size).convert("RGBA")
        mask = _load_annotation_mask(path, target_wh=base.size)
        if mask is None:
            return jsonify({"error": "no annotations for this image"}), 404
        arr = np.array(mask)
        arr[:, :, 3] = (arr[:, :, 3].astype(np.float32) * alpha).astype(np.uint8)
        result = Image.alpha_composite(base, Image.fromarray(arr, "RGBA"))
        buf = io.BytesIO(_to_jpeg(result))
        buf.seek(0)
        return send_file(buf, mimetype="image/jpeg")
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Notes ──────────────────────────────────────────────────────────────────────
@app.get("/api/notes")
def list_notes() -> ResponseReturnValue:
    """Return notes, optionally filtered to a single image stem."""
    stem = request.args.get("stem")
    return jsonify(db.list_notes(stem or None))


@app.post("/api/notes")
def create_note() -> ResponseReturnValue:
    """Create a note for an image stem."""
    body = request.get_json(silent=True) or {}
    stem = body.get("stem", "").strip()
    note = body.get("note", "").strip()
    if not stem or not note:
        return jsonify({"error": "stem and note required"}), 400
    return jsonify(db.create_note(stem, body.get("image_path", ""), note)), 201


@app.put("/api/notes/<int:note_id>")
def update_note(note_id: int) -> ResponseReturnValue:
    """Update the text of an existing note."""
    body = request.get_json(silent=True) or {}
    note = body.get("note", "").strip()
    if not note:
        return jsonify({"error": "note required"}), 400
    updated = db.update_note(note_id, note)
    if not updated:
        return jsonify({"error": "not found"}), 404
    return jsonify(updated)


@app.delete("/api/notes/<int:note_id>")
def delete_note(note_id: int) -> ResponseReturnValue:
    """Delete a note by id."""
    if not db.delete_note(note_id):
        return jsonify({"error": "not found"}), 404
    return "", 204


# ── Cluster partitions ───────────────────────────────────────────────────────────
@app.get("/api/partitions")
def get_partitions() -> ResponseReturnValue:
    """All partition annotations (cluster_id) for a dataset, keyed by image filename."""
    source = request.args.get("source", "").strip()
    if not source:
        return jsonify({"error": "source required"}), 400
    return jsonify({"partitions": db.list_partitions(source)})


@app.post("/api/partitions/commit")
def commit_partitions() -> ResponseReturnValue:
    """Commit a clustering: body {source, entries:[{image_name, cluster_id}]}.

    Upserts each row's cluster_id (overwriting a previous commit).
    """
    body = request.get_json(silent=True) or {}
    source = (body.get("source") or "").strip()
    entries = body.get("entries") or []
    if not source:
        return jsonify({"error": "source required"}), 400
    if not isinstance(entries, list):
        return jsonify({"error": "entries must be a list"}), 400
    clean = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        name = str(e.get("image_name") or "").strip()
        cid = e.get("cluster_id")
        if not name or cid is None:
            continue
        try:
            cid = int(cid)
        except (TypeError, ValueError):
            continue
        clean.append({"image_name": name, "cluster_id": cid})
    return jsonify({"committed": db.commit_partitions(source, clean)})


# ── Datasets ───────────────────────────────────────────────────────────────────
def _resolve_emb_source(source: str) -> str | None:
    """Return the dataset path whose <stem>.json holds the embeddings for `source`.

    Confi/root datasets use their own stem; created (child) datasets inherit the parent's.
    """
    rec = db.get_created_dataset_by_source(source)
    if rec is not None:
        emb_source = rec.get("emb_source") or None
        if not emb_source:
            return None
        stem = os.path.splitext(emb_source)[0]
        emb_path = os.path.normpath(os.path.join(DATA_ROOT, stem + ".json"))
        return emb_source if os.path.isfile(emb_path) and _lfs_pointer_info(emb_path) is None else None
    stem = os.path.splitext(source)[0]
    emb_path = os.path.normpath(os.path.join(DATA_ROOT, stem + ".json"))
    return source if os.path.isfile(emb_path) and _lfs_pointer_info(emb_path) is None else None


def _confi_datasets() -> list[dict]:
    try:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return []
    return cfg.get("datasets", [])


@app.get("/api/datasets")
def list_datasets() -> ResponseReturnValue:
    """Datasets from confi.yaml (permanent) + user-created child datasets (deletable)."""
    try:
        out = []
        for ds in _confi_datasets():
            source = ds.get("source", "")
            path = os.path.normpath(os.path.join(DATA_ROOT, source))
            lfs_info = _lfs_pointer_info(path) if os.path.isfile(path) else None
            meta = {
                "name": ds.get("name", source),
                "description": ds.get("description", ""),
                "source": source,
                "deletable": False,
                "has_embeddings": _resolve_emb_source(source) is not None,
            }
            if os.path.isfile(path):
                meta["file_size_bytes"] = os.path.getsize(path)
            if lfs_info is not None:
                meta["is_lfs_pointer"] = True
                if "size" in lfs_info:
                    meta["lfs_size_bytes"] = lfs_info["size"]
            out.append(meta)

        for rec in db.list_created_datasets():
            path = os.path.normpath(os.path.join(DATA_ROOT, rec["source"]))
            if not os.path.isfile(path):
                continue
            out.append(
                {
                    "id": rec["id"],
                    "name": rec["name"],
                    "description": rec["description"],
                    "source": rec["source"],
                    "parent_source": rec["parent_source"],
                    "row_count": rec["row_count"],
                    "deletable": True,
                    "has_embeddings": _resolve_emb_source(rec["source"]) is not None,
                    "file_size_bytes": os.path.getsize(path),
                }
            )
        return jsonify(out)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


def _detect_path_col(cols: Sequence[str]) -> str | None:
    for c in cols:
        lc = c.lower()
        if lc == "image_path" or ("image" in lc and "path" in lc):
            return c
    return next((c for c in cols if "path" in c.lower()), cols[0] if cols else None)


def _slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_").lower()
    return s or "dataset"


@app.post("/api/datasets")
def create_dataset() -> ResponseReturnValue:
    """Carve a child dataset from a parent's selection.

    Body: {name, description, parent_source, image_names:[...]}. Writes a child CSV
    (subset of the parent rows by image filename) and registers it; embeddings are
    inherited from the parent (resolved at compute time, not copied).
    """
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    description = (body.get("description") or "").strip()
    parent_source = (body.get("parent_source") or "").strip()
    image_names = body.get("image_names") or []
    if not name:
        return jsonify({"error": "name required"}), 400
    if not parent_source or os.path.isabs(parent_source) or ".." in parent_source:
        return jsonify({"error": "invalid parent_source"}), 400
    if not isinstance(image_names, list) or not image_names:
        return jsonify({"error": "image_names required"}), 400

    parent_path = os.path.normpath(os.path.join(DATA_ROOT, parent_source))
    if not os.path.isfile(parent_path):
        return jsonify({"error": "parent dataset not found"}), 404

    # Read the parent CSV (full columns) and keep rows whose image filename is in the selection.
    with open(parent_path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = list(reader)
    col = _detect_path_col(cols)
    wanted = {os.path.basename(str(x)) for x in image_names}
    kept = [r for r in rows if col and os.path.basename(str(r.get(col, ""))) in wanted]
    if not kept:
        return jsonify({"error": "no matching rows"}), 400

    # Unique child path under the writable user-data dir
    base_slug = _slugify(name)
    rel = f"{USER_DATA_REL}/{base_slug}.csv"
    n = 2
    while os.path.isfile(os.path.normpath(os.path.join(DATA_ROOT, rel))) or db.get_created_dataset_by_source(rel):
        rel = f"{USER_DATA_REL}/{base_slug}_{n}.csv"
        n += 1
    out_path = os.path.normpath(os.path.join(DATA_ROOT, rel))
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(kept)

    emb_source = _resolve_emb_source(parent_source)  # None if the parent has no embeddings
    rec = db.create_dataset_record(name, description, rel, parent_source, emb_source, len(kept))
    return jsonify(
        {
            "id": rec["id"],
            "name": rec["name"],
            "description": rec["description"],
            "source": rec["source"],
            "parent_source": rec["parent_source"],
            "row_count": rec["row_count"],
            "deletable": True,
            "has_embeddings": bool(rec["emb_source"]),
        }
    ), 201


def _remove_file(path: str) -> None:
    with contextlib.suppress(OSError):
        os.remove(path)


@app.delete("/api/datasets/<int:ds_id>")
def remove_dataset(ds_id: int) -> ResponseReturnValue:
    """Delete a user-created dataset. Confi datasets are not deletable.

    A row-subset child dataset (parent_source set) only removes its CSV — it inherits
    the parent's embeddings and was never staged from its own images. A dataset built
    from images (parent_source is None) also removes its embeddings JSON, staged
    images folder, and provenance sidecars.
    """
    rec = db.delete_created_dataset(ds_id)
    if rec is None:
        return jsonify({"error": "dataset not found"}), 404
    src = rec["source"]
    if src and src.startswith(USER_DATA_REL + "/") and ".." not in src:
        csv_path = os.path.normpath(os.path.join(DATA_ROOT, src))
        _remove_file(csv_path)
        _remove_file(csv_path + ".provenance.json")
        if rec["parent_source"] is None:
            stem = os.path.splitext(csv_path)[0]
            _remove_file(stem + ".json")
            _remove_file(stem + ".json.provenance.json")
            images_dir = stem + "_images"
            if os.path.isdir(images_dir):
                shutil.rmtree(images_dir, ignore_errors=True)
    # Drop any cached compute results for this source
    for key in [k for k in _COMPUTE_CACHE if isinstance(k, tuple) and len(k) > 1 and k[1] == src]:
        _COMPUTE_CACHE.pop(key, None)
    return ("", 204)


# ── Dataset build (upload / demo) ────────────────────────────────────────────────
@app.get("/api/datasets/demos")
def list_demos() -> ResponseReturnValue:
    """Return the static registry of one-click demo datasets.

    Whether a given demo has already been built is NOT resolved here — the frontend
    checks GET /api/datasets for an existing dataset whose name matches a registry
    entry's name.
    """
    return jsonify(dataset_builder.DEMOS)


@app.post("/api/datasets/build")
def build_dataset() -> ResponseReturnValue:
    """Start a background dataset build.

    This pipeline never computes embeddings — bring your own. multipart form fields:
    - source_type: 'upload' | 'demo'
    - upload mode also needs: name, description (optional), images (one or more
      files, each filename carrying its uploaded relative path for the cluster
      convention), annotations (optional, zero or more COCO-format JSON label
      files, each filename carrying its original relative path — matched to an
      image by basename stem, see tools/staging.py), and embeddings (optional, a
      single pre-computed embeddings JSON — {"embeddings": {"<basename>": [floats]}}
      — staged as-is; the built dataset has no embeddings if omitted)
    - demo mode also needs: demo_key (must name an enabled entry in
      GET /api/datasets/demos); demo builds never have embeddings
    Returns {job_id} (202) immediately; poll GET /api/datasets/build/<job_id>.
    """
    source_type = (request.form.get("source_type") or "").strip()
    try:
        if source_type == "upload":
            name = request.form.get("name", "")
            description = request.form.get("description", "")
            images = [
                dataset_builder.staging_mod.UploadedImage(relative_path=fs.filename or "image", data=fs.read())
                for fs in request.files.getlist("images")
            ]
            annotations = [
                dataset_builder.staging_mod.UploadedImage(relative_path=fs.filename or "annotation", data=fs.read())
                for fs in request.files.getlist("annotations")
            ]
            embeddings_fs = request.files.get("embeddings")
            embeddings_file = embeddings_fs.read() if embeddings_fs else None
            job_id = dataset_builder.start_upload_build(images, annotations, embeddings_file, name, description)
        elif source_type == "demo":
            demo_key = (request.form.get("demo_key") or "").strip()
            job_id = dataset_builder.start_demo_build(demo_key)
        else:
            return jsonify({"error": "source_type must be 'upload' or 'demo'"}), 400
    except dataset_builder.BuildValidationError as exc:
        return jsonify({"error": str(exc)}), 400
    except dataset_builder.BuildInProgressError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify({"job_id": job_id}), 202


@app.get("/api/datasets/build/<job_id>")
def get_build_job(job_id: str) -> ResponseReturnValue:
    """Return the status of a background dataset-build job."""
    job = dataset_builder.get_job(job_id)
    if job is None:
        return jsonify({"error": "job not found"}), 404
    return jsonify(job)


@app.get("/api/dataset/csv")
def get_dataset_csv() -> ResponseReturnValue:
    """Stream a dataset CSV from local disk. `source` is the path from confi.yaml."""
    source = request.args.get("source", "").strip()
    if not source:
        return jsonify({"error": "source required"}), 400
    # Prevent path traversal
    if os.path.isabs(source) or ".." in source:
        return jsonify({"error": "invalid source path"}), 400
    path = os.path.normpath(os.path.join(DATA_ROOT, source))
    if not os.path.isfile(path):
        return jsonify({"error": "dataset not found"}), 404
    lfs_info = _lfs_pointer_info(path)
    if lfs_info is not None:
        return jsonify(
            {
                "error": "dataset is a Git LFS pointer; run git lfs pull to download the real CSV",
                "lfs_size_bytes": lfs_info.get("size"),
            }
        ), 409
    return send_file(path, mimetype="text/csv", as_attachment=False)


@app.get("/api/dataset/embeddings")
def get_dataset_embeddings() -> ResponseReturnValue:
    """Stream an embeddings JSON for a dataset. `source` is the CSV path from confi.yaml.

    Base embeddings = <stem>.json. An optional `variant` (model name) selects a
    comparison file <stem>_<variant>.json. 404 if it does not exist.
    """
    source = request.args.get("source", "").strip()
    variant = request.args.get("variant", "").strip()
    if not source:
        return jsonify({"error": "source required"}), 400
    if os.path.isabs(source) or ".." in source:
        return jsonify({"error": "invalid source path"}), 400
    if variant and not re.fullmatch(r"[A-Za-z0-9_-]+", variant):
        return jsonify({"error": "invalid variant"}), 400
    stem = os.path.splitext(_resolve_emb_source(source) or source)[0]  # child → parent's embeddings
    json_source = f"{stem}_{variant}.json" if variant else f"{stem}.json"
    path = os.path.normpath(os.path.join(DATA_ROOT, json_source))
    if not os.path.isfile(path):
        return jsonify({"error": "embeddings not found"}), 404
    lfs_info = _lfs_pointer_info(path)
    if lfs_info is not None:
        return jsonify(
            {
                "error": "embeddings file is a Git LFS pointer; run git lfs pull to download the real JSON",
                "lfs_size_bytes": lfs_info.get("size"),
            }
        ), 409
    return send_file(path, mimetype="application/json", as_attachment=False)


@app.get("/api/dataset/embeddings/variants")
def list_embedding_variants() -> ResponseReturnValue:
    """List comparison embedding files for a dataset — siblings named <stem>_<model>.json.

    Excludes the base <stem>.json. Returns [{name, variant}].
    """
    source = request.args.get("source", "").strip()
    if not source:
        return jsonify({"error": "source required"}), 400
    if os.path.isabs(source) or ".." in source:
        return jsonify({"error": "invalid source path"}), 400
    stem = os.path.splitext(_resolve_emb_source(source) or source)[0]  # child → parent's variants
    base = os.path.basename(stem)  # datalake_4k
    pattern = os.path.normpath(os.path.join(DATA_ROOT, f"{stem}_*.json"))
    variants = []
    for f in sorted(glob.glob(pattern)):
        fname = os.path.basename(f)  # datalake_4k_seedall22.json
        variant = fname[len(base) + 1 : -len(".json")]  # seedall22
        if variant:
            variants.append({"name": variant, "variant": variant})
    return jsonify(variants)


# ── Embedding compute (projection / clustering / comparison) ────────────────────
# Heavy embedding math runs HERE, not in the browser. Endpoints return only small arrays
# (3D positions, cluster labels, neighbour indices). Results are cached in-process by key.
_COMPUTE_CACHE: dict = {}
_PCA_PRE_DIMS = 30


def _safe_source(source: str) -> str:
    if not source or os.path.isabs(source) or ".." in source:
        raise ValueError("invalid source path")
    return source


def _emb_json_path(source: str, variant: str | None) -> str:
    # Child datasets inherit the parent's embeddings file; align to the child's own CSV rows.
    stem = os.path.splitext(_resolve_emb_source(source) or source)[0]
    rel = f"{stem}_{variant}.json" if variant else f"{stem}.json"
    return os.path.normpath(os.path.join(DATA_ROOT, rel))


def _dataset_paths(source: str) -> list[str]:
    """Ordered image-path values from the dataset CSV + the detected path column."""
    key = ("paths", source)
    if key in _COMPUTE_CACHE:
        return _COMPUTE_CACHE[key]
    path = os.path.normpath(os.path.join(DATA_ROOT, source))
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = list(reader)
    col = None
    for c in cols:
        lc = c.lower()
        if lc == "image_path" or ("image" in lc and "path" in lc):
            col = c
            break
    if col is None:
        col = next((c for c in cols if "path" in c.lower()), cols[0] if cols else None)
    paths = [r.get(col, "") if col else "" for r in rows]
    _COMPUTE_CACHE[key] = paths
    return paths


def _load_emb_by_name(source: str, variant: str | None) -> dict:
    """Parse an embeddings JSON → {image_basename: vector}. Not cached (can be very large)."""
    path = _emb_json_path(source, variant)
    if not os.path.isfile(path):
        raise FileNotFoundError("embeddings not found")
    if _lfs_pointer_info(path) is not None:
        raise FileNotFoundError("embeddings file is a Git LFS pointer; run git lfs pull")
    with open(path) as f:
        data = json.load(f)
    emb = data.get("embeddings", data)
    return {os.path.basename(str(k)): v for k, v in emb.items()}


def _base_matrix(source: str, variant: str | None) -> np.ndarray:
    """[n, d] aligned to CSV row order (rows without an embedding → zeros). Cached as a small array."""
    key = ("mat", source, variant)
    if key in _COMPUTE_CACHE:
        return _COMPUTE_CACHE[key]
    paths = _dataset_paths(source)
    by_name = _load_emb_by_name(source, variant)
    dim = next((len(v) for v in by_name.values() if v), 0)
    matrix = np.zeros((len(paths), dim), dtype=np.float32)
    for i, p in enumerate(paths):
        v = by_name.get(os.path.basename(str(p)))
        if v is not None and len(v) == dim:
            matrix[i] = v
    _COMPUTE_CACHE[key] = matrix
    return matrix


@app.get("/api/embedding/projection")
def embedding_projection() -> ResponseReturnValue:
    """3D projection of a dataset's embeddings. method = pca | tsne | lle. Returns {positions:[[x,y,z]]}."""
    try:
        source = _safe_source(request.args.get("source", "").strip())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    method = request.args.get("method", "pca").lower()
    variant = request.args.get("variant", "").strip() or None
    cache_key = ("proj", source, variant, method)
    if cache_key in _COMPUTE_CACHE:
        return jsonify({"positions": _COMPUTE_CACHE[cache_key]})
    try:
        matrix = _base_matrix(source, variant)
    except (FileNotFoundError, ValueError) as e:
        return jsonify({"error": str(e)}), 404
    n, d = matrix.shape
    if n < 3 or d < 1:
        return jsonify({"positions": [[0.0, 0.0, 0.0]] * n})

    if method == "pca":
        proj = PCA(n_components=3, random_state=0).fit_transform(matrix)
    else:
        pre = (
            PCA(n_components=min(_PCA_PRE_DIMS, d), random_state=0).fit_transform(matrix)
            if d > _PCA_PRE_DIMS
            else matrix
        )
        if method == "tsne":
            perp = max(5, min(30, (n - 1) // 3))
            proj = TSNE(n_components=3, init="pca", perplexity=perp, random_state=0).fit_transform(pre)
        elif method == "lle":
            proj = LocallyLinearEmbedding(n_components=3, n_neighbors=min(12, n - 1), random_state=0).fit_transform(pre)
        else:
            return jsonify({"error": "unknown method"}), 400
    positions = np.asarray(proj, dtype=float).tolist()
    _COMPUTE_CACHE[cache_key] = positions
    return jsonify({"positions": positions})


@app.get("/api/embedding/clusters")
def embedding_clusters() -> ResponseReturnValue:
    """Spherical k-means (cosine) on the FULL embedding space. Returns {labels:[...]}."""
    try:
        source = _safe_source(request.args.get("source", "").strip())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    variant = request.args.get("variant", "").strip() or None
    try:
        k = max(2, min(50, int(request.args.get("k", "20"))))
    except ValueError:
        k = 20
    cache_key = ("clust", source, variant, k)
    if cache_key in _COMPUTE_CACHE:
        return jsonify({"labels": _COMPUTE_CACHE[cache_key]})
    try:
        matrix = _base_matrix(source, variant)
    except (FileNotFoundError, ValueError) as e:
        return jsonify({"error": str(e)}), 404
    n = matrix.shape[0]
    if n < 2:
        return jsonify({"labels": [0] * n})
    normalized = normalize(matrix)  # L2-normalize → KMeans on the unit sphere = cosine
    labels = KMeans(n_clusters=min(k, n), n_init=cast(Any, 4), random_state=0).fit_predict(normalized)
    out = labels.astype(int).tolist()
    _COMPUTE_CACHE[cache_key] = out
    return jsonify({"labels": out})


def _compare_matrix(comp: dict[str, list[float]]) -> tuple[list[str], np.ndarray]:
    """Return names and stacked vectors for `comp` entries that have a full-length embedding.

    Empty lists/array when no comparison item has a usable embedding.
    """
    dim = next((len(v) for v in comp.values() if v), 0)
    names = [nm for nm, v in comp.items() if v and len(v) == dim]
    if not names:
        return [], np.empty((0, dim), dtype=np.float32)
    return names, np.array([comp[nm] for nm in names], dtype=np.float32)


def _neighbor_rows_and_weights(
    row_of: list[int], idx: np.ndarray, dist: np.ndarray, i: int, kk: int
) -> tuple[list[int], list[float]]:
    """Return original-row indices and inverse-distance weights for comparison item `i`'s neighbours.

    Drops the self-match and any neighbour with no corresponding original row; falls
    back to the item's own row (weight 1.0) when nothing else is available.
    """
    neigh_rows: list[int] = []
    weights: list[float] = []
    for jpos in range(kk):
        j = int(idx[i][jpos])
        if j == i:  # drop self
            continue
        r = row_of[j]
        if r < 0:
            continue
        neigh_rows.append(r)
        weights.append(1.0 / (float(np.sqrt(dist[i][jpos])) + 1e-6))
    if not neigh_rows and row_of[i] >= 0:  # fall back to the item's own row if known
        neigh_rows, weights = [row_of[i]], [1.0]
    return neigh_rows, weights


def _compare_result(names: list[str], matrix: np.ndarray, name_to_row: dict[str, int]) -> dict[str, list]:
    """Return neighbour rows/weights/origIdx for every comparison item in `names`."""
    reduced = (
        PCA(n_components=min(_PCA_PRE_DIMS, matrix.shape[1]), random_state=0).fit_transform(matrix)
        if matrix.shape[1] > _PCA_PRE_DIMS
        else matrix
    )
    kk = min(7, len(names))  # self + 6 neighbours
    nn = NearestNeighbors(n_neighbors=kk).fit(reduced)
    dist, idx = nn.kneighbors(reduced)

    row_of = [name_to_row.get(nm, -1) for nm in names]  # comparison item → original row
    nbr, w, orig_idx = [], [], []
    for i in range(len(names)):
        neigh_rows, weights = _neighbor_rows_and_weights(row_of, idx, dist, i, kk)
        if not neigh_rows:
            continue
        s = sum(weights) or 1.0
        nbr.append(neigh_rows)
        w.append([x / s for x in weights])
        orig_idx.append(row_of[i])
    return {"nbr": nbr, "w": w, "origIdx": orig_idx}


@app.get("/api/embedding/compare")
def embedding_compare() -> ResponseReturnValue:
    """Place a comparison model's items in the ORIGINAL layout.

    For each comparison image we take its nearest neighbours IN THE COMPARISON SPACE
    and map them to the original rows by IMAGE NAME, so the frontend can interpolate
    their on-screen positions. Returns {nbr, w, origIdx} (all by row index).
    """
    try:
        source = _safe_source(request.args.get("source", "").strip())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    variant = request.args.get("variant", "").strip()
    if not variant or not re.fullmatch(r"[A-Za-z0-9_-]+", variant):
        return jsonify({"error": "valid variant required"}), 400
    cache_key = ("cmp", source, variant)
    if cache_key in _COMPUTE_CACHE:
        return jsonify(_COMPUTE_CACHE[cache_key])

    try:
        paths = _dataset_paths(source)
        comp = _load_emb_by_name(source, variant)
    except (FileNotFoundError, ValueError) as e:
        return jsonify({"error": str(e)}), 404

    name_to_row: dict[str, int] = {}
    for i, p in enumerate(paths):
        name_to_row.setdefault(os.path.basename(str(p)), i)

    names, matrix = _compare_matrix(comp)
    if not names:
        result = {"nbr": [], "w": [], "origIdx": []}
        _COMPUTE_CACHE[cache_key] = result
        return jsonify(result)

    result = _compare_result(names, matrix, name_to_row)
    _COMPUTE_CACHE[cache_key] = result
    return jsonify(result)


def main() -> None:
    """Run the Flask development server."""
    port = int(os.environ.get("FLASK_PORT", os.environ.get("API_PORT", "5050")))
    debug = os.environ.get("API_DEBUG", "false").strip().lower() in {"1", "true", "yes", "on"}
    app.run(debug=debug, port=port, host="0.0.0.0")


if __name__ == "__main__":
    main()
