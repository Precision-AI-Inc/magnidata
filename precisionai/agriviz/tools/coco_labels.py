# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Per-image COCO label parsing + deterministic polygon rasterization.

The dataset stores one COCO JSON per image under ``<DATASET>/labels/<STEM>.json``:
  images:      [{id, file_name, height, width}]   (exactly one entry)
  annotations: [{id, image_id, category_id, segmentation: [[x,y,x,y,...], ...]}]
               one annotation == one instance; segmentation is POLYGON format
               (a list of flat coordinate lists, one per polygon part).
  categories:  [{id, name}]   (id 0 == "background")

Rasterization uses ``PIL.ImageDraw.polygon`` (deterministic, raster-order fill,
no extra dependency). pycocotools' ``frPyObjects``/``decode`` is the drop-in
alternative if RLE masks ever appear.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

# class_map encoding: 0 = unlabeled, otherwise (category_id + 1). So background
# (category 0) -> 1, foreground classes -> 2.. . Keeps a clean "unlabeled" sentinel
# while staying in uint8 (supports up to 254 categories; datasets here have ~27).
CLASS_MAP_OFFSET = 1
BACKGROUND_CATEGORY_ID = 0


@dataclass
class CocoLabels:
    width: int
    height: int
    categories: dict[int, str]              # category_id -> name
    annotations: list[dict]                 # [{id, category_id, polygons:[[x,y,...]]}]


def parse(json_path: str) -> CocoLabels:
    with open(json_path) as f:
        doc = json.load(f)

    images = doc.get("images") or []
    if not images:
        raise ValueError("COCO file has no images entry")
    img0 = images[0]
    width, height = int(img0["width"]), int(img0["height"])

    categories = {int(c["id"]): str(c.get("name", "")) for c in doc.get("categories", [])}

    anns = []
    for a in doc.get("annotations", []):
        seg = a.get("segmentation")
        polygons = _normalize_polygons(seg)
        if not polygons:
            continue
        anns.append({
            "id": int(a.get("id", 0)),
            "category_id": int(a.get("category_id", BACKGROUND_CATEGORY_ID)),
            "polygons": polygons,
        })
    return CocoLabels(width=width, height=height, categories=categories, annotations=anns)


def _normalize_polygons(seg) -> list[list[float]]:
    """Return a list of flat [x0,y0,x1,y1,...] polygon parts (>=3 points each).

    Accepts COCO polygon format ([[...], ...]) and the occasional single flat list.
    RLE (dict / counts) is not supported here — would need pycocotools.
    """
    if not seg or isinstance(seg, dict):
        return []
    parts = seg if (isinstance(seg, list) and seg and isinstance(seg[0], (list, tuple))) else [seg]
    out = []
    for p in parts:
        if isinstance(p, (list, tuple)) and len(p) >= 6:   # >=3 (x,y) points
            out.append([float(v) for v in p])
    return out


def _scale(wh: tuple[int, int], labels: CocoLabels) -> tuple[float, float]:
    """Polygon-coord scale factors to map label space -> target (w, h)."""
    w, h = wh
    return w / labels.width, h / labels.height


def _pts(poly: list[float], sx: float, sy: float):
    pts = [(poly[i] * sx, poly[i + 1] * sy) for i in range(0, len(poly) - 1, 2)]
    return pts if len(pts) >= 3 else None


def _draw_polygons(polygons, wh, sx, sy, fill: int) -> Image.Image:
    """Render polygon parts onto a fresh 'L' image (0 background, ``fill`` inside)."""
    img = Image.new("L", wh, 0)
    d = ImageDraw.Draw(img)
    for poly in polygons:
        pts = _pts(poly, sx, sy)
        if pts:
            d.polygon(pts, fill=fill, outline=fill)
    return img


def rasterize_class_map(labels: CocoLabels, target_wh: tuple[int, int] | None = None) -> np.ndarray:
    """(H, W) uint8 label map: 0=unlabeled, category_id+1 elsewhere.

    Rendered at ``target_wh`` (the actual image resolution) by scaling polygon coords,
    so it stays aligned to the loaded image even when that is a downscaled thumbnail.
    Annotations are drawn in file order; later ones overwrite earlier on overlap.
    """
    wh = target_wh or (labels.width, labels.height)
    sx, sy = _scale(wh, labels)
    canvas = Image.new("L", wh, 0)
    drawer = ImageDraw.Draw(canvas)
    for a in labels.annotations:
        val = a["category_id"] + CLASS_MAP_OFFSET
        for poly in a["polygons"]:
            pts = _pts(poly, sx, sy)
            if pts:
                drawer.polygon(pts, fill=val, outline=val)
    return np.asarray(canvas, dtype=np.uint8)


def iter_instance_masks(labels: CocoLabels, target_wh: tuple[int, int] | None = None):
    """Yield (index, category_id, mask_bool) for each annotation/instance.

    ``mask_bool`` is a fresh (H, W) bool array at ``target_wh``. Iterating (rather than
    building one giant int label map) keeps memory bounded on the large images while
    letting callers accumulate overlap/area/colour in one pass.
    """
    wh = target_wh or (labels.width, labels.height)
    sx, sy = _scale(wh, labels)
    for idx, a in enumerate(labels.annotations):
        mask = np.asarray(_draw_polygons(a["polygons"], wh, sx, sy, fill=1), dtype=bool)
        yield idx, a["category_id"], mask


def foreground_from_class_map(class_map: np.ndarray) -> np.ndarray:
    """Boolean foreground = labeled AND not background."""
    bg_val = BACKGROUND_CATEGORY_ID + CLASS_MAP_OFFSET
    return (class_map != 0) & (class_map != bg_val)


def background_from_class_map(class_map: np.ndarray) -> np.ndarray:
    return class_map == (BACKGROUND_CATEGORY_ID + CLASS_MAP_OFFSET)
