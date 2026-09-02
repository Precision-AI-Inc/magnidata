# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""COCO-derived instance / entanglement / look-alike-colour features (deterministic).

- Instance stats come from per-annotation polygon masks (one annotation = one object).
- Class colour / distribution come from the non-overlapping class map (each pixel
  belongs to exactly one class), which is the stable basis for look-alike-class
  colour similarity.
"""
from __future__ import annotations

import numpy as np

from . import coco_labels as cl

MIN_INSTANCE_PX = 64        # below this an annotation is treated as a fragment
EPS = 1e-6


# ── sRGB -> CIE Lab (D65) for small (N,3) arrays of mean class colours ────────────
def _rgb_to_lab(rgb_255: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_255, dtype=np.float64) / 255.0
    m = rgb > 0.04045
    lin = np.where(m, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    # sRGB -> XYZ (D65)
    mat = np.array([[0.4124564, 0.3575761, 0.1804375],
                    [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ mat.T
    white = np.array([0.95047, 1.0, 1.08883])
    xyz = xyz / white
    d = 6.0 / 29.0
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d * d) + 4.0 / 29.0)
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=1)


def instance_metrics(labels: cl.CocoLabels, rgb: np.ndarray, class_map: np.ndarray) -> dict:
    h, w = class_map.shape
    total = float(h * w)

    # ── Instance areas + overlap (foreground annotations only) ────────────────────
    covered = np.zeros((h, w), dtype=np.uint16)
    areas: list[int] = []
    for _idx, cat_id, mask in cl.iter_instance_masks(labels, target_wh=(w, h)):
        if cat_id == cl.BACKGROUND_CATEGORY_ID:
            continue
        area = int(mask.sum())
        if area == 0:
            continue
        areas.append(area)
        covered += mask                       # accumulate to find multiply-covered pixels

    areas_arr = np.asarray(areas, dtype=np.float64)
    fg_px = float((covered >= 1).sum())
    overlap_px = float((covered > 1).sum())
    instance_count = int(areas_arr.size)
    real_count = int((areas_arr >= MIN_INSTANCE_PX).sum())
    area_ratios = areas_arr / total if total else areas_arr

    out = {
        "instance_count": instance_count,
        "real_instance_count": real_count,
        "mean_instance_area_ratio": float(area_ratios.mean()) if area_ratios.size else 0.0,
        "instance_area_std": float(area_ratios.std()) if area_ratios.size else 0.0,
        "overlap_ratio": overlap_px / fg_px if fg_px else 0.0,
    }

    # ── Class colour (look-alike) + class distribution from the class map ─────────
    bg_val = cl.BACKGROUND_CATEGORY_ID + cl.CLASS_MAP_OFFSET
    fg_vals = [int(v) for v in np.unique(class_map) if v != 0 and v != bg_val]
    fg_vals.sort()                              # deterministic class ordering

    class_px = []
    mean_colors = []
    for v in fg_vals:
        m = class_map == v
        cnt = int(m.sum())
        if cnt == 0:
            continue
        class_px.append(cnt)
        mean_colors.append(rgb[m].astype(np.float64).mean(axis=0))

    class_px_arr = np.asarray(class_px, dtype=np.float64)
    out["class_count"] = int(class_px_arr.size)

    if class_px_arr.size:
        frac = class_px_arr / class_px_arr.sum()
        out["smallest_class_ratio"] = float(frac.min())
        out["class_entropy"] = float(-(frac * np.log2(frac + EPS)).sum())
    else:
        out["smallest_class_ratio"] = 0.0
        out["class_entropy"] = 0.0

    if len(mean_colors) >= 2:
        lab = _rgb_to_lab(np.vstack(mean_colors))
        n = lab.shape[0]
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(float(np.linalg.norm(lab[i] - lab[j])))
        dists_arr = np.asarray(dists, dtype=np.float64)
        out["mean_pairwise_color_dist"] = float(dists_arr.mean())
        out["interclass_color_sim"] = float(1.0 / (1.0 + dists_arr.min()))   # high = mimic risk
    else:
        out["mean_pairwise_color_dist"] = 0.0
        out["interclass_color_sim"] = 0.0      # single (or no) foreground class => no mimic

    return out
