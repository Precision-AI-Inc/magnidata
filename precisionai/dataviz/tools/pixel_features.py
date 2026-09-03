# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Deterministic RGB-derived features: coverage, illumination, focus, white balance.

All reductions run in float64 on the FULL-RESOLUTION image (no resize/thumbnail).
Every division is guarded against empty masks. Values are rounded only at the output
boundary (see features.py), never mid-computation.
"""

from __future__ import annotations

import numpy as np

LUMA_COEFF = (0.299, 0.587, 0.114)  # Rec.601
OVEREXPOSE_T = 250  # luma >= this (0..255) => over-exposed
UNDEREXPOSE_T = 5  # luma <= this => under-exposed
SHADOW_LO, SHADOW_HI = 5, 60  # foreground luma in this band => shadow proxy
EPS = 1e-6


def to_luma(rgb: np.ndarray) -> np.ndarray:
    """(H, W) float64 Rec.601 luminance in 0..255."""
    r, g, b = (rgb[:, :, i].astype(np.float64) for i in range(3))
    return LUMA_COEFF[0] * r + LUMA_COEFF[1] * g + LUMA_COEFF[2] * b


def coverage(class_map_fg: np.ndarray, rgb: np.ndarray) -> dict:
    """Foreground pixel coverage from the (boolean) foreground mask, plus its mean green channel."""
    total = int(class_map_fg.size)
    fg_px = int(class_map_fg.sum())
    green = rgb[:, :, 1].astype(np.float64)
    fg_green_mean = float(green[class_map_fg].mean()) if fg_px else 0.0
    return {
        "annotated_px_count": fg_px,
        "annotation_ratio": fg_px / total if total else 0.0,
        "fg_green_mean": fg_green_mean,  # mean green-channel value over foreground
    }


def illumination(luma: np.ndarray, fg: np.ndarray) -> dict:
    """Compute whole-image over/under-exposure ratios and a foreground shadow proxy."""
    total = int(luma.size)
    over = float((luma >= OVEREXPOSE_T).sum()) / total if total else 0.0
    under = float((luma <= UNDEREXPOSE_T).sum()) / total if total else 0.0
    fg_px = int(fg.sum())
    if fg_px:
        fg_luma = luma[fg]
        shadow = float(((fg_luma > SHADOW_LO) & (fg_luma <= SHADOW_HI)).sum()) / fg_px
    else:
        shadow = 0.0
    return {
        "overexpose_ratio": over,
        "underexpose_ratio": under,
        # region-based shadow proxy over foreground; NOT the legacy edge-based metric
        "shadow_edge_ratio": shadow,
    }


def focus(luma: np.ndarray) -> dict:
    """Variance of the 3x3 Laplacian over interior pixels (higher = sharper)."""
    if luma.shape[0] < 3 or luma.shape[1] < 3:
        return {"blur_laplacian": 0.0}
    c = luma[1:-1, 1:-1]
    lap = luma[:-2, 1:-1] + luma[2:, 1:-1] + luma[1:-1, :-2] + luma[1:-1, 2:] - 4.0 * c
    return {"blur_laplacian": float(lap.var())}


def white_balance(rgb: np.ndarray) -> dict:
    """Gray-world estimate from pixels (distinct from EXIF wb_* in metadata)."""
    means = rgb.reshape(-1, 3).astype(np.float64).mean(axis=0)
    r, g, b = (float(means[0]), float(means[1]), float(means[2]))
    return {
        "wb_r_gain": g / (r + EPS),  # gain to neutralize R toward gray
        "wb_b_gain": g / (b + EPS),  # gain to neutralize B toward gray
        "wb_rb_ratio": r / (b + EPS),  # comparable to metadata wb_rb_ratio (pixel-derived)
    }
