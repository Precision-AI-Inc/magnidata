# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Interpretable, deterministic image-quality metrics (pure numpy, full-res).

These describe capture quality directly and are reproducible — unlike the learned
no-reference scores (NIMA/NIQE/BRISQUE in nima.py), which are perceptual but
biased toward consumer photography. All work on a float64 Rec.601 luma (0..255)
and/or the uint8 RGB; every reduction is guarded.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6

# Immerkaer (1996) second-derivative mask: cancels smooth content AND edges, so the
# residual is dominated by noise -> a no-reference Gaussian-noise sigma estimate.
_NOISE_N = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


def _conv3_abs_sum(luma: np.ndarray, k: np.ndarray) -> np.ndarray:
    """|3x3 conv| over the interior, built from 9 shifted multiplies (no big views)."""
    acc = np.zeros((luma.shape[0] - 2, luma.shape[1] - 2), dtype=np.float64)
    for di in range(3):
        for dj in range(3):
            acc += k[di, dj] * luma[di : di + luma.shape[0] - 2, dj : dj + luma.shape[1] - 2]
    return np.abs(acc)


def noise_sigma(luma: np.ndarray) -> float:
    """Estimate Gaussian-noise sigma via the Immerkaer second-derivative residual."""
    h, w = luma.shape
    if h < 3 or w < 3:
        return 0.0
    s = _conv3_abs_sum(luma, _NOISE_N).sum()
    return float(np.sqrt(np.pi / 2.0) / (6.0 * (w - 2) * (h - 2)) * s)


def tenengrad(luma: np.ndarray) -> float:
    """Mean Sobel gradient magnitude squared — a focus/sharpness measure."""
    if luma.shape[0] < 3 or luma.shape[1] < 3:
        return 0.0
    gx = luma[:-2, 2:] + 2 * luma[1:-1, 2:] + luma[2:, 2:] - luma[:-2, :-2] - 2 * luma[1:-1, :-2] - luma[2:, :-2]
    gy = luma[2:, :-2] + 2 * luma[2:, 1:-1] + luma[2:, 2:] - luma[:-2, :-2] - 2 * luma[:-2, 1:-1] - luma[:-2, 2:]
    return float((gx * gx + gy * gy).mean())


def rms_contrast(luma: np.ndarray) -> float:
    """Std-dev of normalized luminance (0..1) — flat/hazy vs punchy capture."""
    return float((luma / 255.0).std())


def dynamic_range(luma: np.ndarray) -> float:
    """(p99 - p1) of luminance, normalized to 0..1 — clipping / low-light crush."""
    p1, p99 = np.percentile(luma, [1, 99], method="linear")
    return float((p99 - p1) / 255.0)


def luma_entropy(luma: np.ndarray) -> float:
    """Shannon entropy (bits) of the 256-bin luminance histogram."""
    hist, _ = np.histogram(luma, bins=256, range=(0, 255))
    p = hist.astype(np.float64)
    total = p.sum()
    if total == 0:
        return 0.0
    p = p[p > 0] / total
    return float(-(p * np.log2(p)).sum())


def colorfulness(rgb: np.ndarray) -> float:
    """Hasler & Susstrunk (2003) colorfulness — vividness; strong for saturated, high-chroma content."""
    r, g, b = (rgb[:, :, i].astype(np.float64) for i in range(3))
    rg = r - g
    yb = 0.5 * (r + g) - b
    std = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std + 0.3 * mean)


def compute(rgb: np.ndarray, luma: np.ndarray) -> dict:
    """Compute the full deterministic quality panel: noise, sharpness, contrast, colour."""
    return {
        "noise_sigma": noise_sigma(luma),
        "tenengrad": tenengrad(luma),
        "rms_contrast": rms_contrast(luma),
        "dynamic_range": dynamic_range(luma),
        "luma_entropy": luma_entropy(luma),
        "colorfulness": colorfulness(rgb),
    }
