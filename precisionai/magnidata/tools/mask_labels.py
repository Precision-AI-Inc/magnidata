# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Class maps from colour segmentation mask PNGs, for datasets without COCO labels.

A mask is one PNG per image under ``<DATASET>/masks/<STEM>.png``, coloured per class with
near-black as background — the layout AgriStress-500 and its tiles ship, and the one the
reference trainer reads. Each distinct non-background colour is one class. The class map
uses the same encoding as ``coco_labels.rasterize_class_map``, so the coverage and class
features are computed the same way for both label sources. A class-coloured mask does not
separate touching objects of one class, so it yields no per-instance features.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from . import coco_labels

# Same near-black rule as the trainer's palette (training/data.py BACKGROUND_MAX_CHANNEL).
BACKGROUND_MAX_CHANNEL = 10


def load_class_map(mask_path: str, target_wh: tuple[int, int] | None = None) -> np.ndarray:
    """Return an (H, W) int32 class map decoded from a colour mask PNG.

    Background pixels get the background sentinel and each foreground colour gets its own
    value above it, assigned in ascending packed-RGB order so the mapping is deterministic.
    The mask is resized with nearest-neighbour to ``target_wh`` (the loaded image's size,
    which is smaller when a thumbnail was read) so colours stay exact.

    Parameters
    ----------
    mask_path : str
        Path to the mask PNG.
    target_wh : tuple of int or None, default None
        (width, height) to resample to; None keeps the mask's own size.

    Returns
    -------
    np.ndarray
        Class map in ``coco_labels`` encoding (0 is never produced: every pixel is labeled).
    """
    with Image.open(mask_path) as im:
        mask = im.convert("RGB")
    if target_wh is not None and mask.size != target_wh:
        mask = mask.resize(target_wh, Image.Resampling.NEAREST)
    rgb = np.asarray(mask, dtype=np.int64)
    packed = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    background = (rgb <= BACKGROUND_MAX_CHANNEL).all(axis=-1)

    bg_val = coco_labels.BACKGROUND_CATEGORY_ID + coco_labels.CLASS_MAP_OFFSET
    class_map = np.full(packed.shape, bg_val, dtype=np.int32)
    colours, inverse = np.unique(packed[~background], return_inverse=True)
    if colours.size:
        class_map[~background] = inverse.astype(np.int32) + bg_val + 1
    return class_map
