# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Image and colour-mask helpers shared by the annotation overlay and the training prediction overlay."""

import io

import numpy as np
from PIL import Image

# Background threshold: pixels whose R+G+B sum is below this are treated as background and
# rendered transparent in the mask/overlay views.
BG_THRESHOLD = 30


def img_from_bytes(data: bytes, max_size: int = 1600) -> Image.Image:
    """Decode an image, shrinking it so its longer side is at most ``max_size``."""
    img = Image.open(io.BytesIO(data))
    if max(img.size) > max_size:
        img = img.copy()
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return img


def mask_as_rgba(mask_data: bytes, max_size: int = 0) -> Image.Image:
    """Load an RGB(A) mask PNG as RGBA with near-black background transparent and classes opaque.

    The masks are pre-coloured: each segmentation class already has its own RGB colour and
    black (0, 0, 0) means background.
    """
    mask = Image.open(io.BytesIO(mask_data))
    if mask.mode != "RGBA":
        mask = mask.convert("RGBA")

    arr = np.array(mask, dtype=np.uint8)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    is_bg = (r.astype(np.uint16) + g + b) < BG_THRESHOLD
    arr[:, :, 3] = np.where(is_bg, 0, 255)

    result = Image.fromarray(arr, "RGBA")
    if max_size and max(result.size) > max_size:
        result.thumbnail((max_size, max_size), Image.Resampling.NEAREST)
    return result


def composite_mask(base: Image.Image, mask_rgba: Image.Image, alpha: float = 1.0) -> Image.Image:
    """Alpha-composite an RGBA mask over ``base`` at opacity ``alpha`` (0..1).

    A mask of a different size (e.g. a prediction at training resolution) is resized with
    nearest-neighbour so class colours stay exact.
    """
    base = base.convert("RGBA")
    if mask_rgba.size != base.size:
        mask_rgba = mask_rgba.resize(base.size, Image.Resampling.NEAREST)
    if alpha < 1.0:
        arr = np.array(mask_rgba)
        arr[:, :, 3] = (arr[:, :, 3].astype(np.float32) * alpha).astype(np.uint8)
        mask_rgba = Image.fromarray(arr, "RGBA")
    return Image.alpha_composite(base, mask_rgba)


def to_jpeg(img: Image.Image, quality: int = 88) -> bytes:
    """Encode an image as JPEG bytes."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def to_png(img: Image.Image) -> bytes:
    """Encode an image as PNG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
