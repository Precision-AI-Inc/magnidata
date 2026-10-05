# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Segmentation training data: sample discovery, colour-mask palette, splits and dataset.

Masks are colour PNGs (near-black is background), so each run derives a colour → class
index palette, logs it with the run and decodes masks through it. Splits are decided by a
hash of the image name, so adding rows to a dataset never reshuffles existing samples.
"""

import csv
import hashlib
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from precisionai.magnidata.training import contract
from precisionai.magnidata.training.dataset_check import find_mask, resolve_under

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    raise ImportError("torch is required for training data: pip install -e '.[ml]'") from None

IGNORE_INDEX = 255
BACKGROUND_MAX_CHANNEL = 10
PALETTE_COUNT_SIDE = 1024
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLIT_COLUMN = "split"


@dataclass(frozen=True)
class Sample:
    """One training example: an image, its colour mask and the CSV's own split label (if any)."""

    image_name: str
    image_path: Path
    mask_path: Path
    csv_split: str | None = None


@dataclass
class ResolveReport:
    """What `resolve_samples` found and what it had to skip."""

    total_rows: int = 0
    resolved: int = 0
    missing_image: list[str] = field(default_factory=list)
    missing_mask: list[str] = field(default_factory=list)
    duplicate_names: list[str] = field(default_factory=list)


def resolve_samples(dataset_csv: str | Path, data_root: str | Path) -> tuple[list[Sample], ResolveReport]:
    """Read a dataset CSV and pair every image with its mask.

    Rows whose image or mask file is missing, and rows repeating an earlier image
    basename (which would break per-sample joins), are skipped and counted in the report.
    """
    report = ResolveReport()
    samples: list[Sample] = []
    seen: set[str] = set()
    with open(dataset_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        column = contract.image_path_column(reader.fieldnames or [])
        if column is None:
            raise ValueError(f"{dataset_csv} has no columns")
        for row in reader:
            report.total_rows += 1
            rel = row.get(column) or ""
            name = contract.image_key(rel)
            image = resolve_under(data_root, rel)
            if image is None or not image.is_file():
                report.missing_image.append(name)
                continue
            mask = find_mask(rel, data_root, row)
            if mask is None:
                report.missing_mask.append(name)
                continue
            if name in seen:
                report.duplicate_names.append(name)
                continue
            seen.add(name)
            samples.append(Sample(name, image, mask, (row.get(SPLIT_COLUMN) or None)))
    report.resolved = len(samples)
    return samples, report


def _pack(rgb: np.ndarray) -> np.ndarray:
    flat = rgb.astype(np.int64)
    return (flat[..., 0] << 16) | (flat[..., 1] << 8) | flat[..., 2]


def _is_background(rgb: np.ndarray) -> np.ndarray:
    return np.asarray((rgb <= BACKGROUND_MAX_CHANNEL).all(axis=-1))


@dataclass(frozen=True)
class Palette:
    """Class index → RGB colour. Index 0 is background (black and near-black)."""

    colors: tuple[tuple[int, int, int], ...]

    @property
    def num_classes(self) -> int:
        """Return the number of classes, background included."""
        return len(self.colors) + 1

    def as_mapping(self) -> dict[int, dict[str, Any]]:
        """Return the ``magnidata/palette.json`` content: index → name and RGB."""
        out: dict[int, dict[str, Any]] = {0: {"name": "background", "rgb": [0, 0, 0]}}
        for i, rgb in enumerate(self.colors, start=1):
            out[i] = {"name": f"class_{i}", "rgb": list(rgb)}
        return out


def _load_small_rgb(path: str | Path) -> np.ndarray:
    """Load a mask as RGB, shrunk (nearest) so its longer side is at most `PALETTE_COUNT_SIDE`."""
    with Image.open(path) as im:
        rgb = im.convert("RGB")
    scale = PALETTE_COUNT_SIDE / max(rgb.size)
    if scale < 1:
        rgb = rgb.resize(
            (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))), Image.Resampling.NEAREST
        )
    return np.asarray(rgb)


def build_palette(mask_paths: Sequence[str | Path], max_classes: int = 32) -> Palette:
    """Derive the colour palette from masks: background is 0, other colours rank by pixel count.

    Masks are shrunk with nearest-neighbour sampling before counting, which keeps exact
    colours but makes counts approximate and can miss a colour covering only a few pixels
    of a very large mask; such pixels are later decoded as ``ignore_index``.

    Raises
    ------
    ValueError
        If the masks contain more than ``max_classes`` classes (background included), which
        usually means instance-coloured rather than class-coloured masks.
    """
    counts: Counter[int] = Counter()
    for path in mask_paths:
        arr = _load_small_rgb(path)
        arr = arr[~_is_background(arr)]
        keys, freq = np.unique(_pack(arr), return_counts=True)
        counts.update(dict(zip(keys.tolist(), freq.tolist(), strict=True)))
    ranked = sorted(counts, key=lambda k: (-counts[k], k))
    if len(ranked) + 1 > max_classes:
        raise ValueError(f"too many mask colours: {len(ranked) + 1} classes found, at most {max_classes} allowed")
    return Palette(tuple((k >> 16 & 255, k >> 8 & 255, k & 255) for k in ranked))


def decode_mask(rgb: np.ndarray, palette: Palette, ignore_index: int = IGNORE_INDEX) -> np.ndarray:
    """Convert an ``[H, W, 3]`` colour mask to ``[H, W]`` int64 class indices.

    Near-black is class 0; colours missing from the palette become ``ignore_index``.
    """
    packed = _pack(rgb)
    out = np.full(packed.shape, ignore_index, dtype=np.int64)
    if palette.colors:
        keys = _pack(np.array(palette.colors, dtype=np.int64))
        order = np.argsort(keys)
        sorted_keys = keys[order]
        pos = np.clip(np.searchsorted(sorted_keys, packed), 0, len(keys) - 1)
        hit = sorted_keys[pos] == packed
        out[hit] = order[pos[hit]] + 1
    out[_is_background(rgb)] = 0
    return out


def make_split(
    image_names: Sequence[str],
    val_fraction: float,
    seed: int,
    column_values: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Assign each image to ``train`` or ``val`` (or to the labels in ``column_values``).

    The hash of ``"<seed>:<name>"`` decides, so an image keeps its split when other rows are
    added. On tiny datasets the lowest/highest hash is moved so both splits are non-empty.

    Raises
    ------
    ValueError
        If ``column_values`` contains a label that is not train, val or test.
    """
    column_values = column_values or {}
    bad = sorted({v for v in column_values.values() if v not in contract.SPLITS})
    if bad:
        raise ValueError(f"invalid split label(s) in column: {', '.join(bad)}")

    score = {
        n: int(hashlib.sha1(f"{seed}:{n}".encode(), usedforsecurity=False).hexdigest(), 16) / 16**40
        for n in image_names
    }
    assignment = {n: column_values.get(n) or ("val" if score[n] < val_fraction else "train") for n in image_names}
    if not column_values and 0 < val_fraction < 1 and len(assignment) >= 2:
        ranked = sorted(score, key=lambda n: score[n])
        if "val" not in assignment.values():
            assignment[ranked[0]] = "val"
        elif "train" not in assignment.values():
            assignment[ranked[-1]] = "train"
    return assignment


class SegmentationDataset(Dataset):
    """Images and decoded masks of one split, resized and normalised for training.

    Parameters
    ----------
    samples : Sequence[Sample]
        Candidate samples, filtered to ``split`` using ``splits``.
    palette : Palette
        Colour → class mapping from `build_palette`.
    splits : Mapping[str, str]
        Image name → split label.
    split : str | None
        Keep only this split; all samples when None.
    image_size : int | tuple[int, int]
        Output size, ``(height, width)`` or a square side.
    augment : bool
        Random horizontal flip, applied to the train split only.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        palette: Palette,
        splits: Mapping[str, str],
        split: str | None = None,
        image_size: int | tuple[int, int] = 256,
        augment: bool = False,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        self.samples = [s for s in samples if split is None or splits.get(s.image_name) == split]
        self.palette = palette
        self.split = split
        self.size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.augment = augment and split == "train"
        self.ignore_index = ignore_index
        self._mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        """Return the number of samples in this split."""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return ``pixel_values`` [3,H,W] float32, ``labels`` [H,W] int64 and ``image_name``."""
        sample = self.samples[index]
        height, width = self.size
        with Image.open(sample.image_path) as im:
            image = np.asarray(im.convert("RGB").resize((width, height), Image.Resampling.BILINEAR))
        with Image.open(sample.mask_path) as im:
            mask = np.asarray(im.convert("RGB").resize((width, height), Image.Resampling.NEAREST))

        pixels = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
        labels = torch.from_numpy(decode_mask(mask, self.palette, self.ignore_index))
        pixels = (pixels - self._mean) / self._std
        if self.augment and random.random() < 0.5:
            pixels, labels = pixels.flip(-1), labels.flip(-1)
        return {"pixel_values": pixels, "labels": labels, "image_name": sample.image_name}


def collate_fn(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack tensors into a batch and keep ``image_name`` as a list."""
    return {
        "pixel_values": torch.stack([i["pixel_values"] for i in items]),
        "labels": torch.stack([i["labels"] for i in items]),
        "image_name": [i["image_name"] for i in items],
    }
