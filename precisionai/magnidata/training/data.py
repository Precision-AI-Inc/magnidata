# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Segmentation training data: sample discovery, colour-mask palette, splits and dataset.

Masks are colour PNGs (near-black is background), so each run derives a colour → class
index palette, logs it with the run and decodes masks through it. Splits are decided by a
hash of the image name, so adding rows to a dataset never reshuffles existing samples.
"""

import csv
import hashlib
import multiprocessing
import os
import random
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
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
CACHE_DIRNAME = ".train_cache"
CACHE_MAX_MB_ENV = "TRAIN_CACHE_MAX_MB"
DEFAULT_CACHE_MAX_MB = 4096
WARM_PROGRESS_EVERY = 50


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


def _small_rgb(rgb: Image.Image) -> np.ndarray:
    """Shrink (nearest) so the longer side is at most `PALETTE_COUNT_SIDE`."""
    scale = PALETTE_COUNT_SIDE / max(rgb.size)
    if scale < 1:
        rgb = rgb.resize(
            (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))), Image.Resampling.NEAREST
        )
    return np.asarray(rgb)


def _load_small_rgb(path: str | Path) -> np.ndarray:
    """Load a mask as RGB, shrunk (nearest) so its longer side is at most `PALETTE_COUNT_SIDE`."""
    with Image.open(path) as im:
        return _small_rgb(im.convert("RGB"))


def _colour_counts(arr: np.ndarray) -> np.ndarray:
    """Return a ``[2, N]`` int64 array of packed non-background colours and their pixel counts."""
    arr = arr[~_is_background(arr)]
    keys, freq = np.unique(_pack(arr), return_counts=True)
    return np.stack([keys, freq]).astype(np.int64)


def _palette_from_counts(counts_per_mask: Sequence[np.ndarray], max_classes: int) -> Palette:
    counts: Counter[int] = Counter()
    for pair in counts_per_mask:
        counts.update(dict(zip(pair[0].tolist(), pair[1].tolist(), strict=True)))
    ranked = sorted(counts, key=lambda k: (-counts[k], k))
    if len(ranked) + 1 > max_classes:
        raise ValueError(f"too many mask colours: {len(ranked) + 1} classes found, at most {max_classes} allowed")
    return Palette(tuple((k >> 16 & 255, k >> 8 & 255, k & 255) for k in ranked))


def _mask_colour_counts(path: str | Path) -> np.ndarray:
    return _colour_counts(_load_small_rgb(path))


def build_palette(
    mask_paths: Sequence[str | Path],
    max_classes: int = 32,
    workers: int = 0,
    progress: Callable[[str], None] | None = None,
) -> Palette:
    """Derive the colour palette from masks: background is 0, other colours rank by pixel count.

    Masks are shrunk with nearest-neighbour sampling before counting, which keeps exact
    colours but makes counts approximate and can miss a colour covering only a few pixels
    of a very large mask; such pixels are later decoded as ``ignore_index``. With ``workers`` > 1
    the masks are read in that many processes; ``progress`` receives a line every
    `progress_every` masks.

    Raises
    ------
    ValueError
        If the masks contain more than ``max_classes`` classes (background included), which
        usually means instance-coloured rather than class-coloured masks.
    """
    total = len(mask_paths)
    every = progress_every(total)
    counts = []
    if workers > 1 and total > 1:
        # spawn, not fork: the caller may already run MLflow client threads
        with ProcessPoolExecutor(
            max_workers=min(workers, total), mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for done, pair in enumerate(pool.map(_mask_colour_counts, mask_paths, chunksize=16), start=1):
                counts.append(pair)
                if progress and (done % every == 0 or done == total):
                    progress(f"scanning mask colours {done}/{total}")
    else:
        for done, path in enumerate(mask_paths, start=1):
            counts.append(_mask_colour_counts(path))
            if progress and (done % every == 0 or done == total):
                progress(f"scanning mask colours {done}/{total}")
    return _palette_from_counts(counts, max_classes)


def progress_every(total: int) -> int:
    """Return how many items to process between progress lines: every `WARM_PROGRESS_EVERY`, at most ~100 lines."""
    return max(WARM_PROGRESS_EVERY, total // 100)


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


def _file_signature(path: Path) -> str:
    stat = path.stat()
    return f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"


def cache_max_bytes() -> int:
    """Return the cache size bound in bytes (``TRAIN_CACHE_MAX_MB``, default 4096)."""
    try:
        megabytes = int(os.environ.get(CACHE_MAX_MB_ENV, DEFAULT_CACHE_MAX_MB))
    except ValueError:
        megabytes = DEFAULT_CACHE_MAX_MB
    return max(megabytes, 0) * 1024 * 1024


def _decode_pair(sample: Sample, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode an image and colour mask once; return both resized to ``size`` and the mask's colour counts.

    The image is resized bilinearly, the mask with nearest-neighbour so it keeps only exact
    palette colours. JPEG images are decoded at reduced scale where that is enough.
    """
    height, width = size
    with Image.open(sample.image_path) as im:
        if im.format == "JPEG":
            im.draft("RGB", (width, height))
        image = np.asarray(im.convert("RGB").resize((width, height), Image.Resampling.BILINEAR))
    with Image.open(sample.mask_path) as im:
        full = im.convert("RGB")
        counts = _colour_counts(_small_rgb(full))
        mask = np.asarray(full.resize((width, height), Image.Resampling.NEAREST))
    return image, mask, counts


class ResizedCache:
    """On-disk cache of images and masks resized to the training size.

    Each sample is decoded from its full-resolution files once and stored as one uint8
    ``[H, W, 6]`` array (image channels, then colour mask) next to a small file of the mask's
    colour counts, so later passes and the palette never touch the originals. Entries are keyed
    by path, mtime and size, so an edited file is decoded afresh. Total size is bounded:
    `evict` removes the least recently used entries first (by mtime, refreshed on every hit).
    """

    def __init__(self, root: str | Path, size: tuple[int, int], max_bytes: int | None = None) -> None:
        self.size = size
        self.directory = Path(root) / f"{size[0]}x{size[1]}"
        self.max_bytes = cache_max_bytes() if max_bytes is None else max_bytes

    def key(self, sample: Sample) -> str:
        """Return the entry name for a sample's current image and mask files."""
        text = f"{_file_signature(sample.image_path)}|{_file_signature(sample.mask_path)}"
        return hashlib.sha1(text.encode(), usedforsecurity=False).hexdigest()

    def _paths(self, sample: Sample) -> tuple[Path, Path]:
        key = self.key(sample)
        return self.directory / f"{key}.npy", self.directory / f"{key}.counts.npy"

    def _store(self, path: Path, array: np.ndarray) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        with os.fdopen(fd, "wb") as fh:
            np.save(fh, array)
        os.replace(tmp, path)

    def ensure(self, sample: Sample) -> None:
        """Decode and store a sample unless it is cached already."""
        entry, counts_path = self._paths(sample)
        if entry.is_file() and counts_path.is_file():
            return
        image, mask, counts = _decode_pair(sample, self.size)
        self._store(counts_path, counts)
        self._store(entry, np.concatenate([image, mask], axis=-1))

    def get(self, sample: Sample) -> tuple[np.ndarray, np.ndarray]:
        """Return the resized image and colour mask, decoding and caching them on a miss."""
        entry, _ = self._paths(sample)
        try:
            array = np.load(entry)
        except (OSError, ValueError):
            self.ensure(sample)
            array = np.load(entry)
        else:
            os.utime(entry)
        return array[..., :3], array[..., 3:]

    def counts(self, sample: Sample) -> np.ndarray:
        """Return the sample's mask colour counts (``[2, N]``), decoding it on a miss."""
        self.ensure(sample)
        return np.load(self._paths(sample)[1])

    def size_bytes(self) -> int:
        """Return the total size of the cached files."""
        if not self.directory.is_dir():
            return 0
        return sum(f.stat().st_size for f in self.directory.iterdir() if f.is_file())

    def evict(self, keep: set[str] | None = None) -> int:
        """Delete least recently used entries until the cache fits ``max_bytes``; return how many went.

        Entries named in ``keep`` (keys from `key`) are removed last.
        """
        if not self.directory.is_dir():
            return 0
        keep = keep or set()
        entries: dict[str, list[Path]] = {}
        for file in self.directory.iterdir():
            if file.suffix == ".npy":
                entries.setdefault(file.name.split(".")[0], []).append(file)
        total = self.size_bytes()

        def last_used(key: str) -> float:
            # The main array is the entry's access marker: ``get`` refreshes it, while
            # the companion colour-count file is immutable after creation. Including
            # the companion mtime makes creation timing override the actual LRU order.
            entry = self.directory / f"{key}.npy"
            return entry.stat().st_mtime if entry.is_file() else max(f.stat().st_mtime for f in entries[key])

        order = sorted(
            entries,
            key=lambda k: (k in keep, last_used(k)),
        )
        removed = 0
        for key in order:
            if total <= self.max_bytes:
                break
            for file in entries[key]:
                total -= file.stat().st_size
                file.unlink(missing_ok=True)
            removed += 1
        return removed


def _warm_one(args: tuple[str, Sample, tuple[int, int], int]) -> None:
    root, sample, size, max_bytes = args
    ResizedCache(root, size, max_bytes).ensure(sample)


def warm_cache(
    cache: ResizedCache,
    samples: Sequence[Sample],
    workers: int = 0,
    progress: Callable[[str], None] | None = None,
) -> None:
    """Decode every sample into the cache, in parallel when ``workers`` > 1, then trim it to its bound.

    ``progress`` receives a line every `WARM_PROGRESS_EVERY` images.
    """
    total = len(samples)
    jobs = [(str(cache.directory.parent), s, cache.size, cache.max_bytes) for s in samples]
    if workers > 1 and total > 1:
        # spawn, not fork: the trainer process already runs MLflow client threads
        with ProcessPoolExecutor(
            max_workers=min(workers, total), mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for done, _ in enumerate(pool.map(_warm_one, jobs, chunksize=4), start=1):
                if progress and (done % WARM_PROGRESS_EVERY == 0 or done == total):
                    progress(f"caching resized images {done}/{total}")
    else:
        for done, job in enumerate(jobs, start=1):
            _warm_one(job)
            if progress and (done % WARM_PROGRESS_EVERY == 0 or done == total):
                progress(f"caching resized images {done}/{total}")
    cache.evict({cache.key(s) for s in samples})


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def common_image_size(samples: Sequence[Sample], workers: int = 0) -> tuple[int, int]:
    """Return the ``(width, height)`` every sample's image shares, reading only image headers.

    Raises
    ------
    ValueError
        If the images do not all have the same size.
    """
    paths = [sample.image_path for sample in samples]
    if workers > 1 and len(paths) > 1:
        # spawn, not fork: the trainer process already runs MLflow client threads
        with ProcessPoolExecutor(
            max_workers=min(workers, len(paths)), mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            sizes = set(pool.map(_image_size, paths, chunksize=256))
    else:
        sizes = {_image_size(path) for path in paths}
    if len(sizes) != 1:
        examples = ", ".join(f"{w}x{h}" for w, h in sorted(sizes)[:3])
        raise ValueError(
            f"input_resolution native needs every image the same size; found {len(sizes)} sizes (e.g. {examples})"
        )
    return sizes.pop()


def build_palette_cached(
    cache: ResizedCache,
    samples: Sequence[Sample],
    max_classes: int = 32,
    progress: Callable[[int, int], None] | None = None,
) -> Palette:
    """Derive the palette like `build_palette`, from the cache's per-mask colour counts."""
    counts = []
    every = max(len(samples) // 20, 1)
    for done, sample in enumerate(samples, start=1):
        counts.append(cache.counts(sample))
        if progress and (done % every == 0 or done == len(samples)):
            progress(done, len(samples))
    return _palette_from_counts(counts, max_classes)


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
    cache : ResizedCache | None
        Read resized pixels from this cache instead of decoding the full-resolution files.
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
        cache: ResizedCache | None = None,
    ) -> None:
        self.cache = cache
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
        if self.cache is not None:
            image, mask = self.cache.get(sample)
        else:
            image, mask, _ = _decode_pair(sample, self.size)

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
