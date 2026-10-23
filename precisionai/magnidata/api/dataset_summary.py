# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Summarize a dataset CSV for the workspace detail panel."""

import csv
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from precisionai.magnidata.training import contract
from precisionai.magnidata.training.dataset_check import find_mask

MAX_SCANNED_ROWS = 20_000
HISTOGRAM_BINS = 12
MAX_NUMERIC_COLUMNS = 6
SAMPLE_IMAGES = 8
MASK_SAMPLE = 60


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


@lru_cache(maxsize=256)
def _count_rows_cached(path: str, mtime_ns: int, size: int) -> int:
    # mtime_ns and size are unused in the body; they are part of the cache key so an edited file is recounted.
    with open(path, newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def count_rows(path: str | Path) -> int | None:
    """Return the data-row count of a CSV, or None if it is unreadable."""
    try:
        stat = os.stat(path)
        return _count_rows_cached(str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None


def _evenly_spaced(items: list[Any], count: int) -> list[Any]:
    if len(items) <= count:
        return list(items)
    step = len(items) / count
    return [items[int(i * step)] for i in range(count)]


def _numeric_column(name: str, values: list[float]) -> dict[str, Any]:
    low, high = min(values), max(values)
    bins = [0] * HISTOGRAM_BINS
    span = high - low
    for value in values:
        index = HISTOGRAM_BINS - 1 if span == 0 else min(int((value - low) / span * HISTOGRAM_BINS), HISTOGRAM_BINS - 1)
        bins[index] += 1
    return {"name": name, "min": low, "max": high, "mean": sum(values) / len(values), "histogram": bins}


def summarize_dataset(csv_path: str | Path, data_root: str | Path) -> dict[str, Any]:
    """Return row/column counts, numeric distributions, sample images and mask coverage."""
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        image_column = contract.image_path_column(columns)
        rows: list[dict[str, str]] = []
        total = 0
        for row in reader:
            total += 1
            if len(rows) < MAX_SCANNED_ROWS:
                rows.append(row)

    numeric: list[dict[str, Any]] = []
    for name in columns:
        if name == image_column:
            continue
        values = [v for v in (_to_float(row.get(name)) for row in rows) if v is not None]
        if len(values) >= max(1, len(rows) // 2):
            numeric.append(_numeric_column(name, values))

    images = [row[image_column] for row in rows if image_column and row.get(image_column)]
    probe = _evenly_spaced(rows, MASK_SAMPLE) if image_column else []
    with_masks = sum(1 for row in probe if find_mask(row.get(image_column or "") or "", data_root, row) is not None)

    return {
        "row_count": total,
        "column_count": len(columns),
        "numeric_column_count": len(numeric),
        "numeric_columns": sorted(numeric, key=lambda c: c["max"] - c["min"] == 0)[:MAX_NUMERIC_COLUMNS],
        "image_column": image_column,
        "sample_images": _evenly_spaced(images, SAMPLE_IMAGES),
        "mask_coverage": {"checked": len(probe), "with_masks": with_masks},
        "truncated": total > len(rows),
    }
