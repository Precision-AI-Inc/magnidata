# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Torch-free helpers for locating a dataset's segmentation masks.

Shared by the trainer's data loading and by the API, which rejects datasets without masks
before scheduling a run.
"""

import csv
import os
from collections.abc import Mapping
from pathlib import Path

from precisionai.magnidata.training import contract

MASK_COLUMN = "mask_path"
CHECKED_ROWS = 20


def resolve_under(root: str | Path, relative: str) -> Path | None:
    """Join ``relative`` to ``root``; None if it is absolute, remote or escapes ``root``."""
    if not relative or os.path.isabs(relative) or "://" in relative:
        return None
    base = os.path.realpath(root)
    path = os.path.realpath(os.path.join(base, relative))
    if os.path.commonpath((base, path)) != base:
        return None
    return Path(path)


def sibling_mask_path(image_path: str) -> str:
    """Return ``<dataset>/masks/<stem>.png`` for ``<dataset>/images/<stem>.<ext>``."""
    dataset_dir = os.path.dirname(os.path.dirname(image_path))
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(dataset_dir, "masks", stem + ".png")


def find_mask(image_path: str, data_root: str | Path, row: Mapping[str, str] | None = None) -> Path | None:
    """Return the existing mask file for an image, or None.

    A ``mask_path`` column in the row wins; otherwise the sibling ``masks/<stem>.png``.
    """
    candidates = []
    if row and row.get(MASK_COLUMN):
        candidates.append(row[MASK_COLUMN])
    candidates.append(sibling_mask_path(image_path))
    for candidate in candidates:
        resolved = resolve_under(data_root, candidate)
        if resolved is not None and resolved.is_file():
            return resolved
    return None


def has_masks(dataset_csv: str | Path, data_root: str | Path, rows_to_check: int = CHECKED_ROWS) -> bool:
    """Return True if any of the first rows of a dataset CSV has a mask file."""
    with open(dataset_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        column = contract.image_path_column(reader.fieldnames or [])
        if column is None:
            return False
        for i, row in enumerate(reader):
            if i >= rows_to_check:
                break
            if find_mask(row.get(column) or "", data_root, row) is not None:
                return True
    return False
