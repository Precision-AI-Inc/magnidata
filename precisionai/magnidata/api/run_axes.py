# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Per-sample statistics across training epochs, derived from the per-sample loss files.

Raw per-epoch losses say how hard a sample is *now*. These axes say which samples *stay*
hard: mean, improvement, trend and rank-normalised versions (terms live on different scales),
plus a hardness score for how often a sample sits in the worst fraction of its split.
"""

import csv
import threading
import warnings
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from precisionai.magnidata.training import contract

SLOPE_WINDOW = 5
HARD_QUANTILE = 0.2
CACHE_ENTRIES = 8
OPTIONAL_COLUMNS = ("iou", "confidence")


@dataclass
class _Loaded:
    """Per-sample values of every scored epoch up to the selected one, aligned on image name."""

    names: list[str]
    epochs: list[int]
    splits: list[str]
    matrices: dict[str, np.ndarray] = field(default_factory=dict)


def _to_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _read_epoch(path: Path) -> tuple[list[str], list[str], dict[str, list[float]]]:
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        contract.validate_header(header, contract.parse_loss_columns(header))
        numeric = [c for c in header if c.startswith("loss_") or c in OPTIONAL_COLUMNS]
        names, splits = [], []
        values: dict[str, list[float]] = {c: [] for c in numeric}
        for row in reader:
            names.append(row["image_name"])
            splits.append(row["split"])
            for c in numeric:
                values[c].append(_to_float(row[c]))
    return names, splits, values


def _load(files: Sequence[tuple[int, Path]], up_to_epoch: int | None) -> _Loaded:
    chosen = sorted((e, p) for e, p in files if up_to_epoch is None or e <= up_to_epoch)
    if not chosen:
        raise ValueError("no per-sample files to load")
    index: dict[str, int] = {}
    split_of: dict[str, str] = {}
    reads = []
    for _, path in chosen:
        names, splits, values = _read_epoch(path)
        for name, split in zip(names, splits, strict=True):
            index.setdefault(name, len(index))
            split_of[name] = split
        reads.append((names, values))
    columns = sorted({c for _, values in reads for c in values})
    matrices = {c: np.full((len(index), len(chosen)), np.nan) for c in columns}
    for j, (names, values) in enumerate(reads):
        rows = np.fromiter((index[n] for n in names), dtype=np.int64, count=len(names))
        for c, series in values.items():
            matrices[c][rows, j] = series
    ordered = list(index)
    return _Loaded(ordered, [e for e, _ in chosen], [split_of[n] for n in ordered], matrices)


def load_epoch_matrix(
    files: Sequence[tuple[int, Path]], term: str, up_to_epoch: int | None = None
) -> tuple[list[str], list[int], np.ndarray]:
    """Return ``(image names, epochs, matrix[n_samples, n_epochs])`` of one loss term.

    ``term`` is a loss term name (``total`` for the weighted sum). A sample missing from an
    epoch has NaN there.
    """
    loaded = _load(files, up_to_epoch)
    column = contract.loss_column(term) if term != "total" else "loss_total"
    if column not in loaded.matrices:
        raise ValueError(f"unknown loss term: {term}")
    return loaded.names, loaded.epochs, loaded.matrices[column]


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Return ``(mid-rank - 1) / (n - 1)`` of each value among the non-NaN ones.

    The smallest value gets 0, the largest 1 and ties share their average rank; a lone
    value gets 0.5. NaN stays NaN.
    """
    out = np.full(values.shape, np.nan)
    valid = ~np.isnan(values)
    n = int(valid.sum())
    if n == 0:
        return out
    if n == 1:
        out[valid] = 0.5
        return out
    _, inverse, counts = np.unique(values[valid], return_inverse=True, return_counts=True)
    mid_rank = np.cumsum(counts) - (counts - 1) / 2  # 1-based average rank of each distinct value
    out[valid] = (mid_rank[inverse] - 1) / (n - 1)
    return out


def _rank_within_splits(values: np.ndarray, splits: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan)
    for split in np.unique(splits):
        mask = splits == split
        out[mask] = percentile_rank(values[mask])
    return out


def _slope(matrix: np.ndarray, epochs: np.ndarray, end: int) -> np.ndarray:
    """Least-squares slope over the last `SLOPE_WINDOW` scored epochs up to column ``end``."""
    window = matrix[:, max(0, end - SLOPE_WINDOW + 1) : end + 1]
    x = np.broadcast_to(epochs[max(0, end - SLOPE_WINDOW + 1) : end + 1].astype(float), window.shape)
    valid = ~np.isnan(window)
    n = valid.sum(1)
    xv, yv = np.where(valid, x, 0.0), np.where(valid, window, 0.0)
    sx, sy = xv.sum(1), yv.sum(1)
    sxx, sxy = (xv * xv).sum(1), (xv * yv).sum(1)
    denom = n * sxx - sx * sx
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (n * sxy - sx * sy) / denom
    return np.where((n >= 2) & (denom != 0), slope, np.nan)


def _first_valid(matrix: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(matrix)
    first = valid.argmax(1)
    return np.where(valid.any(1), matrix[np.arange(len(matrix)), first], np.nan)


def _hardness(total: np.ndarray, splits: np.ndarray) -> np.ndarray:
    """Fraction of the epochs a sample appears in where it ranks in the worst `HARD_QUANTILE` of its split."""
    ranks = np.column_stack([_rank_within_splits(total[:, j], splits) for j in range(total.shape[1])])
    present = (~np.isnan(ranks)).sum(1)
    hard = (ranks >= 1 - HARD_QUANTILE).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(present > 0, hard / present, np.nan)


@dataclass
class DerivedAxes:
    """Derived columns for one run at one selected epoch."""

    names: list[str]
    splits: list[str]
    columns: dict[str, np.ndarray]
    epoch: int
    scored_epochs: list[int]

    def to_csv(self, include_split: bool = True) -> str:
        """Render ``image_name``, ``split`` (optional) and every derived column as CSV (NaN → empty)."""
        header = ["image_name", *(["split"] if include_split else []), *self.columns]
        lines = [",".join(header)]
        formatted = {c: [("" if np.isnan(v) else f"{v:.6g}") for v in col] for c, col in self.columns.items()}
        for i, (name, split) in enumerate(zip(self.names, self.splits, strict=True)):
            cells = [
                _csv_cell(name),
                *([_csv_cell(split)] if include_split else []),
                *(formatted[c][i] for c in self.columns),
            ]
            lines.append(",".join(cells))
        return "\n".join(lines) + "\n"


def _csv_cell(text: str) -> str:
    return '"' + text.replace('"', '""') + '"' if any(ch in text for ch in ',"\n') else text


def derive_axes(
    files: Sequence[tuple[int, Path]], epoch: int | str = "latest", terms: Sequence[str] | None = None
) -> DerivedAxes:
    """Derive the per-sample axes of a run at ``epoch`` (``"latest"`` or a scored epoch number).

    For every term ``t`` (``total`` included): ``loss_t`` at the epoch, ``loss_t_mean`` over the
    scored epochs up to it, ``loss_t_delta`` (first value minus current; positive means improved),
    ``loss_t_slope`` (least squares over the last few scored epochs; negative means getting
    easier) and ``loss_t_rank`` (percentile rank within the sample's split). Also ``hardness``,
    and ``iou`` / ``confidence`` when logged.

    Raises
    ------
    ValueError
        If there are no files, or ``epoch`` is not a scored epoch.
    """
    scored = sorted(e for e, _ in files)
    if not scored:
        raise ValueError("run has no per-sample files")
    selected = scored[-1] if epoch == "latest" else int(epoch)
    if selected not in scored:
        raise ValueError(f"epoch {selected} has no per-sample file (scored: {', '.join(map(str, scored))})")

    loaded = _load(files, selected)
    end = loaded.epochs.index(selected)
    epochs = np.array(loaded.epochs)
    splits = np.array(loaded.splits)
    found = ["total", *contract.parse_loss_columns(loaded.matrices)]
    use = [t for t in found if terms is None or t in terms]

    columns: dict[str, np.ndarray] = {}
    for term in use:
        matrix = loaded.matrices[f"loss_{term}"]
        current = matrix[:, end]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows are expected and stay NaN
            mean = np.nanmean(matrix[:, : end + 1], axis=1)
        columns[f"loss_{term}"] = current
        columns[f"loss_{term}_mean"] = mean
        columns[f"loss_{term}_delta"] = _first_valid(matrix[:, : end + 1]) - current
        columns[f"loss_{term}_slope"] = _slope(matrix, epochs, end)
        columns[f"loss_{term}_rank"] = _rank_within_splits(current, splits)
    columns["hardness"] = _hardness(loaded.matrices["loss_total"], splits)
    for extra in OPTIONAL_COLUMNS:
        if extra in loaded.matrices:
            columns[extra] = loaded.matrices[extra][:, end]
    return DerivedAxes(loaded.names, loaded.splits, columns, selected, scored)


def compare_samples(a: DerivedAxes, b: DerivedAxes) -> DerivedAxes:
    """Per-image comparison of two runs: how hardness and total loss changed from ``a`` to ``b``.

    An outer join on image name. ``*_change`` is ``b - a`` (positive means the image got harder)
    and is NaN unless the image is in both runs.

    Raises
    ------
    ValueError
        If the runs share no image.
    """
    names = list(dict.fromkeys([*a.names, *b.names]))
    index_a = {n: i for i, n in enumerate(a.names)}
    index_b = {n: i for i, n in enumerate(b.names)}
    shared = set(index_a) & set(index_b)
    if not shared:
        raise ValueError("the runs have no image in common")

    def pick(axes: DerivedAxes, index: dict[str, int], column: str) -> np.ndarray:
        source = axes.columns[column]
        return np.array([source[index[n]] if n in index else np.nan for n in names], dtype=float)

    hardness_a, hardness_b = pick(a, index_a, "hardness"), pick(b, index_b, "hardness")
    loss_a, loss_b = pick(a, index_a, "loss_total"), pick(b, index_b, "loss_total")
    columns = {
        "hardness_a": hardness_a,
        "hardness_b": hardness_b,
        "hardness_change": hardness_b - hardness_a,
        "loss_total_change": loss_b - loss_a,
        "in_a": np.array([n in index_a for n in names], dtype=float),
        "in_b": np.array([n in index_b for n in names], dtype=float),
    }
    return DerivedAxes(names, [""] * len(names), columns, b.epoch, b.scored_epochs)


def sample_history(files: Sequence[tuple[int, Path]], image_name: str) -> dict | None:
    """Return one image's loss trajectory across the scored epochs, or None if it never appears.

    Values are aligned with ``epochs``; an epoch the image is missing from has None.
    """
    epochs: list[int] = []
    rows: list[dict[str, str] | None] = []
    split = None
    for epoch, path in sorted(files):
        found = None
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                if row["image_name"] == image_name:
                    found = row
                    break
        epochs.append(epoch)
        rows.append(found)
        if found:
            split = found["split"]
    if split is None:
        return None

    def series(column: str) -> list[float | None]:
        out = []
        for row in rows:
            value = _to_float(row[column]) if row and column in row else float("nan")
            out.append(None if np.isnan(value) else value)
        return out

    columns = {c for row in rows if row for c in row}
    terms = {
        "total": series("loss_total"),
        **{t: series(contract.loss_column(t)) for t in contract.parse_loss_columns(sorted(columns))},
    }
    extras = {c: series(c) for c in OPTIONAL_COLUMNS if c in columns}
    return {"image_name": image_name, "split": split, "epochs": epochs, "terms": terms, **extras}


class RunAxesCache:
    """Small LRU of derived axes, keyed by ``(run_id, epoch, latest scored epoch)``.

    Including the latest scored epoch means live polling recomputes only when a new epoch lands.
    """

    def __init__(self, capacity: int = CACHE_ENTRIES) -> None:
        self.capacity = capacity
        self._items: OrderedDict[tuple, DerivedAxes] = OrderedDict()
        self._lock = threading.Lock()

    def get_or_compute(self, key: tuple, compute: Callable[[], DerivedAxes]) -> DerivedAxes:
        """Return the cached value for ``key`` or compute and store it."""
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        value = compute()
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
        return value

    def __contains__(self, key: tuple) -> bool:
        """Return whether ``key`` is cached."""
        with self._lock:
            return key in self._items

    def __len__(self) -> int:
        """Return the number of cached entries."""
        with self._lock:
            return len(self._items)

    def clear(self) -> None:
        """Drop everything."""
        with self._lock:
            self._items.clear()
