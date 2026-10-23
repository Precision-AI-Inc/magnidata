# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Per-sample loss recorder shared by the reference trainer and external trainers.

A `LossRecorder` accumulates per-sample loss values during a scoring pass, validates them
against the training contract and hands one CSV per epoch to a sink. It needs only numpy:
values may be numpy arrays, lists, or anything with ``.tolist()`` (a detached torch tensor),
and torch is never imported.
"""

import csv
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from precisionai.magnidata.training import contract

EXTRA_COLUMNS = ("confidence", "iou")


class RecorderSink(Protocol):
    """Destination for recorder output: files, tags and a cancel flag."""

    def log_file(self, local_path: Path, artifact_dir: str) -> None:
        """Store ``local_path`` under ``artifact_dir`` keeping its file name."""
        ...

    def set_tag(self, key: str, value: str) -> None:
        """Set a run tag."""
        ...

    def cancel_requested(self) -> bool:
        """Return True once the UI asked for the run to stop."""
        ...


class LocalDirSink:
    """Sink that writes into a folder using the artifact layout; for tests and offline runs.

    Parameters
    ----------
    root : Path
        Folder that plays the role of the run's artifact root.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.tags: dict[str, str] = {}

    def log_file(self, local_path: Path, artifact_dir: str) -> None:
        """Copy ``local_path`` into ``root/artifact_dir``."""
        dst = self.root / artifact_dir
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dst / Path(local_path).name)

    def set_tag(self, key: str, value: str) -> None:
        """Remember a tag in ``self.tags``."""
        self.tags[key] = value

    def cancel_requested(self) -> bool:
        """Return True once the cancel tag has been set to ``true``."""
        return self.tags.get(contract.TAG_CANCEL_REQUESTED) == "true"


def _as_array(values: Any, what: str, expected: int) -> np.ndarray:
    """Convert an array-like to a 1-D float array of ``expected`` entries."""
    raw = values.tolist() if hasattr(values, "tolist") else list(values)
    try:
        arr = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as err:
        raise ValueError(f"{what} must be numeric") from err
    if arr.ndim != 1 or len(arr) != expected:
        raise ValueError(f"{what} must be 1-D with {expected} values, got shape {arr.shape}")
    return arr


class LossRecorder:
    """Accumulate per-sample losses for one epoch at a time and log them as contract CSVs.

    Parameters
    ----------
    loss_terms : Sequence[str]
        Names of the loss terms, in column order (``total`` is implicit).
    sink : RecorderSink
        Where files and tags go.
    scratch_dir : Path
        Local folder for files before they are handed to the sink.
    scored_epochs : Sequence[int]
        Epochs the run already has per-sample files for (when attaching to an existing run).
    """

    def __init__(
        self,
        loss_terms: Sequence[str],
        sink: RecorderSink,
        scratch_dir: Path,
        scored_epochs: Sequence[int] = (),
    ) -> None:
        terms = [contract.check_term(t) for t in loss_terms]
        if "total" in terms:
            raise ValueError("loss term 'total' is reserved for the weighted sum")
        if len(set(terms)) != len(terms):
            raise ValueError("duplicate loss term")
        self.loss_terms = terms
        self.sink = sink
        self.scratch_dir = Path(scratch_dir)
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self._scored: list[int] = sorted(set(scored_epochs))
        self._tagged = False
        self._epoch: int | None = None
        self._rows: list[list[Any]] = []
        self._names: set[str] = set()
        self._extras: tuple[str, ...] | None = None

    def write_split(self, assignments: Mapping[str, str]) -> None:
        """Log ``magnidata/split.csv`` (image name to train, val or test) and tag the run ``magnidata.has_split``."""
        bad = sorted({s for s in assignments.values() if s not in contract.SPLITS})
        if bad:
            raise ValueError(f"invalid split label(s): {', '.join(map(str, bad))}")
        path = self.scratch_dir / Path(contract.SPLIT_ARTIFACT).name
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["image_name", "split"])
            writer.writerows(assignments.items())
        self.sink.log_file(path, contract.ARTIFACT_ROOT)
        self.sink.set_tag(contract.TAG_HAS_SPLIT, "true")

    def write_palette(self, palette: Mapping[int, dict]) -> None:
        """Log ``magnidata/palette.json`` (class index → RGB and name)."""
        path = self.scratch_dir / Path(contract.PALETTE_ARTIFACT).name
        path.write_text(json.dumps({str(k): v for k, v in palette.items()}, indent=2))
        self.sink.log_file(path, contract.ARTIFACT_ROOT)

    def begin_epoch(self, epoch: int) -> None:
        """Start collecting rows for ``epoch``, discarding anything left from an unfinished one."""
        self._epoch = epoch
        self._rows = []
        self._names = set()
        self._extras = None

    def add_batch(
        self,
        image_names: Sequence[str],
        split: str | Sequence[str],
        losses: Mapping[str, Any],
        extras: Mapping[str, Any] | None = None,
    ) -> None:
        """Add one batch of per-sample values to the current epoch.

        Parameters
        ----------
        image_names : Sequence[str]
            Image basenames (the join key to the dataset).
        split : str | Sequence[str]
            One split label for the whole batch, or one per sample.
        losses : Mapping[str, Any]
            1-D array-likes keyed by ``total`` and every declared term.
        extras : Mapping[str, Any] | None
            Optional ``confidence`` / ``iou`` arrays; must be the same set in every batch.

        Raises
        ------
        ValueError
            On any contract violation. The batch is rejected as a whole.
        """
        if self._epoch is None:
            raise ValueError("add_batch called before begin_epoch")
        n = len(image_names)
        splits = self._check_splits(split, n)
        columns = self._check_losses(losses, n)
        extra_names = self._check_extras(extras or {}, columns, n)

        finite = np.all([np.isfinite(a) for a in columns.values()], axis=0) if n else np.array([], dtype=bool)
        if not finite.all():
            names = [str(image_names[int(i)]) for i in np.flatnonzero(~finite)]
            raise ValueError(f"non-finite loss values for: {', '.join(names)}")

        keys = [str(name) for name in image_names]
        seen = set(self._names)
        for key in keys:
            if key in seen:
                raise ValueError(f"duplicate image_name in epoch: {key}")
            seen.add(key)

        self._extras = extra_names
        self._names = seen
        ordered = [columns[c] for c in ["total", *self.loss_terms, *extra_names]]
        self._rows.extend([keys[i], splits[i], *(f"{a[i]:.6g}" for a in ordered)] for i in range(n))

    @staticmethod
    def _check_splits(split: str | Sequence[str], n: int) -> list[str]:
        splits = [split] * n if isinstance(split, str) else list(split)
        if len(splits) != n:
            raise ValueError(f"split has {len(splits)} entries for {n} images")
        bad = sorted({s for s in splits if s not in contract.SPLITS})
        if bad:
            raise ValueError(f"invalid split label(s): {', '.join(map(str, bad))}")
        return splits

    def _check_losses(self, losses: Mapping[str, Any], n: int) -> dict[str, np.ndarray]:
        expected = {"total", *self.loss_terms}
        missing = sorted(expected - set(losses))
        unknown = sorted(set(losses) - expected)
        if missing:
            raise ValueError(f"missing loss term(s): {', '.join(missing)}")
        if unknown:
            raise ValueError(f"unknown loss term(s): {', '.join(unknown)}")
        return {name: _as_array(losses[name], f"loss {name}", n) for name in ["total", *self.loss_terms]}

    def _check_extras(self, extras: Mapping[str, Any], columns: dict[str, np.ndarray], n: int) -> tuple[str, ...]:
        unknown = sorted(set(extras) - set(EXTRA_COLUMNS))
        if unknown:
            raise ValueError(f"unknown extra column(s): {', '.join(unknown)}")
        names = tuple(c for c in EXTRA_COLUMNS if c in extras)
        if self._extras is not None and names != self._extras:
            raise ValueError(f"extra columns {list(names)} differ from earlier batches {list(self._extras)}")
        for name in names:
            columns[name] = _as_array(extras[name], f"extra {name}", n)
        return names

    def end_epoch(self) -> str:
        """Write the epoch's CSV, log it, update the run tags and return the artifact path."""
        if self._epoch is None:
            raise ValueError("end_epoch called before begin_epoch")
        if not self._rows:
            raise ValueError(f"no samples recorded for epoch {self._epoch}")
        epoch = self._epoch
        artifact = contract.per_sample_path(epoch)
        path = self.scratch_dir / Path(artifact).name
        header = [
            "image_name",
            "split",
            "loss_total",
            *(contract.loss_column(t) for t in self.loss_terms),
            *(self._extras or ()),
        ]
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(self._rows)
        self.sink.log_file(path, str(Path(artifact).parent))

        if not self._tagged:
            self.sink.set_tag(contract.TAG_PER_SAMPLE_SCHEMA, contract.SCHEMA_VERSION)
            self.sink.set_tag(contract.TAG_LOSS_TERMS, ",".join(self.loss_terms))
            self._tagged = True
        if epoch not in self._scored:
            self._scored = sorted([*self._scored, epoch])
        self.sink.set_tag(contract.TAG_SCORED_EPOCHS, ",".join(str(e) for e in self._scored))

        self._epoch = None
        self._rows = []
        self._names = set()
        return artifact

    def cancel_requested(self) -> bool:
        """Return whether the sink reports a cancel request."""
        return self.sink.cancel_requested()
