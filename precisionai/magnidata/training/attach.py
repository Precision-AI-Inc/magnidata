# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

r"""Attach per-sample loss files to an existing MLflow run.

Usage::

    python -m precisionai.magnidata.training.attach --run-id <id> --epoch 12 --file losses.csv \
        [--split split.csv] [--tracking-uri http://localhost:5052]

The file is validated against the training contract before anything is logged. A
long-format file with an ``epoch`` column is split into one artifact per epoch.
"""

import argparse
import csv
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from precisionai.magnidata.training import contract
from precisionai.magnidata.training.mlflow_sink import MlflowSink
from precisionai.magnidata.training.recorder import EXTRA_COLUMNS, LossRecorder, RecorderSink

DEFAULT_TRACKING_URI = "http://localhost:5052"


class _BufferedSink:
    """Collects sink calls so nothing reaches MLflow unless the whole file validated."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def log_file(self, local_path: Path, artifact_dir: str) -> None:
        self.calls.append(("log_file", (local_path, artifact_dir)))

    def set_tag(self, key: str, value: str) -> None:
        self.calls.append(("set_tag", (key, value)))

    def cancel_requested(self) -> bool:
        return False

    def replay(self, target: RecorderSink) -> None:
        """Forward every buffered call to ``target``."""
        for name, args in self.calls:
            getattr(target, name)(*args)


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def _split_map(path: Path) -> dict[str, str]:
    header, rows = _read_rows(path)
    if "image_name" not in header or "split" not in header:
        raise ValueError(f"split file {path} needs image_name and split columns")
    return {contract.image_key(r["image_name"]): r["split"] for r in rows}


def _group_by_epoch(
    rows: list[dict[str, str]], header: list[str], epoch: int | None
) -> dict[int, list[dict[str, str]]]:
    if "epoch" in header:
        grouped: dict[int, list[dict[str, str]]] = {}
        for row in rows:
            grouped.setdefault(int(row["epoch"]), []).append(row)
        return grouped
    if epoch is None:
        raise ValueError("--epoch is required unless the file has an epoch column")
    return {epoch: rows}


def _stage(
    file: Path, epoch: int | None, split_file: Path | None, scratch: Path, existing_epochs: Sequence[int]
) -> _BufferedSink:
    """Validate ``file`` by running it through a recorder on a buffering sink."""
    header, rows = _read_rows(file)
    if "image_name" not in header or "loss_total" not in header:
        raise ValueError("per-sample file must have image_name and loss_total columns")
    terms = contract.parse_loss_columns(header)
    splits = _split_map(split_file) if split_file else {}
    if "split" not in header and not split_file:
        raise ValueError("per-sample file has no split column; pass --split")
    extras = [c for c in EXTRA_COLUMNS if c in header]

    sink = _BufferedSink()
    recorder = LossRecorder(terms, sink, scratch, scored_epochs=existing_epochs)
    if split_file:
        recorder.write_split(splits)
    for ep, ep_rows in sorted(_group_by_epoch(rows, header, epoch).items()):
        names = [contract.image_key(r["image_name"]) for r in ep_rows]
        try:
            row_splits = [r["split"] if "split" in header else splits[n] for r, n in zip(ep_rows, names, strict=True)]
        except KeyError as err:
            raise ValueError(f"image {err.args[0]} is missing from the split file") from None
        losses = {"total": [float(r["loss_total"]) for r in ep_rows]}
        losses.update({t: [float(r[contract.loss_column(t)]) for r in ep_rows] for t in terms})
        recorder.begin_epoch(ep)
        recorder.add_batch(
            names,
            row_splits,
            losses,
            {c: [float(r[c]) for r in ep_rows] for c in extras},
        )
        recorder.end_epoch()
    return sink


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--file", required=True, type=Path, help="per-sample CSV (wide or long format)")
    parser.add_argument("--epoch", type=int, help="epoch of a wide-format file")
    parser.add_argument(
        "--split", type=Path, help="split.csv (image_name,split) to log and to fill a missing split column"
    )
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    args = parser.parse_args(argv)

    try:
        sink = MlflowSink(args.run_id, args.tracking_uri)
        existing = [int(e) for e in (sink.get_tag(contract.TAG_SCORED_EPOCHS) or "").split(",") if e]
        with tempfile.TemporaryDirectory() as tmp:
            staged = _stage(args.file, args.epoch, args.split, Path(tmp), existing)
            staged.replay(sink)
    except (ValueError, OSError, ImportError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    print(f"attached {sum(1 for n, _ in staged.calls if n == 'log_file')} file(s) to run {args.run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
