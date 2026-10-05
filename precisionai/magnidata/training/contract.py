# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Shared contract for MagniData training runs stored in MLflow.

Every participant in the training feedback loop (the reference trainer, the per-sample
recorder, the worker, the API and external trainers such as pai-seed) exchanges data
through MLflow. This module defines, in one place, which tags link a run to a dataset,
which metric keys it uses, where per-sample artifacts live and how their columns are
named. It deliberately imports neither torch nor mlflow.
"""

import os
import re
from collections.abc import Iterable, Sequence
from enum import Enum

SCHEMA_VERSION = "magnidata-training/1.0.0"

TAG_DATASET_SOURCE = "magnidata.dataset_source"
TAG_DATASET_NAME = "magnidata.dataset_name"
TAG_ORIGIN = "magnidata.origin"
TAG_PRESET = "magnidata.preset"
TAG_LOSS_TERMS = "magnidata.loss_terms"
TAG_PER_SAMPLE_SCHEMA = "magnidata.per_sample_schema"
TAG_SCORED_EPOCHS = "magnidata.scored_epochs"
TAG_CANCEL_REQUESTED = "magnidata.cancel_requested"
TAG_PARENT_RUN_ID = "magnidata.parent_run_id"
TAG_WORKER = "magnidata.worker"

ORIGIN_UI = "ui"
ORIGIN_EXTERNAL = "external"

ARTIFACT_ROOT = "magnidata"
SPLIT_ARTIFACT = f"{ARTIFACT_ROOT}/split.csv"
PALETTE_ARTIFACT = f"{ARTIFACT_ROOT}/palette.json"
CHECKPOINT_ARTIFACT = "checkpoints/last.pt"

LEARNING_RATE_METRIC = "Learning_rate/lr"
MIOU_METRIC = "val_metric/miou"

REQUIRED_COLUMNS = ("image_name", "split", "loss_total")
SPLITS = ("train", "val", "test")

_TERM_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_LOSS_PREFIX = "loss_"
_PER_SAMPLE_RE = re.compile(rf"^{ARTIFACT_ROOT}/per_sample/epoch_(\d{{4,}})\.csv$")


class RunStatus(str, Enum):
    """MLflow run statuses used by the training loop."""

    SCHEDULED = "SCHEDULED"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    KILLED = "KILLED"


def check_term(term: str) -> str:
    """Return ``term`` unchanged if it is a valid loss-term name, else raise ``ValueError``."""
    if not _TERM_RE.match(term):
        raise ValueError(f"invalid loss term name {term!r}: must match {_TERM_RE.pattern}")
    return term


def train_loss_key(term: str = "loss") -> str:
    """Return the metric key of a training loss term (``loss`` is the weighted total)."""
    return f"train_loss/{term}"


def val_loss_key(term: str = "loss") -> str:
    """Return the metric key of a validation loss term (``loss`` is the weighted total)."""
    return f"val_loss/{term}"


def val_iou_key(class_name: str) -> str:
    """Return the metric key of a per-class validation IoU."""
    return f"val_metric/iou_{class_name}"


def per_sample_path(epoch: int) -> str:
    """Return the artifact path of the per-sample file for ``epoch``."""
    return f"{ARTIFACT_ROOT}/per_sample/epoch_{epoch:04d}.csv"


def parse_per_sample_path(path: str) -> int | None:
    """Return the epoch encoded in a per-sample artifact path, or None if it is not one."""
    match = _PER_SAMPLE_RE.match(path)
    return int(match.group(1)) if match else None


def prediction_path(epoch: int, image_name: str) -> str:
    """Return the artifact path of the predicted mask PNG for an image at ``epoch``."""
    stem = os.path.splitext(image_key(image_name))[0]
    return f"{ARTIFACT_ROOT}/predictions/epoch_{epoch:04d}/{stem}.png"


def loss_column(term: str) -> str:
    """Return the per-sample CSV column holding ``term``."""
    return _LOSS_PREFIX + check_term(term)


def parse_loss_columns(header: Iterable[str]) -> list[str]:
    """Return the loss-term names found in a per-sample header, in header order.

    ``loss_total`` is the weighted sum and is not a term. Other columns are ignored.

    Raises
    ------
    ValueError
        If a ``loss_*`` column carries an invalid term name.
    """
    terms = []
    for col in header:
        if not col.startswith(_LOSS_PREFIX) or col == "loss_total":
            continue
        terms.append(check_term(col[len(_LOSS_PREFIX) :]))
    return terms


def validate_header(header: Sequence[str], terms: Iterable[str]) -> None:
    """Check that a per-sample header has the required columns and one column per term.

    Raises
    ------
    ValueError
        If a required column or a declared term's column is missing.
    """
    present = set(header)
    missing = [c for c in REQUIRED_COLUMNS if c not in present]
    missing += [loss_column(t) for t in terms if loss_column(t) not in present]
    if missing:
        raise ValueError(f"per-sample header is missing columns: {', '.join(missing)}")


def image_key(value: str) -> str:
    """Return the join key of an image path: its basename."""
    return os.path.basename(value)


def image_path_column(cols: Sequence[str]) -> str | None:
    """Return the column holding each row's image path, or None for an empty header."""
    for c in cols:
        lc = c.lower()
        if lc == "image_path" or ("image" in lc and "path" in lc):
            return c
    return next((c for c in cols if "path" in c.lower()), cols[0] if cols else None)
