# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the training contract."""

import subprocess
import sys

import pytest

from precisionai.magnidata.training import contract


def test_per_sample_path_round_trip():
    assert contract.per_sample_path(3) == "magnidata/per_sample/epoch_0003.csv"
    assert contract.parse_per_sample_path(contract.per_sample_path(3)) == 3
    assert contract.parse_per_sample_path(contract.per_sample_path(12345)) == 12345


@pytest.mark.parametrize(
    "path",
    ["magnidata/per_sample/epoch_3.csv", "magnidata/split.csv", "other/per_sample/epoch_0001.csv", ""],
)
def test_parse_per_sample_path_rejects_other_paths(path):
    assert contract.parse_per_sample_path(path) is None


def test_prediction_path_uses_stem():
    assert contract.prediction_path(2, "dir/a.b.jpg") == "magnidata/predictions/epoch_0002/a.b.png"


def test_parse_loss_columns_keeps_header_order():
    header = ["image_name", "split", "loss_total", "loss_dice", "confidence", "loss_cce", "iou"]
    assert contract.parse_loss_columns(header) == ["dice", "cce"]


def test_parse_loss_columns_rejects_invalid_term():
    with pytest.raises(ValueError, match="term"):
        contract.parse_loss_columns(["loss_Bad-Name"])


def test_loss_column_rejects_invalid_term():
    with pytest.raises(ValueError, match="term"):
        contract.loss_column("1abc")
    assert contract.loss_column("dice") == "loss_dice"


def test_validate_header_accepts_complete_header():
    contract.validate_header(["image_name", "split", "loss_total", "loss_dice"], ["dice"])


@pytest.mark.parametrize(
    ("header", "missing"),
    [
        (["split", "loss_total", "loss_dice"], "image_name"),
        (["image_name", "split", "loss_dice"], "loss_total"),
        (["image_name", "split", "loss_total"], "loss_dice"),
    ],
)
def test_validate_header_rejects_missing_columns(header, missing):
    with pytest.raises(ValueError, match=missing):
        contract.validate_header(header, ["dice"])


def test_metric_keys_match_pai_seed():
    assert contract.train_loss_key("dice") == "train_loss/dice"
    assert contract.train_loss_key() == "train_loss/loss"
    assert contract.val_loss_key() == "val_loss/loss"
    assert contract.val_iou_key("weed") == "val_metric/iou_weed"
    assert contract.LEARNING_RATE_METRIC == "Learning_rate/lr"
    assert contract.MIOU_METRIC == "val_metric/miou"


def test_run_status_values():
    assert [s.value for s in contract.RunStatus] == ["SCHEDULED", "RUNNING", "FINISHED", "FAILED", "KILLED"]


def test_image_key_is_basename():
    assert contract.image_key("a/b/c.png") == "c.png"


@pytest.mark.parametrize(
    ("header", "expected"),
    [(["a", "image_path"], "image_path"), (["a", "file_path"], "file_path"), (["x", "y"], "x"), ([], None)],
)
def test_image_path_column(header, expected):
    assert contract.image_path_column(header) == expected


def test_import_does_not_pull_in_torch_or_mlflow():
    code = (
        "import sys; import precisionai.magnidata.training.contract; "
        "assert 'torch' not in sys.modules and 'mlflow' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_parse_scored_epochs():
    assert contract.parse_scored_epochs({contract.TAG_SCORED_EPOCHS: "3,1,2,2"}) == [1, 2, 3]
    assert contract.parse_scored_epochs({contract.TAG_SCORED_EPOCHS: ""}) == []
    assert contract.parse_scored_epochs({}) == []
    assert contract.parse_scored_epochs({contract.TAG_SCORED_EPOCHS: "1,x, 4"}) == [1, 4]
