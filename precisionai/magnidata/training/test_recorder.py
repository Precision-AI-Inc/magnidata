# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-sample loss recorder, its sinks and the attach CLI."""

import csv
import subprocess
import sys

import numpy as np
import pytest

from precisionai.magnidata.training import contract
from precisionai.magnidata.training.attach import main as attach_main
from precisionai.magnidata.training.mlflow_sink import MlflowSink
from precisionai.magnidata.training.recorder import LocalDirSink, LossRecorder


def _read(path):
    with open(path, newline="") as fh:
        return list(csv.reader(fh))


@pytest.fixture
def recorder(tmp_path):
    sink = LocalDirSink(tmp_path / "run")
    rec = LossRecorder(["dice", "cce"], sink, tmp_path / "scratch")
    rec.begin_epoch(1)
    return rec


def _batch(names, **overrides):
    n = len(names)
    losses = {"total": np.full(n, 0.5), "dice": np.full(n, 0.3), "cce": np.full(n, 0.2)}
    losses.update(overrides)
    return losses


def test_happy_path_writes_contract_csv(tmp_path):
    sink = LocalDirSink(tmp_path / "run")
    rec = LossRecorder(["dice", "cce"], sink, tmp_path / "scratch")
    rec.write_split({"a.png": "train", "b.png": "val", "c.png": "val"})
    rec.write_palette({0: {"name": "bg", "rgb": [0, 0, 0]}})
    rec.begin_epoch(1)
    rec.add_batch(
        ["a.png", "b.png"],
        ["train", "val"],
        {"total": [0.9, 0.8], "dice": [0.6, 0.5], "cce": [0.3, 0.3]},
        {"confidence": [0.9, 0.7]},
    )
    rec.add_batch(["c.png"], "val", {"total": [0.4], "dice": [0.3], "cce": [0.1]}, {"confidence": [0.5]})
    assert rec.end_epoch() == "magnidata/per_sample/epoch_0001.csv"

    rows = _read(tmp_path / "run" / contract.per_sample_path(1))
    assert rows[0] == ["image_name", "split", "loss_total", "loss_dice", "loss_cce", "confidence"]
    assert [r[0] for r in rows[1:]] == ["a.png", "b.png", "c.png"]
    assert [r[1] for r in rows[1:]] == ["train", "val", "val"]
    assert [float(r[2]) for r in rows[1:]] == pytest.approx([0.9, 0.8, 0.4])
    assert [float(r[5]) for r in rows[1:]] == pytest.approx([0.9, 0.7, 0.5])
    contract.validate_header(rows[0], ["dice", "cce"])

    assert _read(tmp_path / "run" / contract.SPLIT_ARTIFACT)[1] == ["a.png", "train"]
    assert (tmp_path / "run" / contract.PALETTE_ARTIFACT).exists()
    assert sink.tags[contract.TAG_PER_SAMPLE_SCHEMA] == contract.SCHEMA_VERSION
    assert sink.tags[contract.TAG_LOSS_TERMS] == "dice,cce"
    assert sink.tags[contract.TAG_SCORED_EPOCHS] == "1"


def test_scored_epochs_accumulate(tmp_path):
    sink = LocalDirSink(tmp_path / "run")
    rec = LossRecorder(["dice"], sink, tmp_path / "scratch", scored_epochs=[5])
    for epoch in (2, 3):
        rec.begin_epoch(epoch)
        rec.add_batch(["a.png"], "train", {"total": [1.0], "dice": [1.0]})
        rec.end_epoch()
    assert sink.tags[contract.TAG_SCORED_EPOCHS] == "2,3,5"


def test_accepts_numpy_and_lists_and_tolist_objects(recorder):
    class FakeTensor:
        def tolist(self):
            return [0.1, 0.2]

    recorder.add_batch(
        ["a.png", "b.png"], "train", {"total": FakeTensor(), "dice": [0.1, 0.2], "cce": np.array([0.0, 0.0])}
    )
    assert recorder.end_epoch()


def test_add_batch_before_begin_epoch(tmp_path):
    rec = LossRecorder(["dice"], LocalDirSink(tmp_path / "run"), tmp_path / "s")
    with pytest.raises(ValueError, match="before begin_epoch"):
        rec.add_batch(["a.png"], "train", {"total": [1.0], "dice": [1.0]})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"losses": {"total": [1.0], "dice": [1.0]}}, "missing loss term.*cce"),
        ({"losses": {"total": [1.0], "dice": [1.0], "cce": [1.0], "bce": [1.0]}}, "unknown loss term.*bce"),
        ({"losses": {"total": [1.0, 2.0], "dice": [1.0], "cce": [1.0]}}, "must be 1-D with 1 values"),
        ({"losses": {"total": [float("nan")], "dice": [1.0], "cce": [1.0]}}, "non-finite.*a.png"),
        ({"losses": {"total": [1.0], "dice": [float("inf")], "cce": [1.0]}}, "non-finite.*a.png"),
        ({"split": "training"}, "invalid split"),
        ({"split": ["train", "val"]}, "split has 2 entries"),
        ({"extras": {"nonsense": [1.0]}}, "unknown extra"),
    ],
)
def test_add_batch_rejects_bad_input(recorder, kwargs, match):
    args = {"split": "train", "losses": {"total": [1.0], "dice": [1.0], "cce": [1.0]}}
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        recorder.add_batch(["a.png"], **args)


def test_duplicate_image_name_across_batches(recorder):
    recorder.add_batch(["a.png"], "train", _batch(["a.png"]))
    with pytest.raises(ValueError, match=r"duplicate image_name.*a\.png"):
        recorder.add_batch(["b.png", "a.png"], "train", _batch(["b.png", "a.png"]))


def test_rejected_batch_leaves_state_untouched(recorder):
    with pytest.raises(ValueError, match="non-finite"):
        recorder.add_batch(["a.png", "b.png"], "train", _batch(["a", "b"], total=[1.0, float("nan")]))
    recorder.add_batch(["a.png", "b.png"], "train", _batch(["a", "b"]))
    assert recorder.end_epoch()


def test_extras_must_match_between_batches(recorder):
    recorder.add_batch(["a.png"], "train", _batch(["a"]), {"iou": [0.5]})
    with pytest.raises(ValueError, match="differ from earlier batches"):
        recorder.add_batch(["b.png"], "train", _batch(["b"]))


def test_end_epoch_without_samples(recorder):
    with pytest.raises(ValueError, match="no samples"):
        recorder.end_epoch()


def test_invalid_terms_rejected(tmp_path):
    sink = LocalDirSink(tmp_path)
    with pytest.raises(ValueError, match="term"):
        LossRecorder(["Bad-Name"], sink, tmp_path)
    with pytest.raises(ValueError, match="reserved"):
        LossRecorder(["total"], sink, tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        LossRecorder(["dice", "dice"], sink, tmp_path)


def test_write_split_rejects_bad_label(recorder):
    with pytest.raises(ValueError, match="invalid split"):
        recorder.write_split({"a.png": "holdout"})


def test_cancel_requested_delegates_to_sink(tmp_path):
    sink = LocalDirSink(tmp_path)
    rec = LossRecorder(["dice"], sink, tmp_path / "s")
    assert rec.cancel_requested() is False
    sink.set_tag(contract.TAG_CANCEL_REQUESTED, "true")
    assert rec.cancel_requested() is True


def test_mlflow_sink_logs_artifacts_tags_and_cancel(mlflow_tracking, tmp_path):
    uri, client, run_id = mlflow_tracking()
    sink = MlflowSink(run_id, uri, cancel_poll_s=0)
    rec = LossRecorder(["dice"], sink, tmp_path / "scratch")
    rec.write_split({"a.png": "train"})
    rec.begin_epoch(1)
    rec.add_batch(["a.png"], "train", {"total": [0.5], "dice": [0.5]})
    rec.end_epoch()

    paths = [f.path for f in client.list_artifacts(run_id, "magnidata/per_sample")]
    assert paths == [contract.per_sample_path(1)]
    assert [f.path for f in client.list_artifacts(run_id, "magnidata") if not f.is_dir] == [contract.SPLIT_ARTIFACT]
    tags = client.get_run(run_id).data.tags
    assert tags[contract.TAG_PER_SAMPLE_SCHEMA] == contract.SCHEMA_VERSION
    assert tags[contract.TAG_SCORED_EPOCHS] == "1"

    assert rec.cancel_requested() is False
    client.set_tag(run_id, contract.TAG_CANCEL_REQUESTED, "true")
    assert rec.cancel_requested() is True


def test_mlflow_sink_throttles_cancel_polling(mlflow_tracking):
    uri, client, run_id = mlflow_tracking()
    sink = MlflowSink(run_id, uri, cancel_poll_s=3600)
    assert sink.cancel_requested() is False
    client.set_tag(run_id, contract.TAG_CANCEL_REQUESTED, "true")
    assert sink.cancel_requested() is False


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def test_attach_long_format_logs_one_artifact_per_epoch(mlflow_tracking, tmp_path):
    uri, client, run_id = mlflow_tracking()
    losses = tmp_path / "losses.csv"
    _write_csv(
        losses,
        ["epoch", "image_name", "loss_total", "loss_dice"],
        [[1, "dir/a.png", 0.9, 0.9], [1, "b.png", 0.8, 0.8], [2, "dir/a.png", 0.5, 0.5], [2, "b.png", 0.4, 0.4]],
    )
    split = tmp_path / "split.csv"
    _write_csv(split, ["image_name", "split"], [["a.png", "train"], ["b.png", "val"]])

    code = attach_main(["--run-id", run_id, "--file", str(losses), "--split", str(split), "--tracking-uri", uri])
    assert code == 0
    logged = sorted(f.path for f in client.list_artifacts(run_id, "magnidata/per_sample"))
    assert logged == [contract.per_sample_path(1), contract.per_sample_path(2)]
    assert contract.SPLIT_ARTIFACT in [f.path for f in client.list_artifacts(run_id, "magnidata")]
    assert client.get_run(run_id).data.tags[contract.TAG_SCORED_EPOCHS] == "1,2"


def test_attach_wide_format_merges_with_existing_epochs(mlflow_tracking, tmp_path):
    uri, client, run_id = mlflow_tracking()
    client.set_tag(run_id, contract.TAG_SCORED_EPOCHS, "3")
    losses = tmp_path / "losses.csv"
    _write_csv(losses, ["image_name", "split", "loss_total", "loss_dice"], [["a.png", "train", 0.5, 0.5]])
    code = attach_main(["--run-id", run_id, "--epoch", "7", "--file", str(losses), "--tracking-uri", uri])
    assert code == 0
    assert client.get_run(run_id).data.tags[contract.TAG_SCORED_EPOCHS] == "3,7"


@pytest.mark.parametrize(
    ("header", "row", "args", "message"),
    [
        (
            ["image_name", "split", "loss_total", "loss_dice"],
            ["a.png", "train", "nan", "0.5"],
            ["--epoch", "1"],
            "non-finite",
        ),
        (["image_name", "split", "loss_dice"], ["a.png", "train", "0.5"], ["--epoch", "1"], "loss_total"),
        (["image_name", "split", "loss_total"], ["a.png", "train", "0.5"], [], "--epoch"),
        (["image_name", "loss_total"], ["a.png", "0.5"], ["--epoch", "1"], "split"),
    ],
)
def test_attach_invalid_file_logs_nothing(mlflow_tracking, tmp_path, capsys, header, row, args, message):
    uri, client, run_id = mlflow_tracking()
    losses = tmp_path / "losses.csv"
    _write_csv(losses, header, [row])
    code = attach_main(["--run-id", run_id, "--file", str(losses), "--tracking-uri", uri, *args])
    assert code != 0
    assert message in capsys.readouterr().err
    assert client.list_artifacts(run_id) == []
    assert contract.TAG_SCORED_EPOCHS not in client.get_run(run_id).data.tags


def test_attach_rejects_image_missing_from_split_file(mlflow_tracking, tmp_path, capsys):
    uri, client, run_id = mlflow_tracking()
    losses = tmp_path / "losses.csv"
    _write_csv(losses, ["image_name", "loss_total"], [["a.png", "0.5"]])
    split = tmp_path / "split.csv"
    _write_csv(split, ["image_name", "split"], [["other.png", "train"]])
    code = attach_main(
        ["--run-id", run_id, "--epoch", "1", "--file", str(losses), "--split", str(split), "--tracking-uri", uri]
    )
    assert code != 0
    assert "a.png" in capsys.readouterr().err
    assert client.list_artifacts(run_id) == []


def test_recorder_import_does_not_pull_in_torch_or_mlflow():
    code = (
        "import sys; import precisionai.magnidata.training.recorder; "
        "assert 'torch' not in sys.modules and 'mlflow' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
