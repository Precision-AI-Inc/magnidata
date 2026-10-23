# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the derived per-sample axes, using hand-written per-sample CSVs.

Run:  python -m pytest precisionai/magnidata/api/test_run_axes.py -q
"""

import csv
import io

import numpy as np
import pytest
import run_axes

TOTAL = {"a.png": [0.9, 0.8, 0.7, 0.6], "b.png": [0.5, 0.5, 0.4, 0.3], "c.png": [0.2, 0.9, 0.1, 0.1]}
DICE = {"a.png": [0.5, 0.4, 0.4, 0.3], "b.png": [0.3, 0.3, 0.2, 0.2], "c.png": [0.1, 0.6, 0.05, 0.05]}
SPLITS = {"a.png": "train", "b.png": "train", "c.png": "train"}


def _write(tmp_path, epochs, total=TOTAL, dice=DICE, splits=SPLITS, skip=None, extras=False):
    """Write one CSV per epoch (1-based); ``skip`` is a {epoch: image} to omit."""
    files = []
    for epoch in epochs:
        path = tmp_path / f"epoch_{epoch:04d}.csv"
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["image_name", "split", "loss_total", "loss_dice", *(["iou", "confidence"] if extras else [])]
            )
            for name in total:
                if (skip or {}).get(epoch) == name:
                    continue
                row = [name, splits[name], total[name][epoch - 1], dice[name][epoch - 1]]
                writer.writerow(row + ([0.5, 0.9] if extras else []))
        files.append((epoch, path))
    return files


def _column(axes, name):
    return dict(zip(axes.names, axes.columns[name], strict=True))


def test_mean_delta_slope_rank_match_hand_computation(tmp_path):
    files = _write(tmp_path, [1, 2, 3, 4])
    axes = run_axes.derive_axes(files)
    assert axes.epoch == 4
    assert axes.scored_epochs == [1, 2, 3, 4]

    for name, series in TOTAL.items():
        assert _column(axes, "loss_total")[name] == pytest.approx(series[-1])
        assert _column(axes, "loss_total_mean")[name] == pytest.approx(np.mean(series))
        assert _column(axes, "loss_total_delta")[name] == pytest.approx(series[0] - series[-1])
        assert _column(axes, "loss_total_slope")[name] == pytest.approx(np.polyfit([1, 2, 3, 4], series, 1)[0])
    # final-epoch totals are a=0.6, b=0.3, c=0.1 -> ranks 1, 0.5, 0
    assert _column(axes, "loss_total_rank") == pytest.approx({"a.png": 1.0, "b.png": 0.5, "c.png": 0.0})
    assert _column(axes, "loss_dice_mean")["a.png"] == pytest.approx(np.mean(DICE["a.png"]))


def test_slope_uses_only_the_last_five_epochs(tmp_path):
    total = {"a.png": [9.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], "b.png": [1.0] * 7}
    files = _write(tmp_path, range(1, 8), total=total, dice=total, splits={"a.png": "train", "b.png": "train"})
    slope = _column(run_axes.derive_axes(files), "loss_total_slope")
    assert slope["a.png"] == pytest.approx(np.polyfit([3, 4, 5, 6, 7], total["a.png"][2:], 1)[0])
    assert slope["b.png"] == pytest.approx(0.0)


def test_hardness_counts_epochs_in_worst_fifth(tmp_path):
    axes = run_axes.derive_axes(_write(tmp_path, [1, 2, 3, 4]))
    assert _column(axes, "hardness") == pytest.approx({"a.png": 0.75, "b.png": 0.0, "c.png": 0.25})


def test_ranks_are_computed_within_each_split(tmp_path):
    splits = {"a.png": "train", "b.png": "train", "c.png": "val"}
    axes = run_axes.derive_axes(_write(tmp_path, [1, 2], splits=splits))
    ranks = _column(axes, "loss_total_rank")
    assert ranks["c.png"] == pytest.approx(0.5)  # alone in its split
    assert ranks["a.png"] == pytest.approx(1.0)
    assert ranks["b.png"] == pytest.approx(0.0)
    assert axes.splits == ["train", "train", "val"]


def test_earlier_epoch_ignores_later_ones(tmp_path):
    files = _write(tmp_path, [1, 2, 3, 4])
    axes = run_axes.derive_axes(files, epoch=2)
    assert axes.epoch == 2
    assert _column(axes, "loss_total")["a.png"] == pytest.approx(0.8)
    assert _column(axes, "loss_total_mean")["a.png"] == pytest.approx(np.mean(TOTAL["a.png"][:2]))
    assert _column(axes, "loss_total_delta")["a.png"] == pytest.approx(0.9 - 0.8)
    assert _column(axes, "loss_total_slope")["a.png"] == pytest.approx(np.polyfit([1, 2], TOTAL["a.png"][:2], 1)[0])


def test_single_epoch_has_nan_slope_and_zero_delta(tmp_path):
    axes = run_axes.derive_axes(_write(tmp_path, [1]))
    assert np.isnan(axes.columns["loss_total_slope"]).all()
    assert (axes.columns["loss_total_delta"] == 0).all()
    assert axes.columns["loss_total_mean"] == pytest.approx(axes.columns["loss_total"])


def test_sample_missing_from_one_epoch(tmp_path):
    files = _write(tmp_path, [1, 2, 3, 4], skip={2: "b.png"})
    axes = run_axes.derive_axes(files)
    values = TOTAL["b.png"]
    assert _column(axes, "loss_total_mean")["b.png"] == pytest.approx(np.mean([values[0], values[2], values[3]]))
    assert _column(axes, "loss_total_slope")["b.png"] == pytest.approx(
        np.polyfit([1, 3, 4], [values[0], values[2], values[3]], 1)[0]
    )
    assert _column(axes, "hardness")["b.png"] == 0.0


def test_sample_absent_from_selected_epoch_is_nan(tmp_path):
    files = _write(tmp_path, [1, 2], skip={2: "c.png"})
    axes = run_axes.derive_axes(files)
    assert np.isnan(_column(axes, "loss_total")["c.png"])
    assert np.isnan(_column(axes, "loss_total_delta")["c.png"])
    assert _column(axes, "loss_total_mean")["c.png"] == pytest.approx(0.2)


def test_optional_columns_and_term_selection(tmp_path):
    files = _write(tmp_path, [1, 2], extras=True)
    axes = run_axes.derive_axes(files, terms=["dice"])
    assert "iou" in axes.columns
    assert "confidence" in axes.columns
    assert "loss_dice_rank" in axes.columns
    assert "loss_total" not in axes.columns
    assert "hardness" in axes.columns


def test_unscored_epoch_and_empty_input(tmp_path):
    files = _write(tmp_path, [1, 2])
    with pytest.raises(ValueError, match="epoch 7 has no per-sample file"):
        run_axes.derive_axes(files, epoch=7)
    with pytest.raises(ValueError, match="no per-sample files"):
        run_axes.derive_axes([])


def test_invalid_file_is_rejected(tmp_path):
    bad = tmp_path / "epoch_0001.csv"
    bad.write_text("image_name,split,loss_dice\na.png,train,0.1\n")
    with pytest.raises(ValueError, match="loss_total"):
        run_axes.derive_axes([(1, bad)])


def test_load_epoch_matrix(tmp_path):
    names, epochs, matrix = run_axes.load_epoch_matrix(_write(tmp_path, [1, 2, 3], skip={2: "a.png"}), "total")
    assert names == ["a.png", "b.png", "c.png"]
    assert epochs == [1, 2, 3]
    assert np.isnan(matrix[0, 1])
    assert matrix[1].tolist() == TOTAL["b.png"][:3]
    _, epochs, matrix = run_axes.load_epoch_matrix(_write(tmp_path, [1, 2, 3]), "dice", up_to_epoch=2)
    assert epochs == [1, 2]
    assert matrix.shape == (3, 2)
    with pytest.raises(ValueError, match="unknown loss term"):
        run_axes.load_epoch_matrix(_write(tmp_path, [1]), "nope")


def test_percentile_rank_ties_and_nan():
    ranks = run_axes.percentile_rank(np.array([3.0, 1.0, 1.0, np.nan, 2.0]))
    assert ranks[:3] == pytest.approx([1.0, 1 / 6, 1 / 6])
    assert np.isnan(ranks[3])
    assert ranks[4] == pytest.approx(2 / 3)
    assert run_axes.percentile_rank(np.array([5.0])).tolist() == [0.5]
    assert np.isnan(run_axes.percentile_rank(np.array([np.nan]))).all()


def test_csv_output_quotes_and_blanks(tmp_path):
    total = {'we,ird"name.png': [1.0, 2.0], "b.png": [3.0, 4.0]}
    files = _write(tmp_path, [1, 2], total=total, dice=total, splits=dict.fromkeys(total, "train"), skip={2: "b.png"})
    text = run_axes.derive_axes(files).to_csv()
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0][:3] == ["image_name", "split", "loss_total"]
    assert rows[1][0] == 'we,ird"name.png'
    assert rows[2][2] == ""  # b.png has no value at the selected epoch


def test_cache_hits_misses_and_eviction(tmp_path):
    axes = run_axes.derive_axes(_write(tmp_path, [1, 2]))
    cache = run_axes.RunAxesCache(capacity=8)
    calls = []

    def compute():
        calls.append(1)
        return axes

    assert cache.get_or_compute(("r", 1, 2), compute) is axes
    assert cache.get_or_compute(("r", 1, 2), compute) is axes
    assert len(calls) == 1
    cache.get_or_compute(("r", 1, 3), compute)  # a new latest epoch is a different key
    assert len(calls) == 2

    for i in range(8):
        cache.get_or_compute(("other", i, 1), compute)
    assert len(cache) == 8
    assert ("r", 1, 2) not in cache
    cache.clear()
    assert len(cache) == 0
