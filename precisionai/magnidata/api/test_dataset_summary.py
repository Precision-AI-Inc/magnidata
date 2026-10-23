# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dataset summary used by the workspace detail panel."""

import csv
from pathlib import Path

import dataset_summary


def _write(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_numeric_columns_get_histograms_and_text_columns_are_skipped(tmp_path: Path) -> None:
    rows = [[f"img{i}.png", str(i), "a"] for i in range(10)]
    path = _write(tmp_path / "d.csv", ["image_path", "value", "label"], rows)

    summary = dataset_summary.summarize_dataset(path, tmp_path)

    assert summary["row_count"] == 10
    assert [c["name"] for c in summary["numeric_columns"]] == ["value"]
    column = summary["numeric_columns"][0]
    assert (column["min"], column["max"], column["mean"]) == (0, 9, 4.5)
    assert sum(column["histogram"]) == 10
    assert len(summary["sample_images"]) == 8


def test_constant_columns_sort_after_varying_ones(tmp_path: Path) -> None:
    rows = [[f"img{i}.png", "5", str(i)] for i in range(4)]
    path = _write(tmp_path / "d.csv", ["image_path", "flat", "varies"], rows)

    summary = dataset_summary.summarize_dataset(path, tmp_path)

    assert [c["name"] for c in summary["numeric_columns"]] == ["varies", "flat"]
    assert sum(summary["numeric_columns"][1]["histogram"]) == 4


def test_mask_coverage_counts_sampled_rows_with_masks(tmp_path: Path) -> None:
    (tmp_path / "ds" / "images").mkdir(parents=True)
    (tmp_path / "ds" / "masks").mkdir()
    (tmp_path / "ds" / "masks" / "a.png").write_bytes(b"x")
    rows = [["ds/images/a.png"], ["ds/images/b.png"]]
    path = _write(tmp_path / "d.csv", ["image_path"], rows)

    summary = dataset_summary.summarize_dataset(path, tmp_path)

    assert summary["mask_coverage"] == {"checked": 2, "with_masks": 1}


def test_count_rows_reflects_file_changes(tmp_path: Path) -> None:
    path = _write(tmp_path / "d.csv", ["image_path"], [["a"], ["b"]])
    assert dataset_summary.count_rows(path) == 2

    _write(path, ["image_path"], [["a"], ["b"], ["c"]])

    assert dataset_summary.count_rows(path) == 3
    assert dataset_summary.count_rows(tmp_path / "missing.csv") is None
