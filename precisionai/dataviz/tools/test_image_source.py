# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for image_source's pure helpers (no disk datalake, no network)."""

from .image_source import find_path_column


def test_prefers_exact_image_path():
    assert find_path_column(["id", "image_path", "label_path"]) == "image_path"


def test_prefers_image_and_path_together():
    assert find_path_column(["id", "img_path", "note"]) == "img_path"


def test_falls_back_to_any_path_column():
    assert find_path_column(["id", "mask_path", "note"]) == "mask_path"


def test_falls_back_to_first_column():
    assert find_path_column(["id", "note"]) == "id"
