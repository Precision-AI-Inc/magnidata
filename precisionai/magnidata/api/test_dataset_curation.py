# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for cloud-union and feature-complement curation."""

import dataset_curation
import pytest


def _row(name: str, feature: str, second: str = "") -> dict[str, str]:
    return {"image_path": f"images/{name}", "feature": feature, "second": second}


def test_select_complement_excludes_current_and_ranks_deterministically() -> None:
    current = [_row("a.png", "0"), _row("b.png", "10")]
    cloud = [*current, _row("d.png", "9"), _row("c.png", "1"), _row("e.png", "1")]

    selected = dataset_curation.select_complement(current, cloud, ["image_path", "feature", "second"], {"a.png"}, 2)

    assert [dataset_curation.image_name(row, "image_path") for row in selected] == ["c.png", "e.png"]


def test_select_complement_requires_numeric_features() -> None:
    current = [{"image_path": "images/a.png", "label": "a"}]
    cloud = [*current, {"image_path": "images/b.png", "label": "b"}]

    with pytest.raises(ValueError, match="numeric features"):
        dataset_curation.select_complement(current, cloud, ["image_path", "label"], {"a.png"}, 1)
