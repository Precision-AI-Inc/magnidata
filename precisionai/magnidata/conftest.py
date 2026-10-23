# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Fixtures shared by the API and training tests."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@dataclass
class SyntheticDataset:
    """A tiny dataset in the repo's layout: CSV, images and colour masks under ``root``."""

    root: Path
    csv: Path
    names: list[str]
    colors: list[tuple[int, int, int]]
    size: tuple[int, int]


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> SyntheticDataset:
    """Write 8 small PNG images with colour masks (two classes on black) and a CSV."""
    rng = np.random.default_rng(0)
    colors = [(200, 30, 30), (30, 160, 60)]
    width, height = 32, 24
    folder = tmp_path / "data_user" / "syn"
    (folder / "images").mkdir(parents=True)
    (folder / "masks").mkdir()
    names = []
    for i in range(8):
        name = f"img_{i:02d}.png"
        names.append(name)
        Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8)).save(folder / "images" / name)
        mask = np.zeros((height, width, 3), dtype=np.uint8)
        mask[:, : width // 3] = colors[0]
        mask[height // 2 :, width // 2 :] = colors[1]
        Image.fromarray(mask).save(folder / "masks" / name)
    csv_path = tmp_path / "data_user" / "syn.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_path", "width", "height"])
        writer.writerows([f"data_user/syn/images/{n}", width, height] for n in names)
    return SyntheticDataset(tmp_path, csv_path, names, colors, (height, width))
