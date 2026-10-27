# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Fixtures shared by the API and training tests."""

import csv
import shutil
import socket
from dataclasses import dataclass
from pathlib import Path

import mlflow
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


@pytest.fixture(scope="session")
def mlflow_db_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return an empty, fully migrated MLflow sqlite database to copy for each test.

    MLflow migrates a new sqlite file's schema the first time it opens it, which takes seconds.
    Doing that once per session and copying the file gives every test its own database in
    milliseconds.
    """
    path = tmp_path_factory.mktemp("mlflow-template") / "mlflow.db"
    mlflow.MlflowClient(tracking_uri=f"sqlite:///{path}").search_experiments(max_results=1)
    return path


@pytest.fixture
def mlflow_db(tmp_path: Path, mlflow_db_template: Path) -> Path:
    """Return a private, already migrated MLflow sqlite database inside ``tmp_path``."""
    path = tmp_path / "mlflow.db"
    shutil.copyfile(mlflow_db_template, path)
    return path


@pytest.fixture
def closed_port_url() -> str:
    """Return an http URL on a local port nothing listens on, which refuses connections at once.

    A just-released ephemeral port is refused immediately everywhere. A hard-coded low port such
    as 1 is silently dropped by some firewalls (WSL among them), which turns "refused" into a
    connect timeout of many seconds, multiplied by the HTTP retries.
    """
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{sock.getsockname()[1]}"


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
