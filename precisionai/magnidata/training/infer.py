# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Run one persisted segmentation inference from an MLflow checkpoint."""

import argparse
import csv
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from PIL import Image

from precisionai.magnidata.training import contract
from precisionai.magnidata.training.data import IMAGENET_MEAN, IMAGENET_STD
from precisionai.magnidata.training.dataset_check import resolve_under
from precisionai.magnidata.training.models import build_model
from precisionai.magnidata.training.train import resolve_device


def _image_path(client: Any, run_id: str, image_name: str, data_root: Path) -> Path:
    """Resolve one image from the dataset linked to a run."""
    run = client.get_run(run_id)
    source = run.data.tags.get(contract.TAG_DATASET_SOURCE)
    csv_path = resolve_under(data_root, source or "")
    if csv_path is None or not csv_path.is_file():
        raise FileNotFoundError(f"dataset not found: {source}")
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        column = contract.image_path_column(reader.fieldnames or [])
        relative = next(
            (row[column] for row in reader if column and contract.image_key(row[column]) == image_name),
            None,
        )
    image = resolve_under(data_root, relative or "")
    if image is None or not image.is_file():
        raise FileNotFoundError(f"image {image_name} not found in {source}")
    return image


def infer(run_id: str, image_name: str, data_root: Path, tracking_uri: str, device_name: str = "auto") -> int:
    """Generate, upload and return the checkpoint epoch for one prediction mask."""
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    device = resolve_device(device_name)
    with tempfile.TemporaryDirectory(prefix="magnidata-inference-") as directory:
        checkpoint_path = Path(client.download_artifacts(run_id, contract.CHECKPOINT_ARTIFACT, directory))
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        epoch = int(checkpoint["epoch"])
        # Checkpoints from before native-resolution training only record a square image_size.
        width, height = checkpoint.get("input_size") or (int(checkpoint["image_size"]),) * 2
        model = build_model(str(checkpoint["preset"]), int(checkpoint["num_classes"]), pretrained=False)
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()

        image_path = _image_path(client, run_id, image_name, data_root)
        with Image.open(image_path) as opened:
            rgb = opened.convert("RGB").resize((int(width), int(height)), Image.Resampling.BILINEAR)
        pixels = torch.from_numpy(np.asarray(rgb).copy()).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        with torch.inference_mode():
            prediction = model(((pixels - mean) / std).unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy()

        palette = checkpoint["palette"]
        entries = [palette.get(index) or palette.get(str(index)) for index in range(len(palette))]
        if any(entry is None or "rgb" not in entry for entry in entries):
            raise ValueError("checkpoint palette is incomplete")
        colors = np.array([entry["rgb"] for entry in entries if entry is not None], dtype=np.uint8)
        output = Path(directory) / Path(contract.prediction_path(epoch, image_name)).name
        Image.fromarray(colors[np.minimum(prediction, len(colors) - 1)]).save(output)
        client.log_artifact(run_id, str(output), str(Path(contract.prediction_path(epoch, image_name)).parent))
    return epoch


def main(argv: Sequence[str] | None = None) -> int:
    """Run one inference request from command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate one MagniData prediction artifact.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATA_ROOT", ".")))
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5052"))
    parser.add_argument("--device", default=os.environ.get("MAGNIDATA_TRAIN_DEVICE", "auto"))
    args = parser.parse_args(argv)
    epoch = infer(args.run_id, args.image, args.data_root, args.tracking_uri, args.device)
    print(f"prediction saved for {args.image} at epoch {epoch}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
