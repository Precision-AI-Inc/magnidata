# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for persisted, on-demand segmentation inference."""

import pytest

from precisionai.magnidata.training import contract

torch = pytest.importorskip("torch")
infer = pytest.importorskip("precisionai.magnidata.training.infer")
models = pytest.importorskip("precisionai.magnidata.training.models")


def test_infer_logs_prediction_from_checkpoint(mlflow_tracking, synthetic_dataset, tmp_path):
    uri, client, run_id = mlflow_tracking()
    client.set_tag(run_id, contract.TAG_DATASET_SOURCE, "data_user/syn.csv")
    model = models.build_model("tiny_unet", 3, pretrained=False)
    checkpoint = tmp_path / "last.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "preset": "tiny_unet",
            "num_classes": 3,
            "palette": {
                0: {"name": "background", "rgb": [0, 0, 0]},
                1: {"name": "class_1", "rgb": list(synthetic_dataset.colors[0])},
                2: {"name": "class_2", "rgb": list(synthetic_dataset.colors[1])},
            },
            "image_size": 32,
            "epoch": 7,
        },
        checkpoint,
    )
    client.log_artifact(run_id, str(checkpoint), "checkpoints")

    name = synthetic_dataset.names[0]
    assert infer.infer(run_id, name, synthetic_dataset.root, uri, "cpu") == 7
    artifacts = client.list_artifacts(run_id, "magnidata/predictions/epoch_0007")
    assert [artifact.path for artifact in artifacts] == [f"magnidata/predictions/epoch_0007/{name}"]
