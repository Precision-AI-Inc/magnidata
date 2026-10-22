# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AgriTune adapter; they need the sibling AgriTune checkout and skip without it."""

import importlib
import importlib.util
from types import ModuleType, SimpleNamespace

import pytest

# Looked up without importing: importing the adapter pulls in torch, which must not happen at collection time.
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("precisionai.agritune") is None, reason="needs the sibling AgriTune checkout installed"
)


@pytest.fixture(scope="module")
def agritune() -> ModuleType:
    return importlib.import_module("precisionai.magnidata.training.agritune")


@pytest.mark.parametrize(
    ("loss", "term"),
    [("ce", "cce"), ("ce_dice", "cce"), ("dice", "cce"), ("bce", "bce"), ("bce_dice", "bce")],
)
def test_the_scaled_term_follows_the_objective(agritune, loss, term):
    assert agritune._primary_term({"agritune_loss": loss}) == term


def test_missing_objective_defaults_to_cross_entropy(agritune):
    assert agritune._primary_term({}) == "cce"


def test_every_agritune_preset_maps_to_a_decoder(agritune):
    assert set(agritune.PRESET_DECODERS) == {"agritune_mlp_probe", "agritune_token_fpn"}


def _run(preset="agritune_mlp_probe"):
    return SimpleNamespace(info=SimpleNamespace(run_id="r1"), data=SimpleNamespace(params={"preset": preset}, tags={}))


def _palette(agritune):
    return SimpleNamespace(num_classes=2)


def test_cpu_is_used_when_requested(agritune, monkeypatch, tmp_path):
    monkeypatch.setenv("MAGNIDATA_TRAIN_DEVICE", "cpu")

    config = agritune._config(_run(), tmp_path, tmp_path / "m.csv", _palette(agritune))

    assert config.device == "cpu"
    assert config.trainer.precision.mode == "fp32"


def test_requesting_cuda_without_a_gpu_fails_clearly(agritune, monkeypatch, tmp_path):
    monkeypatch.setenv("MAGNIDATA_TRAIN_DEVICE", "cuda")
    torch = importlib.import_module("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA requested but not available"):
        agritune._config(_run(), tmp_path, tmp_path / "m.csv", _palette(agritune))


@pytest.mark.parametrize("workers", [0, 2])
def test_prepare_manifest_stages_resized_images_and_index_masks(agritune, synthetic_dataset, capsys, workers):
    np = importlib.import_module("numpy")
    image_module = importlib.import_module("PIL.Image")
    root = synthetic_dataset.root

    manifest, palette = agritune._prepare_manifest(
        "run1", str(synthetic_dataset.csv.relative_to(root)), root, 16, workers
    )

    rows = manifest.read_text().splitlines()[1:]
    assert len(rows) == len(synthetic_dataset.names)
    assert palette.colors == tuple(synthetic_dataset.colors)
    staged = root / "data_user" / ".agritune" / "run1"
    with image_module.open(staged / "images" / synthetic_dataset.names[0]) as image:
        assert image.size == (16, 16)
    with image_module.open(staged / "masks" / synthetic_dataset.names[0]) as mask:
        assert set(np.unique(np.asarray(mask)).tolist()) == {0, 1, 2}
    output = capsys.readouterr().out
    total = len(synthetic_dataset.names)
    assert f"scanning mask colours {total}/{total}" in output
    assert f"staging images {total}/{total}" in output
