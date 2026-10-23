# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AgriTune adapter; they need the sibling AgriTune checkout and skip without it."""

import csv
import importlib
import importlib.util
from pathlib import Path
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


def _run(preset="agritune_mlp_probe", **params):
    return SimpleNamespace(
        info=SimpleNamespace(run_id="r1"), data=SimpleNamespace(params={"preset": preset, **params}, tags={})
    )


def _staged(agritune, tmp_path, **fields):
    return agritune.StagedDataset(tmp_path / "m.csv", SimpleNamespace(num_classes=2), False, **fields)


def _source(dataset):
    return str(dataset.csv.relative_to(dataset.root))


def _write_existing_manifest(dataset, names):
    """Write an AgriTune manifest keying the synthetic images by foreign sample IDs, as `features build` input."""
    path = dataset.root / "existing" / "manifest.csv"
    path.parent.mkdir()
    with path.open("w") as handle:
        handle.write("sample_id,image_path,mask_path\n")
        for name in names:
            handle.write(f"ext-{name},../data_user/syn/images/{name},../data_user/syn/masks/{name}\n")
    return path


def test_cpu_is_used_when_requested(agritune, monkeypatch, tmp_path):
    monkeypatch.setenv("MAGNIDATA_TRAIN_DEVICE", "cpu")

    config = agritune._config(_run(), tmp_path, _staged(agritune, tmp_path))

    assert config.device == "cpu"
    assert config.trainer.precision.mode == "fp32"


def test_requesting_cuda_without_a_gpu_fails_clearly(agritune, monkeypatch, tmp_path):
    monkeypatch.setenv("MAGNIDATA_TRAIN_DEVICE", "cuda")
    torch = importlib.import_module("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA requested but not available"):
        agritune._config(_run(), tmp_path, _staged(agritune, tmp_path))


@pytest.mark.parametrize("workers", [0, 2])
def test_prepare_manifest_stages_resized_images_and_index_masks(agritune, synthetic_dataset, capsys, workers):
    np = importlib.import_module("numpy")
    image_module = importlib.import_module("PIL.Image")
    root = synthetic_dataset.root

    staged = agritune._prepare_manifest("run1", _source(synthetic_dataset), root, 16, workers=workers)

    rows = staged.manifest.read_text().splitlines()[1:]
    assert len(rows) == len(synthetic_dataset.names)
    assert staged.palette.colors == tuple(synthetic_dataset.colors)
    staged = root / "data_user" / ".agritune" / "run1"
    with image_module.open(staged / "images" / synthetic_dataset.names[0]) as image:
        assert image.size == (16, 16)
    with image_module.open(staged / "masks" / synthetic_dataset.names[0]) as mask:
        assert set(np.unique(np.asarray(mask)).tolist()) == {0, 1, 2}
    output = capsys.readouterr().out
    total = len(synthetic_dataset.names)
    assert f"scanning mask colours {total}/{total}" in output
    assert f"staging images {total}/{total}" in output


def test_cached_features_are_preloaded_in_half_precision_and_shuffled(agritune, monkeypatch, tmp_path):
    monkeypatch.setenv("MAGNIDATA_TRAIN_DEVICE", "cpu")

    config = agritune._config(
        _run(feature_provider="cached", feature_preload="host"), tmp_path, _staged(agritune, tmp_path)
    )

    assert config.feature_provider == "cached"
    assert config.feature_preload == "host"
    assert config.feature_preload_dtype == "float16"
    assert config.shuffle is True


def test_runs_without_preload_keep_agritune_defaults(agritune, monkeypatch, tmp_path):
    monkeypatch.setenv("MAGNIDATA_TRAIN_DEVICE", "cpu")

    config = agritune._config(_run(feature_provider="hybrid"), tmp_path, _staged(agritune, tmp_path))

    assert config.feature_preload == "none"
    assert config.feature_preload_dtype is None
    assert config.shuffle is False


def test_encode_dataset_fills_the_store_once(agritune, synthetic_dataset, monkeypatch, capsys):
    monkeypatch.setenv("MAGNIDATA_TRAIN_DEVICE", "cpu")
    monkeypatch.delenv("AGRITUNE_ENCODER_BASE_URL", raising=False)
    root = synthetic_dataset.root
    total = len(synthetic_dataset.names)
    staged = agritune._prepare_manifest("run1", _source(synthetic_dataset), root, 32)
    config = agritune._config(_run(feature_provider="cached", feature_preload="host"), root, staged)
    store = agritune.DirectoryFeatureStore(config.feature_store_dir)

    agritune._encode_dataset(config, store)
    agritune._encode_dataset(config, store)

    output = capsys.readouterr().out
    assert "offline fake encoder" in output
    assert f"encoding features {total}/{total} ({total} new, 0 stored)" in output
    assert f"encoding features {total}/{total} (0 new, {total} stored)" in output


def test_native_staging_keeps_the_original_images_and_mask_size(agritune, synthetic_dataset, capsys):
    image_module = importlib.import_module("PIL.Image")
    root = synthetic_dataset.root

    staged = agritune._prepare_manifest("run1", _source(synthetic_dataset), root, 16, native=True)

    with staged.manifest.open() as handle:
        rows = list(csv.DictReader(handle))
    assert {Path(row["image_path"]).parent for row in rows} == {root / "data_user" / "syn" / "images"}
    assert not any((root / "data_user" / ".agritune" / "run1" / "images").iterdir())
    height, width = synthetic_dataset.size
    with image_module.open(rows[0]["mask_path"]) as mask:
        assert mask.size == (width, height)
    assert staged.native is True
    assert f"staging {len(synthetic_dataset.names)} masks at {width}x{height}" in capsys.readouterr().out


def test_native_staging_resizes_masks_of_another_size_to_the_shared_size(agritune, synthetic_dataset, capsys):
    image_module = importlib.import_module("PIL.Image")
    name = synthetic_dataset.names[0]
    folder = synthetic_dataset.root / "data_user" / "syn"
    image_module.new("RGB", (40, 30)).save(folder / "images" / name)
    image_module.new("RGB", (40, 30)).save(folder / "masks" / name)

    staged = agritune._prepare_manifest("run1", _source(synthetic_dataset), synthetic_dataset.root, 16, native=True)

    height, width = synthetic_dataset.size
    with image_module.open(synthetic_dataset.root / "data_user" / ".agritune" / "run1" / "masks" / name) as mask:
        assert mask.size == (width, height)
    with staged.manifest.open() as handle:
        assert {Path(row["image_path"]).name for row in csv.DictReader(handle)} == set(synthetic_dataset.names)
    assert "1 others' masks are resized" in capsys.readouterr().out


def test_an_existing_store_is_reused_under_its_own_sample_ids(agritune, synthetic_dataset, monkeypatch, capsys):
    monkeypatch.setenv("MAGNIDATA_TRAIN_DEVICE", "cpu")
    monkeypatch.delenv("AGRITUNE_ENCODER_BASE_URL", raising=False)
    monkeypatch.delenv("AGRITUNE_ENCODER_MODEL", raising=False)
    root = synthetic_dataset.root
    names = synthetic_dataset.names
    existing_manifest = _write_existing_manifest(synthetic_dataset, names)
    existing = agritune.ExistingStore(existing_manifest, root / "existing" / "store", "sharded", 3, "")
    store = agritune.ShardedFeatureStore(existing.root, entries_per_shard=existing.entries_per_shard)
    fingerprint = agritune.EncoderFingerprint(model="fake-encoder", revision="fake-v1", preprocessing="")
    encoder, _ = agritune.build_encoder(base_url=None, api_key=None, model="fake-encoder", preprocessing="")
    asyncio = importlib.import_module("asyncio")
    asyncio.run(
        agritune.build_features(str(existing_manifest), store=store, encoder=encoder, encoder_fingerprint=fingerprint)
    )
    store.flush()

    staged = agritune._prepare_manifest("run1", _source(synthetic_dataset), root, 16, native=True, existing=existing)
    config = agritune._config(_run(feature_provider="cached"), root, staged)
    agritune._encode_dataset(config, agritune.ShardedFeatureStore(existing.root, entries_per_shard=3))

    with staged.manifest.open() as handle:
        assert {row["sample_id"] for row in csv.DictReader(handle)} == {f"ext-{name}" for name in names}
    assert staged.image_names == {f"ext-{name}": name for name in names}
    assert config.feature_store_dir == str(existing.root)
    assert config.store_type == "sharded"
    assert config.encoder_fingerprint.preprocessing == ""
    total = len(names)
    assert f"encoding features {total}/{total} (0 new, {total} stored)" in capsys.readouterr().out


def test_an_existing_store_missing_an_image_is_not_used(agritune, synthetic_dataset, capsys):
    root = synthetic_dataset.root
    existing_manifest = _write_existing_manifest(synthetic_dataset, synthetic_dataset.names[1:])
    existing = agritune.ExistingStore(existing_manifest, root / "existing" / "store", "sharded", 3, "")

    staged = agritune._prepare_manifest("run1", _source(synthetic_dataset), root, 16, native=True, existing=existing)

    assert staged.existing is None
    assert staged.image_names == {}
    assert "existing feature store not used: 1 of" in capsys.readouterr().out


def test_native_runs_key_their_own_features_apart_from_square_ones(agritune, monkeypatch, tmp_path):
    monkeypatch.setenv("MAGNIDATA_TRAIN_DEVICE", "cpu")

    native = agritune._config(
        _run(), tmp_path, agritune.StagedDataset(tmp_path / "m.csv", SimpleNamespace(num_classes=2), True)
    )
    square = agritune._config(_run(), tmp_path, _staged(agritune, tmp_path))

    assert native.encoder_fingerprint.preprocessing == "magnidata-native-v1"
    assert square.encoder_fingerprint.preprocessing == "magnidata-square-v1"


def test_optimiser_and_trainer_settings_reach_agritune(agritune, monkeypatch, tmp_path):
    monkeypatch.setenv("MAGNIDATA_TRAIN_DEVICE", "cpu")
    run = _run(accumulation_steps="32", grad_clip_norm="1.0", weight_decay="0.02", lr="0.001", batch_size="128")

    config = agritune._config(run, tmp_path, _staged(agritune, tmp_path))
    unclipped = agritune._config(_run(grad_clip_norm="0.0"), tmp_path, _staged(agritune, tmp_path))

    assert config.trainer.accumulation_steps == 32
    assert config.trainer.grad_clip_norm == 1.0
    assert config.optimizer.weight_decay == 0.02
    assert config.optimizer.lr == 0.001
    assert config.batch_size == 128
    assert unclipped.trainer.grad_clip_norm is None


def test_existing_store_is_read_from_the_environment(agritune, monkeypatch, tmp_path):
    monkeypatch.delenv("AGRITUNE_EXISTING_MANIFEST", raising=False)
    assert agritune.existing_store_from_env() is None
    monkeypatch.setenv("AGRITUNE_EXISTING_MANIFEST", str(tmp_path / "manifest.csv"))
    monkeypatch.setenv("AGRITUNE_EXISTING_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("AGRITUNE_EXISTING_ENTRIES_PER_SHARD", "200")

    existing = agritune.existing_store_from_env()

    assert existing == agritune.ExistingStore(tmp_path / "manifest.csv", tmp_path / "store", "sharded", 200, "")


def _staged_synthetic(agritune, dataset):
    return agritune._prepare_manifest("run1", _source(dataset), dataset.root, 16)


@pytest.mark.parametrize("preset", ["agritune_mlp_probe", "agritune_token_fpn"])
def test_every_advanced_setting_targets_a_real_agritune_field(agritune, synthetic_dataset, monkeypatch, preset):
    monkeypatch.setenv("MAGNIDATA_TRAIN_DEVICE", "cpu")
    settings = importlib.import_module("precisionai.magnidata.training.agritune_settings")
    staged = _staged_synthetic(agritune, synthetic_dataset)

    config = agritune._config(
        _run(preset, augmentation_mode="hybrid", feature_provider="hybrid"), synthetic_dataset.root, staged
    )

    for setting in settings.SETTINGS:
        if not settings.applies_to(setting, preset) or setting.key.startswith("scheduler."):
            continue
        *parents, name = setting.target.split(".")
        target = config
        for part in parents:
            target = getattr(target, part)
        assert (name in target) if isinstance(target, dict) else hasattr(target, name), setting.key


def test_advanced_settings_reach_agritune(agritune, synthetic_dataset, monkeypatch):
    monkeypatch.setenv("MAGNIDATA_TRAIN_DEVICE", "cpu")
    staged = _staged_synthetic(agritune, synthetic_dataset)
    run = _run(
        epochs="3",
        batch_size="2",
        val_fraction="0.25",
        **{
            "optimizer.name": "sgd",
            "optimizer.betas": "0.8,0.99",
            "scheduler.name": "cosine",
            "scheduler.total_steps": "0",
            "scheduler.warmup_steps": "2",
            "trainer.precision": "bf16",
            "trainer.early_stopping_patience": "5",
            "trainer.checkpoint_every_n_steps": "0",
            "val_metric_name": "loss",
            "decoder.hidden_dims": "256,128",
            "decoder.dropout": "0.1",
            "data.shuffle": "false",
            "data.num_workers": "4",
            "feature_augmentation.patch_dropout_probability": "0.1",
        },
    )

    config = agritune._config(run, synthetic_dataset.root, staged)

    assert config.optimizer.name == "sgd"
    assert config.optimizer.betas == (0.8, 0.99)
    # 8 samples, 6 for training, batches of 2, no accumulation: 3 steps per epoch over 3 epochs.
    assert (config.scheduler.name, config.scheduler.total_steps, config.scheduler.warmup_steps) == ("cosine", 9, 2)
    assert config.trainer.precision.mode == "bf16"
    assert config.trainer.early_stopping_patience == 5
    assert config.trainer.checkpoint_every_n_steps is None
    assert (config.val_metric_name, config.higher_is_better) == ("loss", False)
    assert config.decoder_kwargs == {"hidden_dims": [256, 128], "dropout": 0.1}
    assert config.shuffle is False
    assert config.num_workers == 4
    assert config.feature_augmentation.patch_dropout_probability == 0.1


def test_defaults_leave_the_schedule_constant_and_augmentation_off(agritune, synthetic_dataset, monkeypatch):
    monkeypatch.setenv("MAGNIDATA_TRAIN_DEVICE", "cpu")
    staged = _staged_synthetic(agritune, synthetic_dataset)

    plain = agritune._config(_run(), synthetic_dataset.root, staged)
    augmented = agritune._config(
        _run(augmentation_mode="online", feature_provider="hybrid", **{"augmentation.rotation_max_degrees": "15"}),
        synthetic_dataset.root,
        staged,
    )

    assert plain.scheduler is None
    assert plain.augmentation.geometric.horizontal_flip_probability == 0.0
    assert augmented.augmentation.geometric.horizontal_flip_probability == 0.5
    assert augmented.augmentation.geometric.rotation_max_degrees == 15.0
    assert (plain.val_metric_name, plain.higher_is_better) == ("mean_iou", True)


def test_class_weights_need_one_weight_per_class(agritune, synthetic_dataset, monkeypatch):
    monkeypatch.setenv("MAGNIDATA_TRAIN_DEVICE", "cpu")
    torch = importlib.import_module("torch")
    staged = _staged_synthetic(agritune, synthetic_dataset)

    config = agritune._config(_run(**{"loss.class_weights": "0.5,1,2"}), synthetic_dataset.root, staged)
    with pytest.raises(ValueError, match="2 weights for 3 classes"):
        agritune._config(_run(**{"loss.class_weights": "1,2"}), synthetic_dataset.root, staged)

    assert torch.equal(config.loss.class_weights, torch.tensor([0.5, 1.0, 2.0]))


def test_token_fpn_cls_fusion_and_preload_precision(agritune, synthetic_dataset, monkeypatch):
    monkeypatch.setenv("MAGNIDATA_TRAIN_DEVICE", "cpu")
    staged = _staged_synthetic(agritune, synthetic_dataset)
    run = _run(
        "agritune_token_fpn",
        feature_provider="cached",
        feature_preload="host",
        **{"decoder.cls_fusion": "film", "data.feature_preload_dtype": "float32"},
    )

    config = agritune._config(run, synthetic_dataset.root, staged)
    unpreloaded = agritune._config(
        _run("agritune_token_fpn", feature_provider="hybrid"), synthetic_dataset.root, staged
    )

    assert config.decoder_kwargs["cls_fusion"] is agritune.CLSFusion.FILM
    assert config.feature_preload_dtype == "float32"
    assert unpreloaded.feature_preload_dtype is None
