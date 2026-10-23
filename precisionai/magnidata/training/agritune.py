# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Adapt a queued MagniData dataset/run into AgriTune's training service."""

import argparse
import asyncio
import csv
import math
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from PIL import Image

from precisionai.magnidata.training import agritune_settings, contract, data
from precisionai.magnidata.training.train import default_num_workers, describe_device, log, resolve_device

try:
    import mlflow
    import torch

    from precisionai.agritune.augmentations.image.pipeline import AugmentationMode
    from precisionai.agritune.features.keys import EncoderFingerprint
    from precisionai.agritune.features.precompute import PrecomputeStats
    from precisionai.agritune.features.store import DirectoryFeatureStore, ShardedFeatureStore
    from precisionai.agritune.integrations.magnidata import MagniDataOptions, run_magnidata_training
    from precisionai.agritune.optimization.optimizers import OptimizerConfig
    from precisionai.agritune.optimization.schedulers import SchedulerConfig
    from precisionai.agritune.services.encoder_selection import build_encoder
    from precisionai.agritune.services.feature_service import build_features
    from precisionai.agritune.services.training_service import AugmentationSelection, TrainingRunConfig
    from precisionai.agritune.tasks.segmentation.decoders.token_fpn import CLSFusion
    from precisionai.agritune.tasks.segmentation.losses import SegmentationLossConfig
    from precisionai.agritune.training.precision import PrecisionConfig
    from precisionai.agritune.training.trainer import TrainerConfig
except ImportError as err:
    # An AgriTune checkout without the MagniData integration imports partly, so name what is missing.
    raise ImportError(
        f"AgriTune is required for this preset ({err}); build the trainer with the sibling agritune checkout"
    ) from err

DEFAULT_TRACKING_URI = "http://localhost:5052"
# Preloaded patch tokens are stored in half precision: half the memory (SEEDPLANT: ~38 GB instead of ~75 GB).
PRELOAD_DTYPE = "float16"
PRESET_DECODERS = {
    "agritune_mlp_probe": "mlp_probe",
    "agritune_token_fpn": "token_fpn",
    "agritune_aspp": "aspp",
    "agritune_ppm": "ppm",
    "agritune_segmenter": "segmenter",
    "agritune_mask_former": "mask_former",
}


def _integer(params: dict[str, str], key: str, default: int) -> int:
    return int(params.get(key, default))


def _number(params: dict[str, str], key: str, default: float) -> float:
    return float(params.get(key, default))


def _primary_term(params: dict[str, str]) -> str:
    """Return the MagniData loss term AgriTune's ``ce_weight`` scales: BCE for the ``bce*`` objectives, else CE."""
    return "bce" if params.get("agritune_loss", "ce_dice").startswith("bce") else "cce"


def _boolean(params: dict[str, str], key: str, default: bool) -> bool:
    return params.get(key, str(default)).lower() == "true"


class _StageJob(NamedTuple):
    """One sample to stage: its sources, where to write, the mask's ``(width, height)`` and the palette.

    ``image_path`` is set for square staging, which writes the image resized to ``mask_size`` too; native staging
    leaves the image file alone.
    """

    source_image: Path
    source_mask: Path
    image_path: Path | None
    mask_path: Path
    mask_size: tuple[int, int]
    colors: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class ExistingStore:
    """An AgriTune feature store built earlier, with the manifest whose sample IDs and image files key it."""

    manifest: Path
    root: Path
    store_type: str
    entries_per_shard: int
    preprocessing: str


@dataclass(frozen=True)
class StagedDataset:
    """The run's AgriTune manifest, its palette, and how its features are found."""

    manifest: Path
    palette: data.Palette
    native: bool
    existing: ExistingStore | None = None
    image_names: dict[str, str] = field(default_factory=dict)


def existing_store_from_env() -> ExistingStore | None:
    """Return the existing feature store configured for this trainer, or None.

    Set ``AGRITUNE_EXISTING_MANIFEST`` and ``AGRITUNE_EXISTING_STORE`` (paths as the trainer sees them), plus
    optionally ``AGRITUNE_EXISTING_STORE_TYPE`` (``sharded``, the default, or ``directory``),
    ``AGRITUNE_EXISTING_ENTRIES_PER_SHARD`` and ``AGRITUNE_EXISTING_PREPROCESSING`` (the label the store was
    encoded with, ``""`` for ``agritune features build``'s default).
    """
    manifest = os.environ.get("AGRITUNE_EXISTING_MANIFEST")
    root = os.environ.get("AGRITUNE_EXISTING_STORE")
    if not manifest or not root:
        return None
    return ExistingStore(
        manifest=Path(manifest),
        root=Path(root),
        store_type=os.environ.get("AGRITUNE_EXISTING_STORE_TYPE", "sharded"),
        entries_per_shard=int(os.environ.get("AGRITUNE_EXISTING_ENTRIES_PER_SHARD", "1000")),
        preprocessing=os.environ.get("AGRITUNE_EXISTING_PREPROCESSING", ""),
    )


def _stage_one(job: _StageJob) -> None:
    """Write one sample's mask decoded to class indices at ``mask_size`` (and, when square, its resized image)."""
    if job.image_path is not None:
        with Image.open(job.source_image) as image:
            image.convert("RGB").resize(job.mask_size, Image.Resampling.BILINEAR).save(job.image_path)
    with Image.open(job.source_mask) as mask:
        rgb = mask.convert("RGB")
        if rgb.size != job.mask_size:
            rgb = rgb.resize(job.mask_size, Image.Resampling.NEAREST)
    Image.fromarray(data.decode_mask(np.asarray(rgb), data.Palette(job.colors)).astype(np.uint8), mode="L").save(
        job.mask_path
    )


def _stage(jobs: list[_StageJob], workers: int) -> None:
    total = len(jobs)
    every = data.progress_every(total)
    if workers > 1 and total > 1:
        # spawn, not fork: this process already runs MLflow client threads
        with ProcessPoolExecutor(
            max_workers=min(workers, total), mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for done, _ in enumerate(pool.map(_stage_one, jobs, chunksize=16), start=1):
                if done % every == 0 or done == total:
                    log(f"staging images {done}/{total}")
    else:
        for done, job in enumerate(jobs, start=1):
            _stage_one(job)
            if done % every == 0 or done == total:
                log(f"staging images {done}/{total}")


def _match_existing(samples: list[data.Sample], existing: ExistingStore) -> dict[str, tuple[str, Path]] | None:
    """Map each sample's image name to the existing manifest's (sample_id, image file), or None if any is missing."""
    by_name: dict[str, tuple[str, Path]] = {}
    with existing.manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image = existing.manifest.parent / row["image_path"]
            by_name[contract.image_key(row["image_path"])] = (row["sample_id"], image)
    missing = [sample.image_name for sample in samples if contract.image_key(sample.image_name) not in by_name]
    if missing:
        log(
            f"existing feature store not used: {len(missing)} of {len(samples)} images are not in "
            f"{existing.manifest} (e.g. {missing[0]}); encoding instead"
        )
        return None
    return {sample.image_name: by_name[contract.image_key(sample.image_name)] for sample in samples}


def _prepare_manifest(
    run_id: str,
    dataset_source: str,
    data_root: Path,
    image_size: int,
    *,
    workers: int = 0,
    native: bool = False,
    existing: ExistingStore | None = None,
) -> StagedDataset:
    """Stage masks (and square images) for AgriTune and write its manifest.

    ``native`` keeps every image file as it is, so the encoder sees the original resolution, and decodes masks at the
    size most images share (others' masks are resized to it, as AgriTune's ``geometric.resize`` would). ``existing``
    (native only) reuses a feature store built from another manifest when every image matches one of its rows by
    file name.
    """
    source = data_root / dataset_source
    samples, report = data.resolve_samples(source, data_root)
    if not samples:
        raise ValueError(
            f"dataset has no trainable image/mask pairs ({report.total_rows} rows, "
            f"{len(report.missing_image)} missing images, {len(report.missing_mask)} missing masks)"
        )
    total = len(samples)
    log(f"scanning {total} masks for the colour palette with {max(workers, 1)} process(es)")
    palette = data.build_palette([sample.mask_path for sample in samples], workers=workers, progress=log)
    root = data_root / "data_user" / ".agritune" / run_id
    images = root / "images"
    masks = root / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    if native:
        mask_size, resized = data.native_size(samples, workers)
        log(
            f"native resolution {mask_size[0]}x{mask_size[1]}: shared by most images; {resized} others' masks are resized to it"
        )
    else:
        mask_size = (image_size, image_size)
    jobs = [
        _StageJob(
            sample.image_path,
            sample.mask_path,
            None if native else images / f"{Path(sample.image_name).stem}.png",
            masks / f"{Path(sample.image_name).stem}.png",
            mask_size,
            palette.colors,
        )
        for sample in samples
    ]
    what = "masks" if native else "images"
    log(f"staging {total} {what} at {mask_size[0]}x{mask_size[1]} with {max(workers, 1)} process(es)")
    _stage(jobs, workers)
    matched = _match_existing(samples, existing) if native and existing is not None else None
    rows = []
    for sample, job in zip(samples, jobs, strict=True):
        sample_id, image = (sample.image_name, job.image_path or sample.image_path)
        if matched is not None:
            sample_id, image = matched[sample.image_name]
        rows.append((sample_id, image, job.mask_path, sample.csv_split or ""))
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "image_path", "mask_path", "source_split"])
        writer.writerows(rows)
    if matched is None or existing is None:
        return StagedDataset(manifest, palette, native)
    log(f"reusing the existing feature store {existing.root} for all {total} images (no encoding)")
    names = {sample_id: name for name, (sample_id, _) in matched.items()}
    return StagedDataset(manifest, palette, native, existing, names)


def _config(run: Any, data_root: Path, staged: StagedDataset) -> TrainingRunConfig:
    params = dict(run.data.params)
    preset = run.data.tags.get("magnidata.preset", params.get("preset", ""))
    decoder = PRESET_DECODERS.get(preset)
    if decoder is None:
        raise ValueError(f"unsupported AgriTune preset: {preset}")
    base_url = os.environ.get("AGRITUNE_ENCODER_BASE_URL") or None
    model = os.environ.get("AGRITUNE_ENCODER_MODEL", "pai-embedding" if base_url else "fake-encoder")
    revision = None if base_url else "fake-v1"
    augmentation_mode = AugmentationMode(params.get("augmentation_mode", "none"))
    augmentation = AugmentationSelection(mode=augmentation_mode)
    device = resolve_device(os.environ.get("MAGNIDATA_TRAIN_DEVICE", "auto"))
    log(f"training on {describe_device(device)}")
    preload = params.get("feature_preload", "none")
    log(f"encoder: {'hosted ' + model if base_url else 'offline fake encoder (no AGRITUNE_ENCODER_BASE_URL)'}")
    existing = staged.existing
    store_dir = data_root / "data_user" / ".agritune" / "features" / model.replace("/", "_")
    if existing is not None:
        preprocessing = existing.preprocessing
    else:
        # Native and square features of the same image differ, so their cache keys must too.
        preprocessing = "magnidata-native-v1" if staged.native else "magnidata-square-v1"
    clip = _number(params, "grad_clip_norm", 0.0)
    config = TrainingRunConfig(
        manifest_path=str(staged.manifest),
        feature_store_dir=str(existing.root if existing is not None else store_dir),
        store_type=existing.store_type if existing is not None else "directory",
        entries_per_shard=existing.entries_per_shard if existing is not None else 1000,
        run_root=str(data_root / "data_user" / ".agritune" / "runs"),
        run_id=run.info.run_id,
        num_classes=staged.palette.num_classes,
        encoder_fingerprint=EncoderFingerprint(model=model, revision=revision, preprocessing=preprocessing),
        feature_provider=params.get("feature_provider", "hybrid"),
        feature_preload=preload,
        feature_preload_dtype=PRELOAD_DTYPE if preload != "none" else None,
        # Preloaded features can be gathered in any order; without preload AgriTune keeps manifest order.
        shuffle=preload != "none",
        encoder_base_url=base_url,
        encoder_api_key=os.environ.get("AGRITUNE_ENCODER_API_KEY"),
        augmentation=augmentation,
        decoder_name=decoder,
        device=device.type,
        batch_size=_integer(params, "batch_size", 4),
        val_fraction=_number(params, "val_fraction", 0.2),
        seed=_integer(params, "seed", 0),
        optimizer=OptimizerConfig(lr=_number(params, "lr", 3e-4), weight_decay=_number(params, "weight_decay", 0.01)),
        trainer=TrainerConfig(
            max_epochs=_integer(params, "epochs", 10),
            accumulation_steps=_integer(params, "accumulation_steps", 1),
            grad_clip_norm=clip if clip > 0 else None,
            precision=PrecisionConfig(
                mode="fp16" if _boolean(params, "amp", True) and device.type == "cuda" else "fp32"
            ),
            seed=_integer(params, "seed", 0),
        ),
        loss=SegmentationLossConfig(
            name=params.get("agritune_loss", "ce_dice"),
            ignore_index=data.IGNORE_INDEX,
            ce_weight=_number(params, f"weight_{_primary_term(params)}", 1.0),
            dice_weight=_number(params, "weight_dice", 1.0),
        ),
    )
    _apply_advanced(config, params, preset, staged.manifest)
    return config


def _manifest_rows(manifest: Path) -> int:
    with manifest.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _steps_per_run(config: TrainingRunConfig, samples: int) -> int:
    """Return the optimizer steps of the whole run, for schedules whose total length is left at 0."""
    train = max(round(samples * (1 - config.val_fraction)), 1)
    batches = math.ceil(train / config.batch_size)
    return max(math.ceil(batches / config.trainer.accumulation_steps), 1) * config.trainer.max_epochs


def _set_path(root: Any, path: str, value: Any) -> None:
    *parents, name = path.split(".")
    target = root
    for part in parents:
        target = getattr(target, part)
    if isinstance(target, dict):
        target[name] = value
    else:
        setattr(target, name, value)


def _advanced_value(setting: agritune_settings.Setting, params: dict[str, str]) -> Any:
    value = agritune_settings.from_param(setting, params[setting.key]) if setting.key in params else setting.default
    if setting.zero_is_none and value == 0:
        return None
    if setting.type in ("range", "size") and value is not None:
        return tuple(value)
    return value


def _apply_schedule(config: TrainingRunConfig, values: dict[str, Any], manifest: Path) -> None:
    """Pop the ``scheduler.*`` values; any schedule but ``constant`` becomes AgriTune's scheduler."""
    name = values.pop("scheduler.name")
    schedule = {key.split(".", 1)[1]: values.pop(key) for key in list(values) if key.startswith("scheduler.")}
    if name == "constant":
        return
    if not schedule["total_steps"]:
        schedule["total_steps"] = _steps_per_run(config, _manifest_rows(manifest))
    config.scheduler = SchedulerConfig(name=name, **schedule)


def _apply_special(config: TrainingRunConfig, values: dict[str, Any]) -> None:
    """Pop and apply the values that are not a plain assignment to their target."""
    if (precision := values.pop("trainer.precision")) != "auto":
        config.trainer.precision.mode = precision
    if (shuffle := values.pop("data.shuffle")) != "auto":
        config.shuffle = shuffle == "true"
    # AgriTune rejects a preload precision without preloading.
    preload_dtype = values.pop("data.feature_preload_dtype")
    if config.feature_preload != "none":
        config.feature_preload_dtype = preload_dtype
    config.higher_is_better = values["val_metric_name"] != "loss"
    weights = values.pop("loss.class_weights")
    if weights is not None:
        if len(weights) != config.num_classes:
            raise ValueError(f"loss.class_weights: {len(weights)} weights for {config.num_classes} classes")
        config.loss.class_weights = torch.tensor(weights, dtype=torch.float32)
    if "decoder.hidden_dims" in values:
        values["decoder.hidden_dims"] = list(values["decoder.hidden_dims"] or [])
    if "decoder.cls_fusion" in values:
        values["decoder.cls_fusion"] = CLSFusion(values["decoder.cls_fusion"])


def _apply_advanced(config: TrainingRunConfig, params: dict[str, str], preset: str, manifest: Path) -> None:
    """Write every advanced setting (the run's value, else its default) into AgriTune's config."""
    values = {
        setting.key: _advanced_value(setting, params)
        for setting in agritune_settings.SETTINGS
        if agritune_settings.applies_to(setting, preset)
    }
    _apply_schedule(config, values, manifest)
    _apply_special(config, values)
    augmenting = config.augmentation.mode is not AugmentationMode.NONE
    for key, value in values.items():
        # Image augmentation settings would otherwise turn on transforms for runs without augmentation.
        if key.startswith("augmentation.") and not augmenting:
            continue
        target = agritune_settings.BY_KEY[key].target
        # An unset decoder setting keeps that decoder's own default (they differ between decoders).
        if value is None and target.startswith("decoder_kwargs."):
            continue
        _set_path(config, target, value)


def _encode_dataset(config: TrainingRunConfig, store: DirectoryFeatureStore | ShardedFeatureStore) -> None:
    """Encode every staged sample missing from the feature store, so the cached provider finds them all.

    Samples already in the store (from an earlier run on the same images, encoder and image size) are skipped,
    so only the first run on a dataset pays for encoding.
    """
    encoder, _ = build_encoder(
        base_url=config.encoder_base_url,
        api_key=config.encoder_api_key,
        model=config.encoder_fingerprint.model,
        preprocessing=config.encoder_fingerprint.preprocessing,
    )
    log("encoding images into the feature store (already stored ones are skipped)")

    def progress(stats: PrecomputeStats) -> None:
        done = stats.computed + stats.skipped + stats.failed
        if done % data.progress_every(stats.total) == 0 or done == stats.total:
            log(f"encoding features {done}/{stats.total} ({stats.computed} new, {stats.skipped} stored)")

    stats = asyncio.run(
        build_features(
            config.manifest_path,
            store=store,
            encoder=encoder,
            encoder_fingerprint=config.encoder_fingerprint,
            on_progress=progress,
        )
    )
    if stats.failed:
        raise RuntimeError(
            f"{stats.failed} of {stats.total} images could not be encoded "
            f"(e.g. {', '.join(stats.failed_sample_ids[:3])}); check the encoder URL, key and logs"
        )


def run(run_id: str, tracking_uri: str, data_root: Path) -> int:
    """Train one existing MagniData MLflow run through AgriTune."""
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    queued = client.get_run(run_id)
    params = dict(queued.data.params)
    dataset_source = queued.data.tags.get("magnidata.dataset_source", "")
    native = params.get("input_resolution", "image_size") == "native"
    # Reading an existing store needs its exact image files and the cached provider (nothing is encoded).
    existing = existing_store_from_env() if native and params.get("feature_provider") == "cached" else None
    staged = _prepare_manifest(
        run_id,
        dataset_source,
        data_root,
        _integer(params, "image_size", 256),
        workers=default_num_workers(),
        native=native,
        existing=existing,
    )
    config = _config(queued, data_root, staged)
    store: DirectoryFeatureStore | ShardedFeatureStore
    if config.store_type == "sharded":
        store = ShardedFeatureStore(config.feature_store_dir, entries_per_shard=config.entries_per_shard)
    else:
        store = DirectoryFeatureStore(config.feature_store_dir)
    if config.feature_provider == "cached" and staged.existing is None:
        _encode_dataset(config, store)
    palette_colors = ((0, 0, 0), *staged.palette.colors)
    result = run_magnidata_training(
        config,
        store=store,
        run_id=run_id,
        tracking_uri=tracking_uri,
        options=MagniDataOptions(
            score_every=_integer(params, "score_every", 1),
            top_k_predictions=_integer(params, "top_k_predictions", 50),
            palette=palette_colors,
            image_names=staged.image_names,
        ),
    )
    return 0 if result is not None else 1


def main() -> None:
    """Run the command-line adapter."""
    parser = argparse.ArgumentParser(description="Train a queued MagniData run with AgriTune.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATA_ROOT", ".")))
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    args = parser.parse_args()
    sys.exit(run(args.run_id, args.tracking_uri, args.data_root))


if __name__ == "__main__":
    main()
