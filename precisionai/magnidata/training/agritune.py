# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Adapt a queued MagniData dataset/run into AgriTune's training service."""

import argparse
import asyncio
import csv
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from PIL import Image

from precisionai.magnidata.training import contract, data
from precisionai.magnidata.training.train import default_num_workers, describe_device, log, resolve_device

try:
    import mlflow

    from precisionai.agritune.augmentations.image.pipeline import AugmentationMode, GeometricConfig
    from precisionai.agritune.features.keys import EncoderFingerprint
    from precisionai.agritune.features.precompute import PrecomputeStats
    from precisionai.agritune.features.store import DirectoryFeatureStore, ShardedFeatureStore
    from precisionai.agritune.integrations.magnidata import MagniDataOptions, run_magnidata_training
    from precisionai.agritune.optimization.optimizers import OptimizerConfig
    from precisionai.agritune.services.encoder_selection import build_encoder
    from precisionai.agritune.services.feature_service import build_features
    from precisionai.agritune.services.training_service import AugmentationSelection, TrainingRunConfig
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
    """One sample to stage: its sources, where to write, the target size (None keeps the original) and the palette."""

    source_image: Path
    source_mask: Path
    image_path: Path | None
    mask_path: Path
    size: int | None
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


def _stage_one(job: _StageJob) -> tuple[int, int]:
    """Write one sample's mask decoded to class indices (and, when square, its resized image); return the image size."""
    with Image.open(job.source_image) as image:
        size = image.size
        if job.image_path is not None and job.size is not None:
            image.convert("RGB").resize((job.size, job.size), Image.Resampling.BILINEAR).save(job.image_path)
    with Image.open(job.source_mask) as mask:
        rgb = mask.convert("RGB")
        if job.size is not None:
            rgb = rgb.resize((job.size, job.size), Image.Resampling.NEAREST)
        elif rgb.size != size:
            raise ValueError(f"mask {job.source_mask} is {rgb.size[0]}x{rgb.size[1]}, its image {size[0]}x{size[1]}")
    Image.fromarray(data.decode_mask(np.asarray(rgb), data.Palette(job.colors)).astype(np.uint8), mode="L").save(
        job.mask_path
    )
    return size


def _stage(jobs: list[_StageJob], workers: int) -> list[tuple[int, int]]:
    total = len(jobs)
    every = data.progress_every(total)
    sizes = []
    if workers > 1 and total > 1:
        # spawn, not fork: this process already runs MLflow client threads
        with ProcessPoolExecutor(
            max_workers=min(workers, total), mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for done, size in enumerate(pool.map(_stage_one, jobs, chunksize=16), start=1):
                sizes.append(size)
                if done % every == 0 or done == total:
                    log(f"staging images {done}/{total}")
    else:
        for done, job in enumerate(jobs, start=1):
            sizes.append(_stage_one(job))
            if done % every == 0 or done == total:
                log(f"staging images {done}/{total}")
    return sizes


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

    ``native`` keeps every image file as it is, so the encoder sees the original resolution, and decodes masks at
    their original size; all images must share one size. ``existing`` (native only) reuses a feature store built
    from another manifest when every image matches one of its rows by file name.
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
    jobs = [
        _StageJob(
            sample.image_path,
            sample.mask_path,
            None if native else images / f"{Path(sample.image_name).stem}.png",
            masks / f"{Path(sample.image_name).stem}.png",
            None if native else image_size,
            palette.colors,
        )
        for sample in samples
    ]
    resolution = "their original size" if native else f"{image_size}x{image_size}"
    log(f"staging {total} {'masks' if native else 'images'} at {resolution} with {max(workers, 1)} process(es)")
    sizes = set(_stage(jobs, workers))
    if native and len(sizes) > 1:
        raise ValueError(
            f"input_resolution native needs every image the same size; found {len(sizes)} sizes "
            f"(e.g. {', '.join(f'{w}x{h}' for w, h in sorted(sizes)[:3])}); use input_resolution image_size"
        )
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
    if matched is None:
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
    if augmentation_mode is not AugmentationMode.NONE:
        augmentation.geometric = GeometricConfig(horizontal_flip_probability=0.5)
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
    return TrainingRunConfig(
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
