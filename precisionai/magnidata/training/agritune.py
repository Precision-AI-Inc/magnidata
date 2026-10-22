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
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from precisionai.magnidata.training import data
from precisionai.magnidata.training.train import default_num_workers, describe_device, log, resolve_device

try:
    import mlflow

    from precisionai.agritune.augmentations.image.pipeline import AugmentationMode, GeometricConfig
    from precisionai.agritune.features.keys import EncoderFingerprint
    from precisionai.agritune.features.precompute import PrecomputeStats
    from precisionai.agritune.features.store import DirectoryFeatureStore
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


def _stage_one(job: tuple[Path, Path, Path, Path, int, tuple[tuple[int, int, int], ...]]) -> None:
    """Write one sample's image resized to the run's size and its mask decoded to class indices."""
    source_image, source_mask, image_path, mask_path, image_size, colors = job
    with Image.open(source_image) as image:
        image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR).save(image_path)
    with Image.open(source_mask) as mask:
        rgb = np.asarray(mask.convert("RGB").resize((image_size, image_size), Image.Resampling.NEAREST))
    Image.fromarray(data.decode_mask(rgb, data.Palette(colors)).astype(np.uint8), mode="L").save(mask_path)


def _prepare_manifest(
    run_id: str, dataset_source: str, data_root: Path, image_size: int, workers: int = 0
) -> tuple[Path, data.Palette]:
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
        (
            sample.image_path,
            sample.mask_path,
            images / f"{Path(sample.image_name).stem}.png",
            masks / f"{Path(sample.image_name).stem}.png",
            image_size,
            palette.colors,
        )
        for sample in samples
    ]
    log(f"staging {total} images at {image_size}x{image_size} with {max(workers, 1)} process(es)")
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
    rows = [
        (sample.image_name, image_path, mask_path, sample.csv_split or "")
        for sample, (_, _, image_path, mask_path, _, _) in zip(samples, jobs, strict=True)
    ]
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "image_path", "mask_path", "source_split"])
        writer.writerows(rows)
    return manifest, palette


def _config(run: Any, data_root: Path, manifest: Path, palette: data.Palette) -> TrainingRunConfig:
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
    return TrainingRunConfig(
        manifest_path=str(manifest),
        feature_store_dir=str(data_root / "data_user" / ".agritune" / "features" / model.replace("/", "_")),
        run_root=str(data_root / "data_user" / ".agritune" / "runs"),
        run_id=run.info.run_id,
        num_classes=palette.num_classes,
        encoder_fingerprint=EncoderFingerprint(model=model, revision=revision, preprocessing="magnidata-square-v1"),
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
        optimizer=OptimizerConfig(lr=_number(params, "lr", 3e-4)),
        trainer=TrainerConfig(
            max_epochs=_integer(params, "epochs", 10),
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


def _encode_dataset(config: TrainingRunConfig, store: DirectoryFeatureStore) -> None:
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
    dataset_source = queued.data.tags.get("magnidata.dataset_source", "")
    image_size = _integer(dict(queued.data.params), "image_size", 256)
    manifest, palette = _prepare_manifest(run_id, dataset_source, data_root, image_size, default_num_workers())
    config = _config(queued, data_root, manifest, palette)
    store = DirectoryFeatureStore(config.feature_store_dir)
    if config.feature_provider == "cached":
        _encode_dataset(config, store)
    palette_colors = ((0, 0, 0), *palette.colors)
    result = run_magnidata_training(
        config,
        store=store,
        run_id=run_id,
        tracking_uri=tracking_uri,
        options=MagniDataOptions(
            score_every=_integer(dict(queued.data.params), "score_every", 1),
            top_k_predictions=_integer(dict(queued.data.params), "top_k_predictions", 50),
            palette=palette_colors,
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
