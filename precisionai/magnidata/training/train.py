# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Reference semantic-segmentation trainer that logs everything to an MLflow run.

Run on an existing run (created by the API)::

    python -m precisionai.magnidata.training.train --run-id <id> --data-root <path>

or create the run from the command line::

    python -m precisionai.magnidata.training.train --dataset data_user/my.csv \
        --preset tiny_unet --experiment demo --param epochs=5

Per-sample losses come from a separate eval-mode, no-augmentation scoring pass, because
losses seen while training are noisy (augmentation, dropout, weights changing mid-epoch).
"""

import argparse
import contextlib
import heapq
import os
import random
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from precisionai.magnidata.training import contract, data, losses
from precisionai.magnidata.training.mlflow_sink import MlflowSink
from precisionai.magnidata.training.models import build_model
from precisionai.magnidata.training.recorder import LossRecorder

try:
    import mlflow
    import torch
    from mlflow.entities import Metric, Param
    from torch import nn
    from torch.utils.data import DataLoader
except ImportError:
    raise ImportError("torch and mlflow are required for training: pip install -e '.[training]'") from None

DEFAULT_TRACKING_URI = "http://localhost:5052"
DEFAULT_TERMS = ("dice", "cce")
CANCEL_POLL_S = 5.0
LR_LOG_EVERY = 10
EXIT_FAILED = 1


def default_num_workers() -> int:
    """Return the data-loader process count: one per spare core, at most 8."""
    return max(min(8, (os.cpu_count() or 2) - 1), 0)


@dataclass
class TrainConfig:
    """Hyper-parameters, resolved from run params with defaults."""

    dataset_source: str
    preset: str = "tiny_unet"
    loss_terms: list[str] = field(default_factory=lambda: list(DEFAULT_TERMS))
    loss_weights: dict[str, float] = field(default_factory=dict)
    epochs: int = 10
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    image_size: int = 256
    val_fraction: float = 0.2
    seed: int = 0
    score_every: int = 1
    top_k_predictions: int = 50
    pretrained: bool = True
    checkpoint_every_score: bool = False
    num_workers: int = field(default_factory=default_num_workers)
    amp: bool = True
    split_mode: str = "random"

    def as_params(self) -> dict[str, str]:
        """Return every setting as MLflow param strings."""
        return {
            "preset": self.preset,
            "loss_terms": ",".join(self.loss_terms),
            "epochs": str(self.epochs),
            "batch_size": str(self.batch_size),
            "lr": str(self.lr),
            "weight_decay": str(self.weight_decay),
            "image_size": str(self.image_size),
            "val_fraction": str(self.val_fraction),
            "seed": str(self.seed),
            "score_every": str(self.score_every),
            "top_k_predictions": str(self.top_k_predictions),
            "split_mode": self.split_mode,
            "num_workers": str(self.num_workers),
            "amp": str(self.amp).lower(),
            "pretrained": str(self.pretrained).lower(),
            "checkpoint_every_score": str(self.checkpoint_every_score).lower(),
            **{f"weight_{t}": str(w) for t, w in self.loss_weights.items()},
        }


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def config_from_run(params: Mapping[str, str], tags: Mapping[str, str]) -> TrainConfig:
    """Build the config from a run's params and ``magnidata.*`` tags."""
    source = tags.get(contract.TAG_DATASET_SOURCE)
    if not source:
        raise ValueError(f"run has no {contract.TAG_DATASET_SOURCE} tag")
    terms_text = tags.get(contract.TAG_LOSS_TERMS) or params.get("loss_terms") or ",".join(DEFAULT_TERMS)
    terms = [t.strip() for t in terms_text.split(",") if t.strip()]
    cfg = TrainConfig(dataset_source=source, preset=tags.get(contract.TAG_PRESET) or params.get("preset", "tiny_unet"))
    cfg.loss_terms = terms
    cfg.loss_weights = {t: float(params[f"weight_{t}"]) for t in terms if f"weight_{t}" in params}
    for name in ("epochs", "batch_size", "image_size", "seed", "score_every", "top_k_predictions", "num_workers"):
        if name in params:
            setattr(cfg, name, int(params[name]))
    for name in ("lr", "weight_decay", "val_fraction"):
        if name in params:
            setattr(cfg, name, float(params[name]))
    for name in ("pretrained", "checkpoint_every_score", "amp"):
        if name in params:
            setattr(cfg, name, _as_bool(params[name]))
    cfg.split_mode = params.get("split_mode", cfg.split_mode)
    if cfg.split_mode not in {"random", "column"}:
        raise ValueError(f"unknown split_mode {cfg.split_mode!r}: choose random or column")
    if cfg.preset not in contract.MODEL_PRESETS:
        raise ValueError(f"unknown preset {cfg.preset!r}: choose from {', '.join(contract.MODEL_PRESETS)}")
    return cfg


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    """Return the torch device for ``auto``, ``cpu`` or ``cuda``."""
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but not available (torch {torch.__version__})")
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        else:
            name = "cpu"
            log(f"warning: CUDA not available (torch {torch.__version__}), training on CPU")
    return torch.device(name)


def log(message: str) -> None:
    """Print one progress line right away, so the worker's log file is always current."""
    print(message, flush=True)


def describe_device(device: torch.device) -> str:
    """Return ``cuda:0 (GPU name)`` or ``cpu``."""
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        return f"cuda:{index} ({torch.cuda.get_device_name(index)})"
    return str(device)


def _format_duration(seconds: float) -> str:
    seconds = round(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    return f"{minutes // 60}h{minutes % 60:02d}m"


@dataclass
class AmpMode:
    """Mixed precision on CUDA: bfloat16 where the GPU supports it, otherwise float16 with loss scaling."""

    device_type: str
    dtype: torch.dtype
    scaler: Any = None

    def context(self) -> contextlib.AbstractContextManager:
        """Return the autocast context for forward passes."""
        return torch.autocast(self.device_type, dtype=self.dtype)


def make_amp(device: torch.device, enabled: bool) -> AmpMode | None:
    """Return the AMP settings for ``device``, or None when AMP is off or the device is not CUDA."""
    if not enabled or device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return AmpMode("cuda", torch.bfloat16)
    return AmpMode("cuda", torch.float16, torch.amp.grad_scaler.GradScaler("cuda"))


def _autocast(amp: AmpMode | None) -> contextlib.AbstractContextManager:
    return amp.context() if amp else contextlib.nullcontext()


class MetricBuffer:
    """Collects metrics and sends them to MLflow in one request."""

    def __init__(self, client: Any, run_id: str) -> None:
        self.client = client
        self.run_id = run_id
        self._items: list[Metric] = []

    def add(self, key: str, value: float, step: int) -> None:
        """Queue one metric point."""
        self._items.append(Metric(key, float(value), int(time.time() * 1000), step))

    def flush(self) -> None:
        """Send everything queued."""
        if self._items:
            self.client.log_batch(self.run_id, metrics=self._items)
            self._items = []


@dataclass
class EpochMeans:
    """Mean per-sample loss terms over a pass (``total`` is the weighted sum)."""

    values: dict[str, float]
    cancelled: bool = False


def _accumulate(sums: dict[str, float], per_sample: Mapping[str, torch.Tensor]) -> int:
    count = 0
    for key, tensor in per_sample.items():
        sums[key] = sums.get(key, 0.0) + float(tensor.detach().sum())
        count = len(tensor)
    return count


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: losses.SegmentationCriterion,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    cancel: Callable[[], bool],
    on_step: Callable[[float], None],
    label: str = "",
    amp: AmpMode | None = None,
) -> EpochMeans:
    """Run one training epoch; stop early (``cancelled=True``) when ``cancel()`` is true.

    A progress line is printed about ten times per epoch, prefixed with ``label``.
    """
    model.train()
    sums: dict[str, float] = {}
    seen = 0
    n_batches = len(loader)
    every = max(n_batches // 10, 1)
    started = time.monotonic()
    for index, batch in enumerate(loader, start=1):
        if cancel():
            return EpochMeans({}, cancelled=True)
        with _autocast(amp):
            logits = model(batch["pixel_values"].to(device, non_blocking=True))
        out = criterion(logits.float(), batch["labels"].to(device, non_blocking=True))
        optimizer.zero_grad(set_to_none=True)
        if amp and amp.scaler:
            amp.scaler.scale(out.total).backward()
            amp.scaler.step(optimizer)
            amp.scaler.update()
        else:
            out.total.backward()
            optimizer.step()
        scheduler.step()
        seen += _accumulate(sums, out.per_sample)
        on_step(optimizer.param_groups[0]["lr"])
        if index % every == 0 or index == n_batches:
            rate = index / max(time.monotonic() - started, 1e-9)
            log(
                f"{label} train batch {index}/{n_batches} "
                f"loss={sums.get('total', 0.0) / max(seen, 1):.3f} {rate:.1f} it/s"
            )
    return EpochMeans({k: v / max(seen, 1) for k, v in sums.items()})


def iou_from_confusion(confusion: torch.Tensor) -> tuple[float, list[float | None]]:
    """Return (mean IoU, per-class IoU) from a ``[C, C]`` confusion matrix.

    Classes with no pixels in either target or prediction have IoU ``None`` and are
    left out of the mean.
    """
    conf = confusion.double()
    tp = conf.diag()
    union = conf.sum(0) + conf.sum(1) - tp
    per_class = [float(t / u) if u > 0 else None for t, u in zip(tp, union, strict=True)]
    present = [v for v in per_class if v is not None]
    return (sum(present) / len(present) if present else 0.0), per_class


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: losses.SegmentationCriterion,
    device: torch.device,
    num_classes: int,
    amp: AmpMode | None = None,
) -> tuple[dict[str, float], torch.Tensor]:
    """Return mean per-sample losses and the confusion matrix over a loader (eval mode)."""
    model.eval()
    sums: dict[str, float] = {}
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    seen = 0
    for batch in loader:
        labels = batch["labels"].to(device, non_blocking=True)
        with _autocast(amp):
            logits = model(batch["pixel_values"].to(device, non_blocking=True))
        logits = logits.float()
        seen += _accumulate(sums, criterion(logits, labels).per_sample)
        confusion += losses.confusion_counts(logits, labels, num_classes).cpu()
    return {k: v / max(seen, 1) for k, v in sums.items()}, confusion


@dataclass(order=True)
class _Hard:
    loss: float
    name: str = field(compare=False)
    mask: np.ndarray = field(compare=False)


@torch.no_grad()
def score_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: losses.SegmentationCriterion,
    device: torch.device,
    recorder: LossRecorder,
    splits: Mapping[str, str],
    epoch: int,
    top_k: int,
    amp: AmpMode | None = None,
) -> list[_Hard] | None:
    """Score every sample in eval mode, record per-sample losses, and return the hardest K.

    Returns None if the run was cancelled part-way (nothing is logged for the epoch).
    """
    model.eval()
    recorder.begin_epoch(epoch)
    heap: list[_Hard] = []
    started = time.monotonic()
    n_samples = 0
    log(f"epoch {epoch} scoring {len(loader)} batches")
    for batch in loader:
        if recorder.cancel_requested():
            return None
        labels = batch["labels"].to(device, non_blocking=True)
        with _autocast(amp):
            logits = model(batch["pixel_values"].to(device, non_blocking=True))
        logits = logits.float()
        per_sample = criterion(logits, labels).per_sample
        names = batch["image_name"]
        n_samples += len(names)
        recorder.add_batch(
            names,
            [splits[n] for n in names],
            {k: v.cpu() for k, v in per_sample.items()},
            {
                "iou": losses.iou_per_sample(logits, labels, logits.shape[1]).cpu(),
                "confidence": losses.mean_confidence_per_sample(logits, labels, criterion.ignore_index).cpu(),
            },
        )
        preds = logits.argmax(1).to(torch.uint8).cpu().numpy()
        for i, (name, total) in enumerate(zip(names, per_sample["total"].cpu().tolist(), strict=True)):
            item = _Hard(total, name, preds[i])
            if len(heap) < top_k:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    recorder.end_epoch()
    log(f"epoch {epoch} scored {n_samples} samples in {_format_duration(time.monotonic() - started)}")
    return sorted(heap, reverse=True)


def log_predictions(
    sink: MlflowSink, palette: data.Palette, hardest: Sequence[_Hard], epoch: int, scratch: Path
) -> None:
    """Log the hardest samples' predicted masks as colour PNGs."""
    lookup = np.array([(0, 0, 0), *palette.colors], dtype=np.uint8)
    for item in hardest:
        artifact = contract.prediction_path(epoch, item.name)
        local = scratch / "predictions" / f"{epoch:04d}" / Path(artifact).name
        local.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(lookup[np.minimum(item.mask, len(lookup) - 1)]).save(local)
        sink.log_file(local, str(Path(artifact).parent))


def save_checkpoint(
    sink: MlflowSink, model: nn.Module, cfg: TrainConfig, palette: data.Palette, epoch: int, scratch: Path
) -> None:
    """Log ``checkpoints/last.pt`` with the weights and what is needed to rebuild the model."""
    path = scratch / Path(contract.CHECKPOINT_ARTIFACT).name
    state = {
        "model": model.state_dict(),
        "preset": cfg.preset,
        "num_classes": palette.num_classes,
        "palette": palette.as_mapping(),
        "image_size": cfg.image_size,
        "epoch": epoch,
    }
    torch.save(state, path)
    sink.log_file(path, str(Path(contract.CHECKPOINT_ARTIFACT).parent))


@dataclass
class _Setup:
    """Everything built once before the epoch loop."""

    cfg: TrainConfig
    palette: data.Palette
    splits: dict[str, str]
    train_loader: DataLoader
    val_loader: DataLoader
    score_loader: DataLoader
    n_train: int
    n_val: int


def prepare_data(
    cfg: TrainConfig, data_root: Path, recorder: LossRecorder, device: torch.device | None = None
) -> _Setup:
    """Resolve samples, cache them resized, build palette and split, log them, and make the data loaders."""
    csv_path = data_root / cfg.dataset_source
    if not csv_path.is_file():
        raise FileNotFoundError(f"dataset CSV not found: {cfg.dataset_source}")
    samples, report = data.resolve_samples(csv_path, data_root)
    if not samples:
        raise ValueError(
            f"no usable samples in {cfg.dataset_source}: {len(report.missing_mask)} without mask, "
            f"{len(report.missing_image)} without image"
        )
    size = cfg.image_size
    cache = data.ResizedCache(data_root / "data_user" / data.CACHE_DIRNAME, (size, size))
    log(f"caching {len(samples)} images at {size}x{size} with {max(cfg.num_workers, 1)} process(es)")
    started = time.monotonic()
    data.warm_cache(cache, samples, cfg.num_workers, log)
    log(f"cache ready in {_format_duration(time.monotonic() - started)}")
    palette = data.build_palette_cached(cache, samples)
    column = {s.image_name: s.csv_split for s in samples if s.csv_split} if cfg.split_mode == "column" else {}
    if cfg.split_mode == "column" and not column:
        raise ValueError("split_mode=column but the dataset CSV has no split column")
    splits = data.make_split([s.image_name for s in samples], cfg.val_fraction, cfg.seed, column or None)
    recorder.write_split(splits)
    recorder.write_palette(palette.as_mapping())

    train_ds = data.SegmentationDataset(samples, palette, splits, "train", size, augment=True, cache=cache)
    val_ds = data.SegmentationDataset(samples, palette, splits, "val", size, cache=cache)
    score_ds = data.SegmentationDataset(samples, palette, splits, None, size, cache=cache)
    pin = device is not None and device.type == "cuda"
    if len(train_ds) == 0:
        raise ValueError("the train split is empty")

    def loader(ds: data.SegmentationDataset, shuffle: bool) -> DataLoader:
        gen = torch.Generator().manual_seed(cfg.seed)
        workers = cfg.num_workers if shuffle else (max(cfg.num_workers // 2, 1) if cfg.num_workers else 0)
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            collate_fn=data.collate_fn,
            num_workers=workers,
            pin_memory=pin,
            persistent_workers=workers > 0,
            generator=gen if shuffle else None,
        )

    return _Setup(
        cfg,
        palette,
        splits,
        loader(train_ds, True),
        loader(val_ds, False),
        loader(score_ds, False),
        len(train_ds),
        len(val_ds),
    )


def log_epoch_metrics(
    buffer: MetricBuffer,
    palette: data.Palette,
    epoch: int,
    train: Mapping[str, float],
    val: Mapping[str, float],
    confusion: torch.Tensor | None,
) -> None:
    """Queue the contract's per-epoch aggregate metrics."""
    for term, value in train.items():
        buffer.add(contract.train_loss_key("loss" if term == "total" else term), value, epoch)
    for term, value in val.items():
        buffer.add(contract.val_loss_key("loss" if term == "total" else term), value, epoch)
    if confusion is not None:
        miou, per_class = iou_from_confusion(confusion)
        buffer.add(contract.MIOU_METRIC, miou, epoch)
        names = palette.as_mapping()
        for index, value in enumerate(per_class):
            if value is not None:
                buffer.add(contract.val_iou_key(names[index]["name"]), value, epoch)


def run_epochs(
    client: Any,
    run_id: str,
    sink: MlflowSink,
    setup: _Setup,
    recorder: LossRecorder,
    scratch: Path,
    device: torch.device,
) -> str:
    """Train, validate and score for every epoch; return the final status (FINISHED or KILLED)."""
    cfg = setup.cfg
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    amp = make_amp(device, cfg.amp)
    log(f"loader workers={cfg.num_workers} amp={amp.dtype if amp else 'off'}")
    model = build_model(cfg.preset, setup.palette.num_classes, cfg.pretrained).to(device)
    criterion = losses.SegmentationCriterion(cfg.loss_terms, cfg.loss_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(cfg.epochs * len(setup.train_loader), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    buffer = MetricBuffer(client, run_id)
    step = [0]

    def on_step(lr: float) -> None:
        step[0] += 1
        if step[0] % LR_LOG_EVERY == 1:
            buffer.add(contract.LEARNING_RATE_METRIC, lr, step[0])

    best = float("inf")
    run_started = time.monotonic()
    for epoch in range(1, cfg.epochs + 1):
        epoch_started = time.monotonic()
        train = train_epoch(
            model,
            setup.train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            sink.cancel_requested,
            on_step,
            label=f"epoch {epoch}/{cfg.epochs}",
            amp=amp,
        )
        if train.cancelled:
            buffer.flush()
            return "KILLED"
        val, confusion = (
            evaluate(model, setup.val_loader, criterion, device, setup.palette.num_classes, amp)
            if setup.n_val
            else ({}, None)
        )
        log_epoch_metrics(buffer, setup.palette, epoch, train.values, val, confusion)
        monitored = val.get("total", train.values["total"])
        if monitored < best:
            best = monitored
            buffer.add("best_val_loss", best, epoch)
            buffer.add("best_epoch", epoch, epoch)
        seconds = time.monotonic() - epoch_started
        buffer.add(contract.EPOCH_SECONDS_METRIC, seconds, epoch)
        buffer.add(contract.SAMPLES_PER_SEC_METRIC, setup.n_train / max(seconds, 1e-9), epoch)
        buffer.flush()
        eta = (time.monotonic() - run_started) / epoch * (cfg.epochs - epoch)
        sink.set_tag(contract.TAG_ETA_SECONDS, str(int(eta)))
        log(
            f"epoch {epoch}/{cfg.epochs} done train={train.values['total']:.3f} "
            f"val={val.get('total', float('nan')):.3f} {seconds:.1f}s ETA {_format_duration(eta)}"
        )
        if epoch == 1 or epoch % cfg.score_every == 0 or epoch == cfg.epochs:
            hardest = score_epoch(
                model, setup.score_loader, criterion, device, recorder, setup.splits, epoch, cfg.top_k_predictions, amp
            )
            if hardest is None:
                return "KILLED"
            log_predictions(sink, setup.palette, hardest, epoch, scratch)
            if cfg.checkpoint_every_score:
                save_checkpoint(sink, model, cfg, setup.palette, epoch, scratch)
    save_checkpoint(sink, model, cfg, setup.palette, cfg.epochs, scratch)
    return "FINISHED"


def log_missing_params(client: Any, run: Any, cfg: TrainConfig, extra: Mapping[str, str]) -> None:
    """Log the resolved settings the run does not already carry (MLflow params are write-once)."""
    wanted = {**cfg.as_params(), **extra}
    missing = [Param(k, v) for k, v in wanted.items() if k not in run.data.params]
    if missing:
        client.log_batch(run.info.run_id, params=missing)


def run(run_id: str, tracking_uri: str, data_root: Path, device_name: str = "auto") -> int:
    """Train the MLflow run ``run_id``; return the process exit code.

    The run ends ``FINISHED``, ``KILLED`` (cancel tag; exit code 0) or ``FAILED`` (the first
    line of the error goes to the ``magnidata.error`` tag and the traceback to stdout).
    """
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    existing = client.get_run(run_id)
    client.update_run(run_id, status="RUNNING")
    sink = MlflowSink(run_id, tracking_uri, cancel_poll_s=CANCEL_POLL_S)
    try:
        cfg = config_from_run(existing.data.params, existing.data.tags)
        seed_everything(int(cfg.seed))
        device = resolve_device(device_name)
        description = describe_device(device)
        log(f"device={description} torch={torch.__version__}")
        sink.set_tag(contract.TAG_DEVICE, description)
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp)
            if sink.cancel_requested():
                status = "KILLED"
            else:
                recorder = LossRecorder(cfg.loss_terms, sink, scratch / "per_sample")
                setup = prepare_data(cfg, data_root, recorder, device)
                log_missing_params(
                    client, existing, cfg, {"num_classes": str(setup.palette.num_classes), "device": str(device)}
                )
                status = run_epochs(client, run_id, sink, setup, recorder, scratch, device)
    except Exception as err:
        traceback.print_exc(file=sys.stdout)
        client.set_tag(run_id, "magnidata.error", (str(err).splitlines() or [type(err).__name__])[0][:250])
        client.set_terminated(run_id, "FAILED")
        return EXIT_FAILED
    client.set_terminated(run_id, status)
    return 0


def create_run(client: Any, args: argparse.Namespace) -> str:
    """Create an ``external``-origin run for command-line use and return its id."""
    experiment = client.get_experiment_by_name(args.experiment)
    experiment_id = (
        experiment.experiment_id
        if experiment
        else client.create_experiment(args.experiment, artifact_location=args.artifact_location)
    )
    params = dict(item.split("=", 1) for item in args.param)
    tags = {
        contract.TAG_DATASET_SOURCE: args.dataset,
        contract.TAG_DATASET_NAME: Path(args.dataset).stem,
        contract.TAG_ORIGIN: contract.ORIGIN_EXTERNAL,
        contract.TAG_PRESET: args.preset,
    }
    created = client.create_run(experiment_id, tags=tags)
    for key, value in params.items():
        client.log_param(created.info.run_id, key, value)
    return created.info.run_id


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; return the process exit code."""
    parser = argparse.ArgumentParser(description="Train a segmentation model and log it to MLflow.")
    parser.add_argument("--run-id", help="existing MLflow run to train (otherwise one is created)")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATA_ROOT", ".")))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    parser.add_argument("--dataset", help="dataset CSV relative to --data-root (when creating a run)")
    parser.add_argument("--preset", default="tiny_unet", choices=contract.MODEL_PRESETS)
    parser.add_argument("--experiment", default="magnidata")
    parser.add_argument("--param", action="append", default=[], metavar="K=V", help="extra run param (repeatable)")
    parser.add_argument("--artifact-location", help="artifact root for a newly created experiment")
    args = parser.parse_args(argv)
    if not args.run_id and not args.dataset:
        parser.error("give --run-id, or --dataset to create a run")
    if any("=" not in p for p in args.param):
        parser.error("--param expects K=V")

    run_id = args.run_id or create_run(mlflow.MlflowClient(tracking_uri=args.tracking_uri), args)
    return run(run_id, args.tracking_uri, args.data_root, args.device)


if __name__ == "__main__":
    sys.exit(main())
