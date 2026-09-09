# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Download COCO128-seg and build an magnidata dashboard dataset.

This script is intentionally standard-library-heavy so a user can run it from a
fresh checkout after installing the normal magnidata tool dependencies:

    python3 -m precisionai.magnidata.scripts.prepare_coco128_dashboard --device cpu

It downloads Ultralytics COCO128-seg (the same 128 images as COCO128, but with real
per-instance segmentation polygons instead of just bounding boxes), stages images
under image_sets/, converts the YOLO-segment labels into the per-image COCO polygon
JSON files expected by tools/features.py, generates the dashboard CSV (real
instance/coverage/class metrics from the actual polygon shapes, not a box
approximation), generates DINOv2 embeddings, and registers the dataset in
precisionai/magnidata/confi.yaml.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128-seg.zip"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Ultralytics COCO128 uses the standard 80-class COCO taxonomy in YOLO labels.
COCO80 = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


@dataclass(frozen=True)
class YoloInstance:
    """One YOLO-segment instance parsed from a label file.

    Attributes
    ----------
    class_id : int
        Zero-based COCO80 class index.
    points : tuple[tuple[float, float], ...]
        Normalized (x, y) polygon vertices, len >= 3.
    """

    class_id: int
    points: tuple[tuple[float, float], ...]  # normalized (x, y) polygon vertices, len >= 3


def repo_root() -> Path:
    """Return the repository root directory.

    Returns
    -------
    Path
        Absolute path four levels above this file (the git repository root).
    """
    return Path(__file__).resolve().parents[3]


def log(message: str) -> None:
    """Print a status message prefixed with this script's log tag.

    Parameters
    ----------
    message : str
        Text to print.
    """
    print(f"[coco128] {message}", flush=True)


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 checksum of a file's contents.

    Parameters
    ----------
    path : Path
        File to hash.

    Returns
    -------
    str
        Hex-encoded digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, force: bool = False) -> None:
    """Download a URL to a local file, reusing a cached copy when possible.

    Parameters
    ----------
    url : str
        Source URL; must use the ``http`` or ``https`` scheme.
    dest : Path
        Destination file path.
    force : bool
        Redownload even if ``dest`` already exists.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        log(f"Using cached download: {dest}")
        return

    log(f"Downloading {url}")
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        raise ValueError(f"unsupported URL scheme for download: {url!r}")
    with urllib.request.urlopen(url) as response, dest.open("wb") as f:
        total = int(response.headers.get("Content-Length") or 0)
        seen = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            seen += len(chunk)
            if total:
                pct = 100.0 * seen / total
                print(f"\r[coco128] download {pct:5.1f}% ({seen / 1_000_000:.1f} MB)", end="", file=sys.stderr)
        if total:
            print(file=sys.stderr)
    log(f"Downloaded {dest} ({dest.stat().st_size / 1_000_000:.1f} MB)")


def extract(zip_path: Path, extract_dir: Path, force: bool = False) -> Path:
    """Extract the COCO128 zip archive, reusing a cached extraction when possible.

    Parameters
    ----------
    zip_path : Path
        Path to the downloaded zip archive.
    extract_dir : Path
        Directory to extract into.
    force : bool
        Delete and re-extract even if a cached extraction already exists.

    Returns
    -------
    Path
        Path to the ``coco128`` directory containing ``images/train2017`` and
        ``labels/train2017``.
    """
    coco_root = extract_dir / "coco128"
    if coco_root.is_dir() and not force:
        log(f"Using cached extraction: {coco_root}")
        return coco_root
    if force and extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    log(f"Extracting {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    if not coco_root.is_dir():
        matches = [
            p
            for p in extract_dir.rglob("*")
            if (p / "images" / "train2017").is_dir() and (p / "labels" / "train2017").is_dir()
        ]
        if not matches:
            raise FileNotFoundError("could not find extracted coco128/images/train2017 and labels/train2017")
        coco_root = matches[0]
    return coco_root


def read_yolo_seg_labels(path: Path) -> list[YoloInstance]:
    """Parse a YOLO-segment label file into a list of instances.

    Each line has the form ``class_id x1 y1 x2 y2 ... xn yn`` with normalized
    0..1 polygon vertices and n >= 3 per instance (Ultralytics' coco128-seg
    format). Malformed or non-numeric rows are skipped with a log message.
    """
    if not path.is_file():
        return []
    instances: list[YoloInstance] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) < 7 or len(parts) % 2 == 0:  # class_id + >=3 (x,y) pairs = odd, >=7
            log(f"Skipping malformed segmentation row {path}:{line_no}")
            continue
        try:
            class_id = int(float(parts[0]))
            coords = [clamp(float(v), 0.0, 1.0) for v in parts[1:]]
        except ValueError:
            log(f"Skipping non-numeric segmentation row {path}:{line_no}")
            continue
        if class_id < 0 or class_id >= len(COCO80):
            continue
        points = tuple(zip(coords[0::2], coords[1::2], strict=True))
        instances.append(YoloInstance(class_id, points))
    return instances


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a value to the inclusive range [lo, hi].

    Parameters
    ----------
    value : float
        Value to clamp.
    lo : float
        Lower bound.
    hi : float
        Upper bound.

    Returns
    -------
    float
        The clamped value.
    """
    return max(lo, min(hi, value))


def polygon_area(points: Iterable[tuple[float, float]]) -> float:
    """Compute polygon area with the shoelace formula.

    Works in any consistent unit (normalized or pixel) — used both for real
    (pixel) area and for comparing instances by relative size.

    Parameters
    ----------
    points : Iterable[tuple[float, float]]
        Polygon vertices in order.

    Returns
    -------
    float
        The polygon's unsigned area.
    """
    pts = list(points)
    n = len(pts)
    s = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def polygon_bbox(points: Iterable[tuple[float, float]]) -> tuple[float, float, float, float]:
    """Compute the axis-aligned bounding box of a polygon.

    Parameters
    ----------
    points : Iterable[tuple[float, float]]
        Polygon vertices, each an (x, y) pair.

    Returns
    -------
    tuple[float, float, float, float]
        ``(x, y, width, height)`` of the bounding box.
    """
    xs, ys = zip(*points, strict=True)
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return x1, y1, x2 - x1, y2 - y1


def write_coco_json(path: Path, image_name: str, width: int, height: int, instances: list[YoloInstance]) -> None:
    """Write one per-image COCO polygon JSON from real YOLO-segment instances.

    Existing feature code reserves category_id=0 for background. COCO's "person"
    is class 0, so every YOLO class id is offset by +1 here.
    """
    annotations = []
    for ann_id, inst in enumerate(instances, start=1):
        points_px = [(x * width, y * height) for x, y in inst.points]
        x, y, w, h = polygon_bbox(points_px)
        if w <= 0 or h <= 0:
            continue
        flat = [c for p in points_px for c in p]
        annotations.append(
            {
                "id": ann_id,
                "image_id": 1,
                "category_id": inst.class_id + 1,
                "bbox": [x, y, w, h],
                "area": polygon_area(points_px),
                "iscrowd": 0,
                "segmentation": [flat],
            }
        )

    doc = {
        "images": [{"id": 1, "file_name": image_name, "width": width, "height": height}],
        "annotations": annotations,
        "categories": [{"id": 0, "name": "background"}]
        + [{"id": i + 1, "name": name} for i, name in enumerate(COCO80)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True))


def stage_file(src: Path, dst: Path, mode: str) -> None:
    """Copy, hardlink, or symlink one image into the staging directory.

    Parameters
    ----------
    src : Path
        Source file.
    dst : Path
        Destination path; removed first if it already exists.
    mode : str
        One of ``"copy"``, ``"hardlink"``, or ``"symlink"``. Hardlinking
        falls back to a copy if the filesystem does not support it.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    if mode == "symlink":
        dst.symlink_to(os.path.relpath(src, dst.parent))
        return
    shutil.copy2(src, dst)


def dominant_class(instances: list[YoloInstance]) -> str:
    """Return the class name of the largest-area instance.

    Parameters
    ----------
    instances : list[YoloInstance]
        Parsed instances for one image.

    Returns
    -------
    str
        The COCO80 class name of the largest polygon, or ``"background"``
        when there are no instances.
    """
    if not instances:
        return "background"
    # Normalized-space area comparison preserves the same ordering as pixel area
    # (every instance in one image shares the same width/height scale factor).
    best = max(instances, key=lambda inst: polygon_area(inst.points))
    return COCO80[best.class_id]


def first_class(instances: list[YoloInstance]) -> str:
    """Return the class name of the first instance in label order.

    Parameters
    ----------
    instances : list[YoloInstance]
        Parsed instances for one image.

    Returns
    -------
    str
        The COCO80 class name of the first instance, or ``"background"``
        when there are no instances.
    """
    if not instances:
        return "background"
    return COCO80[instances[0].class_id]


def folder_cluster(image_path: Path, images_dir: Path) -> tuple[str, str]:
    """Derive cluster labels from an image's parent folder path.

    Parameters
    ----------
    image_path : Path
        Path to the image file.
    images_dir : Path
        Root directory ``image_path`` is relative to.

    Returns
    -------
    tuple[str, str]
        ``(cluster, cluster_l2)`` — the full relative folder path and its
        first path segment, or ``("all", "all")`` when the image sits
        directly under ``images_dir``.
    """
    parent = image_path.relative_to(images_dir).parent
    if not parent.parts:
        return "all", "all"
    return parent.as_posix(), parent.parts[0]


def pick_cluster(mode: str, image_path: Path, images_dir: Path, instances: list[YoloInstance]) -> tuple[str, str]:
    """Compute the ground-truth cluster labels for one image.

    Parameters
    ----------
    mode : str
        One of ``"single"``, ``"folder"``, ``"first-class"``, or
        ``"dominant-class"``.
    image_path : Path
        Path to the image file.
    images_dir : Path
        Root directory images are staged under.
    instances : list[YoloInstance]
        Parsed instances for the image.

    Returns
    -------
    tuple[str, str]
        ``(cluster, cluster_l2)`` labels selected according to ``mode``.
    """
    if mode == "single":
        return "all", "all"
    if mode == "folder":
        return folder_cluster(image_path, images_dir)
    if mode == "first-class":
        name = first_class(instances)
        return name, "background" if name == "background" else "object"
    name = dominant_class(instances)
    return name, "background" if name == "background" else "object"


def iter_images(images_dir: Path) -> Iterable[Path]:
    """List image files under a directory, sorted for deterministic ordering.

    Parameters
    ----------
    images_dir : Path
        Directory to search recursively.

    Returns
    -------
    Iterable[Path]
        Image file paths whose suffix is a recognized image extension.
    """
    return sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


MANIFEST_FIELDNAMES = [
    "image_path",
    "cluster",
    "cluster_l2",
    "primary_class",
    "object_count",
    "source_path",
]

METADATA_FIELDNAMES = [
    "id",
    "original_name",
    "uuid",
    "camera_view",
    "angle",
    "gsd",
    "domain_metric",
]


def _stage_one_image(
    src: Path,
    images_dir: Path,
    labels_dir: Path,
    stage_images: Path,
    stage_labels: Path,
    root: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, str], str, int, bool]:
    """Stage one source image, its COCO polygon labels, and its manifest row.

    Parameters
    ----------
    src : Path
        Source image file.
    images_dir : Path
        Root images directory ``src`` is relative to.
    labels_dir : Path
        Root YOLO-segment labels directory matching ``images_dir``.
    stage_images : Path
        Destination directory for staged images.
    stage_labels : Path
        Destination directory for the generated COCO polygon JSON files.
    root : Path
        Repository root, used to compute repo-relative paths.
    args : argparse.Namespace
        Parsed CLI arguments (``stage_mode`` and ``cluster_by`` are used).

    Returns
    -------
    tuple[dict[str, object], dict[str, str], str, int, bool]
        The manifest row, the metadata row, the checksum line, the number of
        parsed instances, and whether the YOLO label file was missing.
    """
    rel = src.relative_to(images_dir)
    staged = stage_images / rel
    stage_file(src, staged, args.stage_mode)

    yolo_path = labels_dir / rel.with_suffix(".txt")
    missing_label = not yolo_path.exists()
    instances = read_yolo_seg_labels(yolo_path)

    with Image.open(staged) as img:
        width, height = img.size
    write_coco_json(stage_labels / rel.with_suffix(".json"), staged.name, width, height, instances)

    cluster, cluster_l2 = pick_cluster(args.cluster_by, src, images_dir, instances)
    row = {
        "image_path": staged.relative_to(root).as_posix(),
        "cluster": cluster,
        "cluster_l2": cluster_l2,
        "primary_class": dominant_class(instances),
        "object_count": len(instances),
        "source_path": src.relative_to(root).as_posix(),
    }
    metadata_row = {
        "id": staged.stem,
        "original_name": staged.stem,
        "uuid": staged.stem,
        "camera_view": "",
        "angle": "",
        "gsd": "",
        "domain_metric": "",
    }
    checksum_line = f"{sha256_file(src)}  {src.relative_to(root).as_posix()}"
    return row, metadata_row, checksum_line, len(instances), missing_label


def _write_manifest_csv(manifest: Path, rows: list[dict[str, object]]) -> None:
    """Write the per-image staging manifest CSV.

    Parameters
    ----------
    manifest : Path
        Output CSV path.
    rows : list[dict[str, object]]
        One row per staged image, keyed by ``MANIFEST_FIELDNAMES``.
    """
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _write_metadata_csv(stage_meta: Path, metadata_rows: list[dict[str, str]]) -> None:
    """Write the per-image metadata CSV.

    Parameters
    ----------
    stage_meta : Path
        Output directory; the file is written to ``images_metadata.csv``
        inside it.
    metadata_rows : list[dict[str, str]]
        One row per staged image, keyed by ``METADATA_FIELDNAMES``.
    """
    stage_meta.mkdir(parents=True, exist_ok=True)
    with (stage_meta / "images_metadata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDNAMES)
        writer.writeheader()
        writer.writerows(metadata_rows)


def stage_coco128(
    args: argparse.Namespace, root: Path, coco_root: Path, stage_root: Path | None = None
) -> tuple[Path, dict]:
    """Stage COCO128 images, convert labels to COCO polygons, and write the manifest.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments (``dataset_stem``, ``limit``, ``stage_mode``,
        ``cluster_by``).
    root : Path
        Repository root, used to compute repo-relative paths in the manifest.
    coco_root : Path
        Root of the extracted COCO128 archive (containing
        ``images/train2017`` and ``labels/train2017``).
    stage_root : Path | None
        Destination directory for staged images/labels/metadata; defaults to
        ``root / "image_sets" / args.dataset_stem``.

    Returns
    -------
    tuple[Path, dict]
        The written manifest CSV path and a dict of staging statistics.
    """
    images_dir = coco_root / "images" / "train2017"
    labels_dir = coco_root / "labels" / "train2017"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"missing image directory: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"missing label directory: {labels_dir}")

    stage_root = stage_root or Path(root / "image_sets" / args.dataset_stem)
    stage_images = stage_root / "images"
    stage_labels = stage_root / "labels"
    stage_meta = stage_root / "metadata"
    manifest = stage_root / f"{args.dataset_stem}_input.csv"
    checksums = stage_root / f"{args.dataset_stem}_checksums.sha256"

    images = list(iter_images(images_dir))
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise RuntimeError("no COCO128 images found")

    rows = []
    checksum_lines = []
    metadata_rows = []
    annotation_count = 0
    missing_yolo_labels = 0

    log(f"Staging {len(images)} image(s) into {stage_root}")
    for src in images:
        row, metadata_row, checksum_line, instance_count, missing_label = _stage_one_image(
            src, images_dir, labels_dir, stage_images, stage_labels, root, args
        )
        rows.append(row)
        metadata_rows.append(metadata_row)
        checksum_lines.append(checksum_line)
        annotation_count += instance_count
        if missing_label:
            missing_yolo_labels += 1

    _write_manifest_csv(manifest, rows)
    _write_metadata_csv(stage_meta, metadata_rows)
    checksums.write_text("\n".join(checksum_lines) + "\n")

    clusters: dict[str, int] = {}
    for row in rows:
        clusters[row["cluster"]] = clusters.get(row["cluster"], 0) + 1
    stats = {
        "rows": len(rows),
        "annotations": annotation_count,
        "missing_yolo_label_files": missing_yolo_labels,
        "clusters": clusters,
        "manifest": str(manifest.relative_to(root)),
        "checksums": str(checksums.relative_to(root)),
    }
    log(f"Manifest: {stats['manifest']}")
    log(f"Staged {annotation_count} real segmentation instance(s) as COCO polygons")
    return manifest, stats


def _ensure_importable(module_name: str, hint: str) -> None:
    """Raise a clear error if a module required for embeddings cannot be imported.

    Parameters
    ----------
    module_name : str
        Module to import-check (e.g. ``"torch"``).
    hint : str
        Extra guidance appended to the error message.

    Raises
    ------
    RuntimeError
        If the module cannot be imported.
    """
    try:
        __import__(module_name)
    except Exception as exc:
        raise RuntimeError(f"{module_name} is required for embeddings. {hint}") from exc


def run_pipeline(
    args: argparse.Namespace, root: Path, manifest: Path, data_dir: Path | None = None
) -> tuple[Path, Path | None]:
    """Run feature extraction and, optionally, embeddings generation.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments (``dataset_stem``, ``device``, ``with_nima``,
        ``no_embeddings``, ``embedding_model``).
    root : Path
        Repository root; ``tools.features``/``tools.embeddings`` are invoked
        with paths relative to it.
    manifest : Path
        Input manifest CSV produced by :func:`stage_coco128`.
    data_dir : Path | None
        Output directory for the feature CSV and embeddings JSON; defaults to
        ``root / "data"``.

    Returns
    -------
    tuple[Path, Path | None]
        The feature CSV path, and the embeddings JSON path (``None`` when
        ``args.no_embeddings`` is set).
    """
    # This import cannot move to module level: this script is designed to run from an
    # uninstalled checkout (see module docstring), so precisionai.magnidata.tools is not
    # importable until sys.path is patched with the repo root, immediately above.
    sys.path.insert(0, str(root))
    from precisionai.magnidata.tools import embeddings, features

    data_dir = data_dir or (root / "data")
    data_dir.mkdir(exist_ok=True)
    feature_csv = data_dir / f"{args.dataset_stem}.csv"
    embeddings_json = data_dir / f"{args.dataset_stem}.json"

    if args.with_nima:
        try:
            __import__("pyiqa")
        except Exception as exc:
            raise RuntimeError(
                "--with-nima was requested, but pyiqa is not importable. "
                "Install precisionai/magnidata/requirements.txt first."
            ) from exc

    log(f"Generating features -> {feature_csv.relative_to(root)}")
    features.run(
        str(manifest.relative_to(root)),
        str(feature_csv.relative_to(root)),
        device=args.device,
        skip_nima=not args.with_nima,
        image_mode="fullres",
    )

    if args.no_embeddings:
        return feature_csv, None

    for module_name in ("torch", "timm"):
        _ensure_importable(
            module_name,
            "Install precisionai/magnidata/requirements.txt or rerun with --no-embeddings.",
        )

    log(f"Generating {args.embedding_model} embeddings -> {embeddings_json.relative_to(root)}")
    embeddings.run(
        str(feature_csv.relative_to(root)),
        str(embeddings_json.relative_to(root)),
        model=args.embedding_model,
        device=args.device,
        image_mode="fullres",
    )
    return feature_csv, embeddings_json


def register_dataset(root: Path, stem: str, name: str, description: str) -> None:
    """Append a dataset entry to confi.yaml if it is not already registered.

    Parameters
    ----------
    root : Path
        Repository root.
    stem : str
        Dataset stem; the CSV source is ``data/<stem>.csv``.
    name : str
        Dashboard card name.
    description : str
        Dashboard card description.
    """
    config = root / "precisionai" / "magnidata" / "confi.yaml"
    source = f"data/{stem}.csv"
    text = config.read_text() if config.exists() else "datasets:\n"
    if f"source: {source}" in text or f'source: "{source}"' in text:
        log(f"{source} is already registered in {config.relative_to(root)}")
        return
    if not text.rstrip():
        text = "datasets:\n"
    if not text.endswith("\n"):
        text += "\n"
    text += f"\n- name: {json.dumps(name)}\n  description: {json.dumps(description)}\n  source: {source}\n"
    config.write_text(text)
    log(f"Registered {source} in {config.relative_to(root)}")


def validate_outputs(root: Path, feature_csv: Path, embeddings_json: Path | None) -> dict:
    """Validate the generated feature CSV and, if present, embeddings JSON.

    Parameters
    ----------
    root : Path
        Repository root; image paths in the CSV are resolved relative to it.
    feature_csv : Path
        Generated feature CSV to validate.
    embeddings_json : Path | None
        Generated embeddings JSON to validate, or ``None`` to skip that check.

    Returns
    -------
    dict
        Validation statistics: row/column counts, cluster histogram, any
        missing image paths, and embedding coverage.

    Raises
    ------
    RuntimeError
        If the CSV has no rows, references missing images, or the embeddings
        do not match the CSV image basenames.
    """
    with feature_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"{feature_csv} has no rows")

    missing_images = [row["image_path"] for row in rows if not (root / row["image_path"]).is_file()]
    clusters: dict[str, int] = {}
    for row in rows:
        clusters[row.get("cluster", "")] = clusters.get(row.get("cluster", ""), 0) + 1

    result = {
        "csv_rows": len(rows),
        "csv_columns": len(rows[0]),
        "clusters": clusters,
        "missing_images": missing_images,
        "embeddings": None,
        "embedding_dims": [],
        "missing_embeddings": [],
        "extra_embeddings": [],
    }

    if embeddings_json is not None:
        with embeddings_json.open() as f:
            doc = json.load(f)
        embeddings = doc.get("embeddings", doc)
        image_names = {Path(row["image_path"]).name for row in rows}
        emb_names = set(embeddings)
        result.update(
            {
                "embeddings": len(embeddings),
                "embedding_dims": sorted({len(v) for v in embeddings.values()}),
                "missing_embeddings": sorted(image_names - emb_names),
                "extra_embeddings": sorted(emb_names - image_names),
            }
        )
        if result["missing_embeddings"] or result["extra_embeddings"]:
            raise RuntimeError("embedding names do not match CSV image basenames")

    if missing_images:
        raise RuntimeError(f"{len(missing_images)} CSV image paths are missing")
    return result


def validate_api(stem: str) -> None:
    """Poll the running dashboard API until the dataset's endpoints respond.

    Parameters
    ----------
    stem : str
        Dataset stem used to build the ``source`` query parameter.

    Raises
    ------
    RuntimeError
        If an endpoint does not become ready within the poll deadline.
    """
    source = urllib.parse.quote(f"data/{stem}.csv", safe="")
    endpoints = {
        "csv": f"http://localhost:5175/api/dataset/csv?source={source}",
        "projection": f"http://localhost:5175/api/embedding/projection?source={source}&method=pca",
    }
    for label, url in endpoints.items():
        if urllib.parse.urlparse(url).scheme not in ("http", "https"):
            raise ValueError(f"unsupported URL scheme for {label} endpoint: {url!r}")
        deadline = time.monotonic() + 90
        last_exc: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=10) as response:
                    if response.status != 200:
                        raise RuntimeError(f"{label} endpoint returned {response.status}")
                    if label == "projection":
                        payload = json.load(response)
                        positions = payload.get("positions", [])
                        if not positions:
                            raise RuntimeError("projection endpoint returned no positions")
                break
            except (ConnectionError, OSError, urllib.error.URLError, RuntimeError, json.JSONDecodeError) as exc:
                last_exc = exc
                time.sleep(2)
        else:
            raise RuntimeError(f"{label} endpoint did not become ready within 90s") from last_exc
    log("API validation passed via http://localhost:5175")


def start_dashboard(root: Path) -> None:
    """Start the dashboard stack with Docker Compose.

    Parameters
    ----------
    root : Path
        Repository root; used as the Compose project directory.

    Raises
    ------
    RuntimeError
        If the ``docker`` executable cannot be found on ``PATH``.
    """
    log("Starting dashboard with docker compose")
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError("docker executable not found on PATH")
    subprocess.run([docker, "compose", "up", "--build", "-d"], cwd=root, check=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the COCO128 dashboard pipeline.

    Parameters
    ----------
    argv : list[str] | None
        Argument list to parse; defaults to ``sys.argv[1:]`` when ``None``.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    ap = argparse.ArgumentParser(description="Download COCO128 and build an magnidata dashboard dataset.")
    ap.add_argument("--dataset-stem", default="coco128", help="output stem: data/<stem>.csv and data/<stem>.json")
    ap.add_argument("--dataset-name", default="COCO128", help="dashboard card name")
    ap.add_argument("--description", default="Ultralytics COCO128 prepared for magnidata")
    ap.add_argument("--download-url", default=COCO128_URL, help="COCO128 zip URL")
    ap.add_argument("--cache-dir", default="image_sets/_downloads", help="download/extract cache under the repo root")
    ap.add_argument("--force-download", action="store_true", help="redownload the zip even if it is cached")
    ap.add_argument("--force-extract", action="store_true", help="re-extract the cached zip")
    ap.add_argument(
        "--stage-mode",
        choices=["copy", "hardlink", "symlink"],
        default="copy",
        help="how to stage images under image_sets/<stem>/images",
    )
    ap.add_argument(
        "--cluster-by",
        choices=["dominant-class", "first-class", "folder", "single"],
        default="dominant-class",
        help="ground-truth cluster column source; dominant-class is useful because COCO128 is flat",
    )
    ap.add_argument("--limit", type=int, default=None, help="optional first-N subset for quick smoke tests")
    ap.add_argument("--device", default=os.environ.get("DEVICE", "cpu"), help="torch device: cpu or cuda")
    ap.add_argument(
        "--with-nima",
        action="store_true",
        help="run learned IQA (NIMA/NIQE/BRISQUE). Default skips it for a faster local smoke run.",
    )
    ap.add_argument("--embedding-model", default="dinov2", help="embedding model registry key")
    ap.add_argument("--no-embeddings", action="store_true", help="skip embeddings JSON generation")
    ap.add_argument("--no-register", action="store_true", help="do not append the dataset to confi.yaml")
    ap.add_argument("--start-dashboard", action="store_true", help="run docker compose up --build -d after generation")
    ap.add_argument("--validate-api", action="store_true", help="validate the running dashboard API on localhost:5175")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the end-to-end COCO128 download, staging, and dashboard pipeline.

    Parameters
    ----------
    argv : list[str] | None
        Argument list to parse; defaults to ``sys.argv[1:]`` when ``None``.
    """
    args = parse_args(argv)
    root = repo_root()
    os.chdir(root)

    cache_dir = root / args.cache_dir
    # Named after the actual URL (not hardcoded to "coco128.zip") so switching
    # --download-url (e.g. box-only coco128 vs. the default coco128-seg) can never
    # silently reuse a stale cached zip from a different source.
    zip_path = cache_dir / Path(urllib.parse.urlparse(args.download_url).path).name
    extract_dir = cache_dir / "extracted"

    download(args.download_url, zip_path, force=args.force_download)
    coco_root = extract(zip_path, extract_dir, force=args.force_extract)
    log(f"COCO128 root: {coco_root.relative_to(root)}")

    manifest, stage_stats = stage_coco128(args, root, coco_root)
    feature_csv, embeddings_json = run_pipeline(args, root, manifest)

    if not args.no_register:
        register_dataset(root, args.dataset_stem, args.dataset_name, args.description)

    validation = validate_outputs(root, feature_csv, embeddings_json)
    log("Local validation:")
    print(json.dumps({"staging": stage_stats, "outputs": validation}, indent=2, sort_keys=True))

    if args.start_dashboard:
        start_dashboard(root)
    if args.validate_api or args.start_dashboard:
        validate_api(args.dataset_stem)

    log("Done. Open http://localhost:5175 and load the dataset card.")


if __name__ == "__main__":
    main()
