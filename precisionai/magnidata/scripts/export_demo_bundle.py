# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

r"""Export a built dataset as a compact demo bundle another deployment can install.

A bundle keeps what the dashboard shows and drops what a small CPU-only deployment cannot afford:
the feature table and the embeddings are copied as they are (they were computed from the
full-resolution images), while the images are re-encoded as JPEGs capped at ``--max-size`` pixels,
with a smaller thumbnail set for the grid and the colour masks resized to match.

    python -m precisionai.magnidata.scripts.export_demo_bundle \
        --data-root /path/to/DATA_ROOT --stem agristress500 --name AgriStress-500 \
        --description "..." --out bundles/agristress500

Images are renamed to ``<stem>.jpg``. The embeddings file is keyed by image basename, so its keys
are renamed the same way and checked against the table before anything is reported as done.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import shutil
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

BUNDLE_FORMAT = 1
MANIFEST_NAME = "bundle.json"
USER_DATA_REL = "data_user"
IMAGE_MAX_SIZE = 2048
THUMBNAIL_MAX_SIZE = 720
JPEG_QUALITY = 85
MASK_SUFFIXES = (".png", ".jpg", ".jpeg")
MAX_REPORTED_PROBLEMS = 5


class BundleError(ValueError):
    """Raised when a dataset cannot be exported or a bundle is malformed."""


@dataclass(frozen=True)
class _Job:
    """One image to convert: where it comes from and where its three outputs go."""

    image: Path
    mask: Path | None
    image_out: Path
    thumbnail_out: Path
    mask_out: Path | None
    max_size: int
    thumbnail_size: int


def _convert(job: _Job) -> None:
    """Write the capped JPEG, its thumbnail and the matching mask for one image."""
    with Image.open(job.image) as source:
        full = source.convert("RGB")
    full.thumbnail((job.max_size, job.max_size), Image.Resampling.LANCZOS)  # never enlarges
    full.save(job.image_out, "JPEG", quality=JPEG_QUALITY, optimize=True)

    thumbnail = full.copy()
    thumbnail.thumbnail((job.thumbnail_size, job.thumbnail_size), Image.Resampling.LANCZOS)
    thumbnail.save(job.thumbnail_out, "JPEG", quality=JPEG_QUALITY, optimize=True)

    if job.mask is not None and job.mask_out is not None:
        with Image.open(job.mask) as mask:
            # Nearest-neighbour keeps every class colour exact.
            mask.convert("RGB").resize(full.size, Image.Resampling.NEAREST).save(job.mask_out, "PNG", optimize=True)


def _find_mask(data_root: Path, image_path: str) -> Path | None:
    """Return the ``masks/<stem>.<ext>`` file beside an ``images/<stem>.<ext>`` file, if any."""
    image = Path(image_path)
    for suffix in MASK_SUFFIXES:
        candidate = data_root / image.parent.parent / "masks" / f"{image.stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _read_table(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    if "image_path" not in columns:
        raise BundleError(f"{csv_path.name} has no image_path column")
    if not rows:
        raise BundleError(f"{csv_path.name} has no rows")
    return columns, rows


def _read_embeddings(path: Path) -> dict[str, list[float]]:
    with open(path) as handle:
        payload = json.load(handle)
    table = payload.get("embeddings", payload) if isinstance(payload, dict) else None
    if not isinstance(table, dict):
        raise BundleError(f"{path.name} is not an embeddings object")
    return {os.path.basename(str(name)): vector for name, vector in table.items() if isinstance(vector, list)}


def _plan(
    data_root: Path, rows: list[dict[str, str]], images_dir: Path, max_size: int, thumbnail_size: int
) -> tuple[list[_Job], list[str]]:
    """Return one conversion job per row, and the new image name of each row."""
    jobs: list[_Job] = []
    names: list[str] = []
    for row in rows:
        source = data_root / row["image_path"]
        if not source.is_file():
            raise BundleError(f"image not found: {row['image_path']}")
        name = f"{Path(row['image_path']).stem}.jpg"
        mask = _find_mask(data_root, row["image_path"])
        jobs.append(
            _Job(
                image=source,
                mask=mask,
                image_out=images_dir / "images" / name,
                thumbnail_out=images_dir / "thumbnails" / name,
                mask_out=images_dir / "masks" / f"{Path(name).stem}.png" if mask else None,
                max_size=max_size,
                thumbnail_size=thumbnail_size,
            )
        )
        names.append(name)
    if len(set(names)) != len(names):
        raise BundleError("two images share a name once their extension is dropped")
    return jobs, names


def _run_jobs(jobs: list[_Job], workers: int) -> None:
    if workers <= 1:
        for job in jobs:
            _convert(job)
        return
    # "spawn", not Linux's default "fork": forking a process that already runs threads can deadlock.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for _ in pool.map(_convert, jobs, chunksize=4):
            pass


def export_bundle(
    data_root: Path,
    stem: str,
    out_dir: Path,
    *,
    name: str,
    description: str,
    max_size: int = IMAGE_MAX_SIZE,
    thumbnail_size: int = THUMBNAIL_MAX_SIZE,
    workers: int | None = None,
) -> dict[str, Any]:
    """Write a bundle of the dataset ``data_user/<stem>.csv`` under ``data_root`` to ``out_dir``.

    Parameters
    ----------
    data_root : Path
        The ``DATA_ROOT`` the dataset was built in.
    stem : str
        Dataset stem: ``data_user/<stem>.csv``, ``<stem>.json`` and ``<stem>_images/``.
    out_dir : Path
        New, empty directory to write the bundle into.
    name, description : str
        Display name and description the installing deployment registers.
    max_size : int
        Longest side, in pixels, of the exported images.
    thumbnail_size : int
        Longest side, in pixels, of the thumbnails.
    workers : int | None
        Processes converting images; defaults to the CPU count, at most 8.

    Returns
    -------
    dict[str, Any]
        The manifest written to ``bundle.json``.

    Raises
    ------
    BundleError
        If an input is missing or malformed, or the finished bundle fails validation.
    """
    if out_dir.exists() and any(out_dir.iterdir()):
        raise BundleError(f"{out_dir} is not empty")
    columns, rows = _read_table(data_root / USER_DATA_REL / f"{stem}.csv")
    embeddings = _read_embeddings(data_root / USER_DATA_REL / f"{stem}.json")

    images_dir = out_dir / f"{stem}_images"
    for sub in ("images", "thumbnails", "masks"):
        (images_dir / sub).mkdir(parents=True, exist_ok=True)
    jobs, names = _plan(data_root, rows, images_dir, max_size, thumbnail_size)

    renamed: dict[str, list[float]] = {}
    for row, new_name in zip(rows, names, strict=True):
        vector = embeddings.get(os.path.basename(row["image_path"]))
        if vector is None:
            raise BundleError(f"no embedding for {os.path.basename(row['image_path'])}")
        renamed[new_name] = vector
        row["image_path"] = f"{USER_DATA_REL}/{stem}_images/images/{new_name}"
        if "thumbnail_image" in columns:
            row["thumbnail_image"] = f"{USER_DATA_REL}/{stem}_images/thumbnails/{new_name}"

    _run_jobs(jobs, workers if workers is not None else min(os.cpu_count() or 1, 8))
    if not any(job.mask for job in jobs):
        shutil.rmtree(images_dir / "masks")

    with open(out_dir / f"{stem}.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / f"{stem}.json", "w") as handle:
        json.dump({"embeddings": renamed}, handle)

    manifest = {
        "format": BUNDLE_FORMAT,
        "stem": stem,
        "name": name,
        "description": description,
        "row_count": len(rows),
        "image_max_size": max_size,
        "thumbnail_max_size": thumbnail_size,
        "has_masks": any(job.mask for job in jobs),
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    validate_bundle(out_dir)
    return manifest


def _missing_files(bundle_dir: Path, stem: str, rows: list[dict[str, str]], has_masks: bool) -> list[str]:
    problems: list[str] = []
    for row in rows:
        paths = [row["image_path"], row.get("thumbnail_image", "")]
        if has_masks:
            paths.append(f"{USER_DATA_REL}/{stem}_images/masks/{Path(row['image_path']).stem}.png")
        problems += [f"missing {rel}" for rel in paths if rel and not (bundle_dir / _strip_user_data(rel)).is_file()]
    return problems


def _strip_user_data(rel: str) -> str:
    """Map an installed ``data_user/...`` path onto the bundle's own layout."""
    prefix = f"{USER_DATA_REL}/"
    return rel[len(prefix) :] if rel.startswith(prefix) else rel


def validate_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Check that a bundle is complete and consistent, and return its manifest.

    Every table row must have its image, thumbnail and (when the bundle has masks) mask, and its
    embedding, found under the image's new basename.

    Raises
    ------
    BundleError
        Listing up to a handful of the problems found.
    """
    manifest_path = bundle_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise BundleError(f"{bundle_dir} has no {MANIFEST_NAME}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != BUNDLE_FORMAT:
        raise BundleError(f"unsupported bundle format {manifest.get('format')!r}")
    stem = manifest["stem"]
    for field in ("name", "description", "row_count"):
        if field not in manifest:
            raise BundleError(f"{MANIFEST_NAME} has no {field}")

    for required in (f"{stem}.csv", f"{stem}.json"):
        if not (bundle_dir / required).is_file():
            raise BundleError(f"{bundle_dir} has no {required}")
    _, rows = _read_table(bundle_dir / f"{stem}.csv")
    embeddings = _read_embeddings(bundle_dir / f"{stem}.json")

    problems: list[str] = []
    if len(rows) != manifest["row_count"]:
        problems.append(f"table has {len(rows)} rows, the manifest says {manifest['row_count']}")
    problems += _missing_files(bundle_dir, stem, rows, bool(manifest.get("has_masks")))
    problems += [
        f"no embedding for {os.path.basename(row['image_path'])}"
        for row in rows
        if os.path.basename(row["image_path"]) not in embeddings
    ]
    if problems:
        shown = "; ".join(problems[:MAX_REPORTED_PROBLEMS])
        more = f" (and {len(problems) - MAX_REPORTED_PROBLEMS} more)" if len(problems) > MAX_REPORTED_PROBLEMS else ""
        raise BundleError(f"invalid bundle: {shown}{more}")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Run the exporter from the command line and return a process exit code."""
    parser = argparse.ArgumentParser(description="Export a built dataset as a compact demo bundle.")
    parser.add_argument("--data-root", type=Path, required=True, help="DATA_ROOT the dataset was built in")
    parser.add_argument("--stem", required=True, help="dataset stem, e.g. agristress500")
    parser.add_argument("--name", required=True, help="display name to register")
    parser.add_argument("--description", required=True, help="description to register")
    parser.add_argument("--out", type=Path, required=True, help="new, empty output directory")
    parser.add_argument("--max-size", type=int, default=IMAGE_MAX_SIZE, help="longest image side in pixels")
    parser.add_argument("--thumbnail-size", type=int, default=THUMBNAIL_MAX_SIZE, help="longest thumbnail side")
    parser.add_argument("--workers", type=int, default=None, help="conversion processes (default: CPU count, max 8)")
    args = parser.parse_args(argv)
    try:
        manifest = export_bundle(
            args.data_root,
            args.stem,
            args.out,
            name=args.name,
            description=args.description,
            max_size=args.max_size,
            thumbnail_size=args.thumbnail_size,
            workers=args.workers,
        )
    except (BundleError, OSError) as err:
        print(f"export failed: {err}", file=sys.stderr)
        return 1
    print(f"exported {manifest['row_count']} images to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
