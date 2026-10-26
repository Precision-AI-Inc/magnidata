# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tile a dataset into fixed-size crops, split them into sub-datasets and register their lineage.

The sub-datasets are built with the portal's own *Prepare From Server Folder* pipeline, so each one gets
the standard feature columns and appears in the workspace lineage graph as a child of the source dataset.

Run from the repository root with the stack up::

    python -m precisionai.magnidata.scripts.partition_dataset --limit-images 24
"""

import argparse
import contextlib
import csv
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from precisionai.magnidata.training import contract
from precisionai.magnidata.training.dataset_check import find_mask, resolve_under

DEFAULT_SOURCE = "data_user/agristress500.csv"
DEFAULT_NAME = "AgriStress-500 tiles"
MANIFEST_NAME = "tile_manifest.csv"
MANIFEST_FIELDS: list[str] = ["tile_name", "source_image", "x", "y", "width", "height"]
POLL_SECONDS = 2.0


@dataclass(frozen=True)
class SourceImage:
    """One source image with its mask (if any) and the number of full tiles it yields."""

    name: str
    image: Path
    mask: Path | None
    width: int
    height: int
    tiles: int


def tile_origins(width: int, height: int, size: int) -> list[tuple[int, int]]:
    """Return the top-left corner of every full ``size`` x ``size`` tile, row by row; partial edge tiles are dropped."""
    return [(x, y) for y in range(0, height - size + 1, size) for x in range(0, width - size + 1, size)]


def tile_name(stem: str, x: int, y: int) -> str:
    """Return the file name of the tile of ``stem`` whose top-left corner is ``(x, y)``."""
    return f"{stem}_x{x:05d}_y{y:05d}.png"


def split_images(counts: dict[str, int], parts: int, seed: int) -> list[list[str]]:
    """Distribute whole images over ``parts`` groups so the tile counts stay as even as possible.

    Images are placed largest first on the currently smallest group (ties broken by a seeded shuffle), so no
    source image is shared between groups. That keeps the groups independent when one is used to validate
    a model trained on another.
    """
    if parts < 1:
        raise ValueError("parts must be at least 1")
    names = sorted(counts)
    random.Random(seed).shuffle(names)
    names.sort(key=lambda name: -counts[name])
    groups: list[list[str]] = [[] for _ in range(parts)]
    totals = [0] * parts
    for name in names:
        target = totals.index(min(totals))
        groups[target].append(name)
        totals[target] += counts[name]
    return [sorted(group) for group in groups]


def split_tiles(tiles: Sequence[tuple[str, int, int]], parts: int, seed: int) -> list[list[tuple[str, int, int]]]:
    """Deal individual tiles round-robin after a seeded shuffle; group sizes differ by at most one."""
    if parts < 1:
        raise ValueError("parts must be at least 1")
    shuffled = sorted(tiles)
    random.Random(seed).shuffle(shuffled)
    return [sorted(shuffled[index::parts]) for index in range(parts)]


def slugify(name: str) -> str:
    """Return the slug the API derives from a folder or dataset name."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_").lower() or "dataset"


def part_folder(prefix: str, index: int) -> str:
    """Return the ``image_sets`` folder name of the 1-based part ``index``."""
    return f"{slugify(prefix)}-part{index}"


def read_source(csv_path: Path, data_root: Path, tile: int, limit: int | None, seed: int) -> list[SourceImage]:
    """Read the source CSV and describe the images (and masks) that yield at least one full tile."""
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        column = contract.image_path_column(reader.fieldnames or [])
        if column is None:
            raise ValueError(f"{csv_path} has no image path column")
        rows = [row for row in reader if row.get(column)]
    if limit is not None and limit < len(rows):
        rows = sorted(random.Random(seed).sample(rows, limit), key=lambda row: row[column])
    images: list[SourceImage] = []
    for row in rows:
        relative = row[column]
        path = resolve_under(data_root, relative)
        if path is None or not path.is_file():
            continue
        with Image.open(path) as opened:
            width, height = opened.size
        mask = find_mask(relative, data_root, row)
        tiles = len(tile_origins(width, height, tile))
        if tiles:
            images.append(SourceImage(path.name, path, mask, width, height, tiles))
    return images


def write_part(
    folder: Path, assignments: dict[str, list[tuple[int, int]]], images: dict[str, SourceImage], tile: int
) -> int:
    """Write the tiles of one part under ``folder`` (``images/``, ``masks/`` and a manifest) and return the count."""
    (folder / "images").mkdir(parents=True, exist_ok=True)
    (folder / "masks").mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    for name in sorted(assignments):
        source = images[name]
        stem = source.image.stem
        with Image.open(source.image) as image:
            mask_image = Image.open(source.mask) if source.mask else None
            try:
                if mask_image is not None and mask_image.size != image.size:
                    mask_image = mask_image.resize(image.size, Image.Resampling.NEAREST)
                for x, y in assignments[name]:
                    box = (x, y, x + tile, y + tile)
                    out = tile_name(stem, x, y)
                    image.crop(box).save(folder / "images" / out)
                    if mask_image is not None:
                        mask_image.crop(box).save(folder / "masks" / out)
                    rows.append({"tile_name": out, "source_image": name, "x": x, "y": y, "width": tile, "height": tile})
            finally:
                if mask_image is not None:
                    mask_image.close()
    with open(folder / MANIFEST_NAME, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def part_is_complete(folder: Path, expected: int) -> bool:
    """Return True when ``folder`` already holds a manifest and image files for exactly ``expected`` tiles."""
    manifest = folder / MANIFEST_NAME
    if not manifest.is_file():
        return False
    with open(manifest, newline="") as handle:
        listed = sum(1 for _ in csv.DictReader(handle))
    images = folder / "images"
    return listed == expected and images.is_dir() and sum(1 for _ in images.glob("*.png")) == expected


class ApiError(RuntimeError):
    """The MagniData API rejected a request or could not be reached."""


class ApiClient:
    """Minimal client for the dataset endpoints this tool needs."""

    def __init__(self, base_url: str) -> None:
        """Create a client for the API at ``base_url`` (for example ``http://localhost:5051``)."""
        self.base_url = base_url.rstrip("/")

    def _request(self, method: str, path: str, form: dict[str, str] | None = None, body: Any = None) -> Any:
        data = None
        headers = {}
        if form is not None:
            data = urllib.parse.urlencode(form).encode()
        elif body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.base_url + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read() or b"null")
        except urllib.error.HTTPError as err:
            detail = err.read().decode(errors="replace")
            with contextlib.suppress(ValueError):
                detail = json.loads(detail).get("error", detail)
            raise ApiError(f"{method} {path} failed ({err.code}): {detail}") from None
        except urllib.error.URLError as err:
            raise ApiError(f"cannot reach the API at {self.base_url}: {err.reason}") from None

    def datasets(self) -> list[dict[str, Any]]:
        """Return the registered datasets."""
        return self._request("GET", "/api/datasets")

    def local_folders(self) -> list[dict[str, Any]]:
        """Return the ``image_sets`` folders the API can build from."""
        return self._request("GET", "/api/datasets/build/local-folders")

    def build_local(self, folder: str, name: str, description: str) -> str:
        """Start building a dataset from an ``image_sets`` folder and return the job id."""
        form = {"source_type": "local", "folder": folder, "name": name, "description": description}
        return self._request("POST", "/api/datasets/build", form=form)["job_id"]

    def job(self, job_id: str) -> dict[str, Any]:
        """Return the status of a build job."""
        return self._request("GET", f"/api/datasets/build/{job_id}")

    def set_parents(self, dataset_id: int, parents: list[str]) -> dict[str, Any]:
        """Replace the lineage parents of a registered dataset."""
        return self._request("POST", f"/api/datasets/{dataset_id}/parents", body={"parent_sources": parents})


def wait_for_job(client: ApiClient, job_id: str, label: str) -> dict[str, Any]:
    """Poll a build job until it finishes and return the registered dataset record."""
    last = ""
    while True:
        status = client.job(job_id)
        line = f"  {label}: {status.get('status')} {status.get('percent') or 0}% {status.get('message') or ''}".rstrip()
        if line != last:
            print(line, flush=True)
            last = line
        if status.get("status") == "done":
            return status["dataset"]
        if status.get("status") == "error":
            raise ApiError(f"building {label} failed: {status.get('error') or status.get('message')}")
        time.sleep(POLL_SECONDS)


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python -m precisionai.magnidata.scripts.partition_dataset",
        description="Tile a dataset, split the tiles into sub-datasets and register them under the source in the lineage graph.",
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="source dataset CSV, relative to the data root")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", "precisionai/magnidata")),
        help="folder that holds data_user/ (host path of the API's DATA_ROOT)",
    )
    parser.add_argument(
        "--image-sets", type=Path, default=Path("image_sets"), help="host folder mounted as image_sets/"
    )
    parser.add_argument("--name", default=DEFAULT_NAME, help="dataset name prefix; parts are named '<name> part N'")
    parser.add_argument("--tile", type=int, default=256, help="tile size in pixels")
    parser.add_argument("--parts", type=int, default=2, help="number of sub-datasets")
    parser.add_argument(
        "--split-by",
        choices=("image", "tile"),
        default="image",
        help="image keeps all tiles of a source image in one part (no leakage between parts)",
    )
    parser.add_argument("--limit-images", type=int, default=None, help="only tile a seeded sample of N source images")
    parser.add_argument("--seed", type=int, default=0, help="seed for sampling and splitting")
    parser.add_argument(
        "--api", default=os.environ.get("MAGNIDATA_API_URL", "http://localhost:5051"), help="API base URL"
    )
    parser.add_argument(
        "--portal", default=os.environ.get("MAGNIDATA_PORTAL_URL", "http://localhost:5175"), help="portal URL"
    )
    parser.add_argument(
        "--no-register", action="store_true", help="only write the tiles; do not build or register datasets"
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit without writing anything")
    return parser


def plan_parts(
    images: list[SourceImage], parts: int, split_by: str, seed: int, tile: int
) -> list[dict[str, list[tuple[int, int]]]]:
    """Return, per part, ``{source image name: [tile origins]}``."""
    origins = {image.name: tile_origins(image.width, image.height, tile) for image in images}
    if split_by == "image":
        groups = split_images({name: len(points) for name, points in origins.items()}, parts, seed)
        return [{name: origins[name] for name in group} for group in groups]
    flat = [(name, x, y) for name, points in origins.items() for x, y in points]
    plan: list[dict[str, list[tuple[int, int]]]] = []
    for group in split_tiles(flat, parts, seed):
        part: dict[str, list[tuple[int, int]]] = {}
        for name, x, y in group:
            part.setdefault(name, []).append((x, y))
        plan.append(part)
    return plan


def tile_parts(
    args: argparse.Namespace, plan: list[dict[str, list[tuple[int, int]]]], images: list[SourceImage]
) -> None:
    """Write every part's tiles, skipping parts that are already complete."""
    by_name = {image.name: image for image in images}
    for index, part in enumerate(plan, start=1):
        folder = args.image_sets / part_folder(args.name, index)
        expected = sum(len(points) for points in part.values())
        if part_is_complete(folder, expected):
            print(f"  part {index}: {folder} already tiled, skipping")
            continue
        print(f"  part {index}: writing {expected} tiles to {folder}", flush=True)
        write_part(folder, part, by_name, args.tile)


def register_parts(args: argparse.Namespace, parts: int) -> None:
    """Build each tiled folder into a dataset (unless it already is one) and attach it to the source in the lineage."""
    client = ApiClient(args.api)
    built = {entry["folder"] for entry in client.local_folders() if entry.get("built")}
    known = {dataset["source"]: dataset for dataset in client.datasets()}
    for index in range(1, parts + 1):
        folder_name = part_folder(args.name, index)
        label = f"{args.name} part {index}"
        source = f"data_user/{slugify(folder_name)}.csv"
        if folder_name in built and source in known:
            record = known[source]
            print(f"  {label}: already registered")
        else:
            job = client.build_local(folder_name, label, f"Tiles of {args.source}, part {index}")
            record = wait_for_job(client, job, label)
        client.set_parents(record["id"], [args.source])


def run(args: argparse.Namespace) -> int:
    """Execute the partitioning described by parsed ``args``; return a process exit code."""
    if args.parts < 1 or args.tile < 16:
        print("error: --parts must be at least 1 and --tile at least 16", file=sys.stderr)
        return 2
    csv_path = resolve_under(args.data_root, args.source)
    if csv_path is None or not csv_path.is_file():
        print(f"error: source dataset {args.source!r} not found under {args.data_root}", file=sys.stderr)
        return 1
    images = read_source(csv_path, args.data_root, args.tile, args.limit_images, args.seed)
    if not images:
        print("error: the source dataset has no readable images", file=sys.stderr)
        return 1
    plan = plan_parts(images, args.parts, args.split_by, args.seed, args.tile)
    masked = sum(1 for image in images if image.mask)
    print(
        f"Source {args.source}: {len(images)} images ({masked} with masks) -> {sum(i.tiles for i in images)} tiles of "
        f"{args.tile}x{args.tile}, split by {args.split_by} into {args.parts} parts"
    )
    for index, part in enumerate(plan, start=1):
        print(f"  part {index}: {len(part)} images, {sum(len(points) for points in part.values())} tiles")
    if args.dry_run:
        return 0
    tile_parts(args, plan, images)
    if not args.no_register:
        register_parts(args, len(plan))
        print(f"Done. Open {args.portal} to see the sub-datasets under {args.source} in the lineage graph.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line tool."""
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (ApiError, ValueError, OSError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
