# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Build cloud unions and feature-similar dataset complements."""

import csv
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CLOUD_NAME = "Cloud Dataset"
CLOUD_SOURCE = "data_user/cloud-dataset.csv"
MAX_MERGE_SOURCES = 20
MINI_SOURCES = ("data_user/agristress500-mini1.csv", "data_user/agristress500-mini2.csv")


@dataclass(frozen=True)
class Derivation:
    """Rows and lineage metadata produced by one curation operation."""

    columns: list[str]
    rows: list[dict[str, str]]
    complement_count: int


@dataclass(frozen=True)
class CreateRequest:
    """Validated dataset-derivation request values."""

    name: str
    description: str
    parent_source: str
    image_names: list[str]
    query_image_names: list[str]
    cloud_source: str
    complement_top_k: int


def parse_create_request(body: dict[str, Any]) -> CreateRequest:
    """Validate and normalize a dataset-derivation request body."""
    name = str(body.get("name") or "").strip()
    parent = str(body.get("parent_source") or "").strip()
    images = body.get("image_names") or []
    query = body.get("query_image_names") or []
    cloud = str(body.get("cloud_source") or CLOUD_SOURCE).strip()
    try:
        top_k = int(body.get("complement_top_k") or 0)
    except (TypeError, ValueError):
        raise ValueError("complement_top_k must be an integer") from None
    if not name:
        raise ValueError("name required")
    if not parent or os.path.isabs(parent) or ".." in parent:
        raise ValueError("invalid parent_source")
    if not isinstance(images, list) or not images:
        raise ValueError("image_names required")
    if top_k < 0 or top_k > 5000:
        raise ValueError("complement_top_k must be between 0 and 5000")
    if top_k and (not isinstance(query, list) or not query):
        raise ValueError("query_image_names required for complement")
    if os.path.isabs(cloud) or ".." in cloud:
        raise ValueError("invalid cloud_source")
    return CreateRequest(name, str(body.get("description") or "").strip(), parent, images, query, cloud, top_k)


@dataclass(frozen=True)
class MergeRequest:
    """Validated request to merge several datasets into a new one."""

    name: str
    description: str
    sources: list[str]


def parse_merge_request(body: dict[str, Any]) -> MergeRequest:
    """Validate and normalize a dataset-merge request body."""
    name = str(body.get("name") or "").strip()
    raw = body.get("sources")
    if not name:
        raise ValueError("name required")
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError("sources must be a list of dataset sources")
    sources = list(dict.fromkeys(item.strip() for item in raw if item.strip()))
    if len(sources) < 2:
        raise ValueError("at least two distinct sources required")
    if len(sources) > MAX_MERGE_SOURCES:
        raise ValueError(f"at most {MAX_MERGE_SOURCES} sources can be merged")
    if any(os.path.isabs(source) or ".." in source for source in sources):
        raise ValueError("invalid source")
    return MergeRequest(name=name, description=str(body.get("description") or "").strip(), sources=sources)


def read_rows(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV's header and rows."""
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: str | Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    """Write rows using a fixed CSV schema."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def image_column(columns: list[str]) -> str | None:
    """Return the first supported image-path column."""
    return next((name for name in ("image_path", "file_path", "path", "image") if name in columns), None)


def image_name(row: dict[str, str], column: str) -> str:
    """Return the basename used as the cross-dataset sample key."""
    return os.path.basename(str(row.get(column, "")))


def combine_sources(paths: Sequence[str | Path]) -> tuple[list[str], list[dict[str, str]]]:
    """Return the de-duplicated union of compatible CSV sources."""
    columns: list[str] = []
    combined: list[dict[str, str]] = []
    seen: set[str] = set()
    key_column: str | None = None
    for path in paths:
        source_columns, rows = read_rows(path)
        if not columns:
            columns = source_columns
            key_column = image_column(columns)
        elif source_columns != columns:
            raise ValueError("cloud members must use the same CSV columns")
        if key_column is None:
            raise ValueError("cloud members need an image path column")
        for row in rows:
            key = image_name(row, key_column)
            if key and key not in seen:
                combined.append(row)
                seen.add(key)
    return columns, combined


def ensure_cloud_dataset(data_root: str, database: Any) -> dict[str, Any] | None:
    """Create the AgriStress cloud union and lineage edges when both mini datasets exist."""
    member_paths = [Path(data_root) / source for source in MINI_SOURCES]
    if not all(path.is_file() for path in member_paths):
        return None
    cloud_path = Path(data_root) / CLOUD_SOURCE
    record = database.get_created_dataset_by_source(CLOUD_SOURCE)
    if record is None or not cloud_path.is_file():
        columns, rows = combine_sources(member_paths)
        write_rows(cloud_path, columns, rows)
        if record is None:
            record = database.create_dataset_record(
                CLOUD_NAME,
                "Combined AgriStress sample cloud used for training and complement curation.",
                CLOUD_SOURCE,
                None,
                None,
                len(rows),
            )
    for source in MINI_SOURCES:
        if database.get_created_dataset_by_source(source) is not None:
            database.set_dataset_parents(source, [CLOUD_SOURCE])
    return record


def _number(value: str | None) -> float:
    try:
        number = float(value or "")
    except ValueError:
        return math.nan
    return number if math.isfinite(number) else math.nan


def _feature_space(
    query: list[dict[str, str]], cloud_rows: list[dict[str, str]], columns: list[str], key_column: str
) -> list[tuple[str, float, float]]:
    """Return usable columns as ``(name, query centroid, cloud scale)`` tuples."""
    usable = []
    for column in columns:
        if column == key_column:
            continue
        cloud_values = np.array([_number(row.get(column)) for row in cloud_rows], dtype=float)
        finite = cloud_values[np.isfinite(cloud_values)]
        query_values = np.array([_number(row.get(column)) for row in query], dtype=float)
        query_values = query_values[np.isfinite(query_values)]
        scale = float(np.std(finite)) if len(finite) >= 2 else 0.0
        if scale > 1e-12 and len(query_values):
            usable.append((column, float(np.mean(query_values)), scale))
    return usable


def _distance(row: dict[str, str], features: list[tuple[str, float, float]]) -> float | None:
    terms = [((_number(row.get(column)) - center) / scale) ** 2 for column, center, scale in features]
    finite = [term for term in terms if math.isfinite(term)]
    return sum(finite) / len(finite) if finite else None


def select_complement(
    current_rows: list[dict[str, str]],
    cloud_rows: list[dict[str, str]],
    columns: list[str],
    query_names: set[str],
    top_k: int,
) -> list[dict[str, str]]:
    """Select cloud-only rows nearest to the filtered query's standardized feature centroid."""
    key_column = image_column(columns)
    if key_column is None:
        raise ValueError("dataset needs an image path column")
    current_names = {image_name(row, key_column) for row in current_rows}
    query = [row for row in current_rows if image_name(row, key_column) in query_names]
    candidates = [row for row in cloud_rows if image_name(row, key_column) not in current_names]
    if not query:
        raise ValueError("filtered selection does not match the current dataset")
    if not candidates or top_k <= 0:
        return []

    usable = _feature_space(query, cloud_rows, columns, key_column)
    if not usable:
        raise ValueError("datasets have no shared numeric features for complement similarity")

    ranked: list[tuple[float, str, dict[str, str]]] = []
    for row in candidates:
        distance = _distance(row, usable)
        if distance is not None:
            ranked.append((distance, image_name(row, key_column), row))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [row for _, _, row in ranked[:top_k]]


def derive_dataset(
    parent_path: str | Path,
    image_names: list[str],
    *,
    cloud_path: str | Path | None = None,
    query_image_names: list[str] | None = None,
    top_k: int = 0,
) -> Derivation:
    """Retain named parent rows and optionally append a ranked cloud complement."""
    columns, parent_rows = read_rows(parent_path)
    key_column = image_column(columns)
    wanted = {os.path.basename(str(name)) for name in image_names}
    kept = [row for row in parent_rows if key_column and image_name(row, key_column) in wanted]
    if not kept:
        raise ValueError("no matching rows")
    if not top_k:
        return Derivation(columns, kept, 0)
    if cloud_path is None:
        raise ValueError("cloud dataset not found")
    cloud_columns, cloud_rows = read_rows(cloud_path)
    if cloud_columns != columns:
        raise ValueError("cloud and current datasets use different columns")
    complement = select_complement(
        parent_rows,
        cloud_rows,
        columns,
        {os.path.basename(str(name)) for name in query_image_names or []},
        top_k,
    )
    if not complement:
        raise ValueError("the cloud complement has no eligible samples")
    return Derivation(columns, [*kept, *complement], len(complement))
