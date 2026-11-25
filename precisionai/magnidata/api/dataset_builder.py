# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Orchestrate a "build a dataset from images" job.

Stage images, extract features, and register in the created_datasets table. Runs in
a background thread; progress is tracked in an in-process job dict polled by
GET /api/datasets/build/<job_id>, and every change is also written to the build_jobs
table so a reloaded page can find the running build again
(GET /api/datasets/build/current) and a build cut off by an API restart is reported as
interrupted, with its partial output removed, instead of being forgotten.

A build claims its output name (its "stem" under data_user/) by creating the dataset
CSV exclusively before any work starts, so nothing else — another build, or a selection
saved in the meantime — can take the same path while it runs (see reserve_stem).

This build pipeline does not compute embeddings — no model runs here. Embeddings are
bring-your-own: an upload build may optionally include a pre-computed embeddings JSON
(the standard {"embeddings": {"<basename>": [floats...]}} shape), which is staged
alongside the CSV as-is; a dataset built without one simply has no embeddings (the
"Embeddings" view is unavailable for it, same as any other CSV that never had a sibling
.json file).

Two build modes:
  - upload: a user-supplied batch of images (plus optional per-image COCO-format
    annotation JSON files, plus an optional embeddings JSON) is staged and built into
    a brand-new dataset.
  - demo: one of a small static registry of named demo datasets (see DEMOS) is built
    from a known source, both downloaded and staged locally with features computed
    here — COCO128 (Ultralytics, real segmentation polygons; a DINOv2 embeddings JSON
    for its fixed 128-image set is precomputed offline and shipped in scripts/, then
    just attached — see COCO128_EMBEDDINGS_ASSET) and AgriStress-500 (images + masks
    from its Hugging Face repo, thumbnails generated locally, embeddings brought in
    as-is from the CDN's pre-computed SEED-Embeddings.json). A disabled registry entry
    means nothing to build it from yet — see app.py's GET /api/datasets/demos.

Everything a build writes lives under DATA_ROOT/data_user/ — the only read-write volume
in the Docker deployment (data/, image_sets/, and confi.yaml are read-only). Built
datasets register with parent_source=None (first-class dataset, not a row-subset) and
emb_source=<own source> when an embeddings file was provided, else None — the existing
created_datasets schema and _resolve_emb_source() in app.py already handle exactly this
shape; no DB migration.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import re
import shutil
import sys
import threading
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import db
import embeddings_io
import projections
from PIL import Image


# ── Make the precisionai.magnidata.{tools,scripts} package importable ─────────────
# This file's own location tells us where the package chain lives, regardless of
# the process's cwd or DATA_ROOT (which is purely a data-output location, below):
#   - Docker layout: this file is copied flat to /app/dataset_builder.py, and
#     /app/precisionai/... sits right next to it (0 levels up).
#   - Dev/checkout layout: this file lives at precisionai/magnidata/api/dataset_builder.py,
#     3 levels below the repo root that contains precisionai/ (matching
#     scripts/prepare_coco128_dashboard.py's own repo_root() convention).
# Walk up from this file looking for a directory containing a "precisionai" package,
# and add the first match to sys.path.
def _find_package_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "precisionai" / "__init__.py").is_file():
            return candidate
    return start  # fall back; the subsequent import will raise a clear ImportError


_PACKAGE_ROOT = _find_package_root(Path(__file__).resolve().parent)
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from precisionai.magnidata.scripts import prepare_coco128_dashboard as coco128  # noqa: E402
from precisionai.magnidata.tools import features as features_mod  # noqa: E402
from precisionai.magnidata.tools import staging as staging_mod  # noqa: E402

# COCO128 is a small, fixed image set (same 128 files every time), so rather than
# computing embeddings at build time — the API image deliberately has no torch/timm,
# see Dockerfile.api — a DINOv2 embeddings JSON for it is precomputed once offline
# (scripts/prepare_coco128_dashboard.py --embedding-model dinov2) and checked into the
# repo next to that script, keyed by image basename like any other embeddings file.
# Located relative to coco128.__file__ (not this file's own __file__) since that
# resolves correctly under both the Docker layout (this file copied flat to /app/) and
# the dev/checkout layout (this file 3 levels below the repo root) — see
# _find_package_root above for why those differ.
COCO128_EMBEDDINGS_ASSET = Path(coco128.__file__).resolve().parent / "coco128_embeddings.json"

DATA_ROOT = os.environ.get("DATA_ROOT", ".")
USER_DATA_REL = "data_user"

INCOMING_UPLOAD_DIRNAME = "_incoming"

# Server-side image folders, mounted read-only at <DATA_ROOT>/image_sets. A dataset is
# built from one of these in place: nothing is copied and nothing is written back into
# the folder, so the upload size caps below do not apply to that path at all.
IMAGE_SETS_REL = "image_sets"
LOCAL_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


MAX_IMAGES = _env_int("DATASET_BUILD_MAX_IMAGES", 500)
MAX_TOTAL_BYTES = _env_int("DATASET_BUILD_MAX_TOTAL_BYTES", 16 * 1024 * 1024 * 1024)
MAX_EMBEDDINGS_BYTES = _env_int("DATASET_BUILD_MAX_EMBEDDINGS_BYTES", 100 * 1024 * 1024)
MAX_REQUEST_OVERHEAD_BYTES = _env_int("DATASET_BUILD_MAX_REQUEST_OVERHEAD_BYTES", 64 * 1024 * 1024)
MAX_REQUEST_BYTES = MAX_TOTAL_BYTES + MAX_EMBEDDINGS_BYTES + MAX_REQUEST_OVERHEAD_BYTES

COCO128_KEY = "coco128"
COCO128_STEM = "coco128"
COCO128_NAME = "COCO128"
COCO128_DESCRIPTION = (
    "Ultralytics COCO128 sample, built in-app (embeddings precomputed offline and shipped with the tool)"
)

AGRISTRESS_KEY = "agristress500"
AGRISTRESS_STEM = "agristress500"
AGRISTRESS_NAME = "AgriStress-500"
AGRISTRESS_DESCRIPTION = (
    "451 field images + real segmentation masks, downloaded from "
    "Hugging Face and processed locally (features computed here; "
    "embeddings brought in from the public CDN)"
)
# Images + masks: Hugging Face (the full-resolution source; downloaded and processed
# locally — features are computed from these, not taken pre-built from anywhere).
AGRISTRESS_HF_API_URL = "https://huggingface.co/api/datasets/precisionaiinc/AgriStress-500"
AGRISTRESS_HF_RESOLVE_BASE = "https://huggingface.co/datasets/precisionaiinc/AgriStress-500/resolve/main"
# Embeddings: still the CDN's pre-computed SEED-Embeddings.json — this pipeline never
# computes embeddings itself (see the module docstring), so these are brought in as-is.
AGRISTRESS_EMBEDDINGS_URL = "https://d379glmvb7evte.cloudfront.net/internal/data/embeddings/SEED-Embeddings.json"
AGRISTRESS_THUMB_MAX = 720
AGRISTRESS_DOWNLOAD_WORKERS = 8

# Static demo registry (GET /api/datasets/demos serves this verbatim).
DEMOS: list[dict] = [
    {"key": COCO128_KEY, "name": COCO128_NAME, "description": COCO128_DESCRIPTION, "enabled": True},
    {"key": AGRISTRESS_KEY, "name": AGRISTRESS_NAME, "description": AGRISTRESS_DESCRIPTION, "enabled": True},
]

# Demo datasets always build to these fixed stems, so no other dataset may take them.
DEMO_STEMS = frozenset({COCO128_STEM, AGRISTRESS_STEM})
# Everything a dataset with stem S may have next to its CSV under data_user/. A stem is only
# handed out while none of these exist, so whatever is found under a dataset's stem is its own.
EMBEDDING_FILE_SUFFIXES = (".json", ".json.provenance.json", "_projections.json")
STEM_FILE_SUFFIXES = (".csv", ".csv.provenance.json", *EMBEDDING_FILE_SUFFIXES)
STEM_IMAGES_SUFFIX = "_images"
# Feature manifest that local builds used to leave beside the dataset CSV; removed on delete.
LEGACY_MANIFEST_SUFFIX = "_input.csv"
# Stems whose own files carry these names would sit inside another stem's files (S_projections.json,
# S_input.csv), so S and S<suffix> are never both handed out.
OVERLAPPING_STEM_SUFFIXES = ("_projections", "_input")
IMAGE_PATH_COLUMNS = ("image_path", "thumbnail_image")

FINISHED_STATUSES = frozenset(db.FINISHED_BUILD_STATUSES)
INTERRUPTED_MESSAGE = "Interrupted: the API restarted while this build was running. Start it again."

_jobs: dict[str, dict] = {}
_job_keys: dict[str, tuple[str, str]] = {}  # job_id -> (kind, stem), the stored record's fixed fields
_jobs_lock = threading.Lock()
_build_lock = threading.Lock()
_reserve_lock = threading.Lock()
# Name and job id of the build holding the single build slot, so a rejected concurrent request
# can say which build is running and a reloaded page can find it. Empty while no build runs.
_active_build: dict[str, str] = {}


class BuildValidationError(ValueError):
    """A bad build request — reported synchronously, before any thread starts."""


class BuildInProgressError(RuntimeError):
    """Raised when a build is requested while another build is still running."""


def get_job(job_id: str) -> dict | None:
    """Return a snapshot of the job's current state, or None if job_id is unknown.

    Jobs started by an earlier API process are read from the build_jobs table. The returned
    dict is a copy, safe to read without holding _jobs_lock.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            return dict(job)
    return db.get_build_job(job_id)


def current_job() -> dict | None:
    """Return the build that holds the build slot, or None while no build is running.

    Returns
    -------
    dict or None
        The same snapshot as `get_job`, including ``job_id``, so a client that lost track
        of the build (a reloaded page) can resume polling it.
    """
    job_id = _active_build.get("job_id")
    job = get_job(job_id) if job_id else None
    if job is None or job["status"] in FINISHED_STATUSES:
        return None
    return job


def _new_job(name: str, kind: str, stem: str) -> str:
    job_id = uuid.uuid4().hex
    state = {
        "job_id": job_id,
        "kind": kind,
        "status": "queued",
        "percent": 0,
        "message": "Queued…",
        "name": name,
        "dataset": None,
        "error": None,
        "skipped": [],
    }
    with _jobs_lock:
        _jobs[job_id] = state
        _job_keys[job_id] = (kind, stem)
        db.save_build_job(job_id, kind, stem, state)
    _active_build["job_id"] = job_id
    return job_id


def recover_interrupted_builds() -> list[str]:
    """Close out builds that an earlier API process left running, and remove what they left behind.

    Called once at API start-up. Each stored build that never reached done or error is
    marked as an interrupted error (so a client polling it, or reading it later, is told
    what happened) and its partial output is removed. The AgriStress-500 download is kept
    so that starting the demo again resumes it. Upload scratch files are removed too.

    Returns
    -------
    list of str
        The ids of the builds marked as interrupted.
    """
    recovered = []
    for record in db.list_unfinished_build_jobs():
        if record["job_id"] in _jobs:  # running in this process
            continue
        keep_images = record["kind"] == f"demo:{AGRISTRESS_KEY}"
        _cleanup_partial(record["stem"], keep_images=keep_images)
        message = INTERRUPTED_MESSAGE
        if keep_images:
            message += " Files already downloaded are kept, so the download resumes."
        state = {**record["state"], "status": "error", "percent": None, "message": message, "error": message}
        db.save_build_job(record["job_id"], record["kind"], record["stem"], state)
        recovered.append(record["job_id"])
    if not _build_lock.locked():
        shutil.rmtree(Path(DATA_ROOT) / USER_DATA_REL / INCOMING_UPLOAD_DIRNAME, ignore_errors=True)
    return recovered


def _acquire_build_slot(name: str) -> None:
    """Claim the single build slot for `name`, or raise BuildInProgressError naming the running build."""
    if not _build_lock.acquire(blocking=False):
        running = _active_build.get("name")
        detail = f" ({running})" if running else ""
        raise BuildInProgressError(
            f"Another dataset build is already in progress{detail}. "
            "MagniData prepares one dataset at a time; try again once it finishes."
        )
    _active_build["name"] = name


def _release_build_slot() -> None:
    _active_build.clear()
    _build_lock.release()


def _update_job(job_id: str, **fields: Any) -> None:
    with _jobs_lock:
        job = _jobs[job_id]
        job.update(fields)
        kind, stem = _job_keys[job_id]
        db.save_build_job(job_id, kind, stem, job)


def _slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_").lower()
    return s or "dataset"


def _stem_base(stem: str) -> str:
    return os.path.join(DATA_ROOT, USER_DATA_REL, stem)


def _stem_occupied(stem: str, registered: set[str]) -> bool:
    base = _stem_base(stem)
    return (
        stem in DEMO_STEMS
        or f"{USER_DATA_REL}/{stem}.csv" in registered
        or any(os.path.lexists(base + suffix) for suffix in STEM_FILE_SUFFIXES)
        or os.path.lexists(base + STEM_IMAGES_SUFFIX)
    )


def _stem_in_use(stem: str, registered: set[str]) -> bool:
    related = [stem + suffix for suffix in OVERLAPPING_STEM_SUFFIXES]
    related += [stem.removesuffix(suffix) for suffix in OVERLAPPING_STEM_SUFFIXES if stem.endswith(suffix)]
    return any(_stem_occupied(s, registered) for s in (stem, *related))


def reserve_stem(base_slug: str) -> str:
    """Claim a free output stem under data_user/ and return it.

    The claim is the dataset CSV itself, created empty and exclusively, so it holds across
    threads and survives an API restart (an interrupted build's claim is released by
    `recover_interrupted_builds`). A stem is free only when no dataset is registered under
    it and none of the files and folders a dataset keeps beside its CSV exist; the demo
    stems are never handed out. ``S`` and ``S_projections`` / ``S_input`` count as one name,
    since the second's files would carry the first's sidecar names. The caller owns the CSV from here on and must either write
    it or remove it.

    Parameters
    ----------
    base_slug : str
        Preferred stem; ``_2``, ``_3``, … are appended until a free one is found.

    Returns
    -------
    str
        The reserved stem; its CSV is ``data_user/<stem>.csv``.
    """
    with _reserve_lock:
        registered = {r["source"] for r in db.list_created_datasets()}
        stem, n = base_slug, 2
        while True:
            if not _stem_in_use(stem, registered):
                with contextlib.suppress(FileExistsError), open(_stem_base(stem) + ".csv", "x"):
                    return stem
            stem = f"{base_slug}_{n}"
            n += 1


def _reserve_demo_stem(stem: str) -> None:
    """Claim a demo's fixed stem; the caller has already checked that no dataset is registered there."""
    with _reserve_lock, open(_stem_base(stem) + ".csv", "w"):
        pass


def _image_values(csv_path: str) -> list[str]:
    try:
        with open(csv_path, newline="") as f:
            return [row.get(column) or "" for row in csv.DictReader(f) for column in IMAGE_PATH_COLUMNS]
    except (OSError, UnicodeDecodeError, csv.Error):
        return []


def _remove_stem_files(stem: str, others: list[dict], other_image_values: list[str]) -> list[str]:
    source = f"{USER_DATA_REL}/{stem}.csv"
    images_prefix = f"{USER_DATA_REL}/{stem}{STEM_IMAGES_SUFFIX}/"
    kept_suffixes = list(EMBEDDING_FILE_SUFFIXES) if any(r["emb_source"] == source for r in others) else []
    if any(images_prefix in value for value in other_image_values):
        kept_suffixes.append(STEM_IMAGES_SUFFIX)
    manifest = f"{USER_DATA_REL}/{stem}{LEGACY_MANIFEST_SUFFIX}"
    if all(r["source"] != manifest for r in others):
        with contextlib.suppress(OSError):
            os.remove(os.path.join(DATA_ROOT, manifest))
    base = _stem_base(stem)
    for suffix in STEM_FILE_SUFFIXES:
        if suffix not in kept_suffixes:
            with contextlib.suppress(OSError):
                os.remove(base + suffix)
    if STEM_IMAGES_SUFFIX not in kept_suffixes:
        shutil.rmtree(base + STEM_IMAGES_SUFFIX, ignore_errors=True)
    return [f"{USER_DATA_REL}/{stem}{suffix}" for suffix in kept_suffixes if os.path.lexists(base + suffix)]


def _referenced_stems(emb_source: str | None, image_values: list[str]) -> set[str]:
    pattern = re.compile(rf"(?:^|/){USER_DATA_REL}/([^/]+){STEM_IMAGES_SUFFIX}/")
    stems = {match[1] for value in image_values if (match := pattern.search(value))}
    if emb_source:
        folder, filename = os.path.split(emb_source)
        if folder == USER_DATA_REL and filename.endswith(".csv"):
            stems.add(filename.removesuffix(".csv"))
    return stems


def remove_dataset_files(source: str, emb_source: str | None = None) -> list[str]:
    """Delete what a dataset keeps under data_user/: its CSV and the files and folders named after its stem.

    Stems are only handed out while nothing exists under them (see `reserve_stem`), so
    these files belong to this dataset however it was made and whatever its lineage says
    now. Its embeddings (``<stem>.json``, its provenance and projections) are kept while
    another registered dataset still takes its embeddings from this one, and its images
    folder is kept while another registered dataset's rows still point into it. A leftover
    ``<stem>_input.csv`` manifest is removed too, unless it is another registered dataset's CSV.

    Files kept that way when an earlier dataset was deleted are removed here once this was
    the last dataset using them: the embeddings of `emb_source` and the images folders this
    dataset's rows point into, for each such stem that no longer has a dataset CSV.

    Parameters
    ----------
    source : str
        The dataset's CSV path relative to DATA_ROOT; must be directly under data_user/.
        The dataset must already be unregistered.
    emb_source : str, optional
        The dataset path whose embeddings this dataset used.

    Returns
    -------
    list of str
        Paths relative to DATA_ROOT that were kept because other datasets still use them.
    """
    folder, filename = os.path.split(source)
    stem = filename.removesuffix(".csv")
    if folder != USER_DATA_REL or stem == filename or stem in ("", ".", ".."):
        raise ValueError(f"not a dataset CSV directly under {USER_DATA_REL}/: {source!r}")
    others = db.list_created_datasets()
    registered = {r["source"] for r in others}
    other_image_values = [v for r in others for v in _image_values(os.path.join(DATA_ROOT, r["source"]))]
    referenced = _referenced_stems(emb_source, _image_values(os.path.join(DATA_ROOT, source)))
    kept = _remove_stem_files(stem, others, other_image_values)
    for orphan in sorted(referenced - {stem}):
        orphan_csv = f"{USER_DATA_REL}/{orphan}.csv"
        if orphan_csv not in registered and not os.path.lexists(os.path.join(DATA_ROOT, orphan_csv)):
            kept += _remove_stem_files(orphan, others, other_image_values)
    return kept


def _count_csv_rows(path: str) -> int:
    with open(path, newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _check_embeddings_match_csv(feature_csv: Path, embeddings_json: Path) -> None:
    """Verify the shipped embeddings JSON's basenames match the built CSV's images.

    COCO128's basenames are fixed (same 128 Ultralytics files every build), but this
    is a cheap cross-check against the shipped embeddings asset drifting out of sync
    with a future COCO128 source change — raise rather than silently registering a
    dataset whose embeddings don't actually cover (or don't match) its images.
    """
    with open(feature_csv, newline="") as f:
        image_names = {os.path.basename(r["image_path"]) for r in csv.DictReader(f)}
    with open(embeddings_json) as f:
        emb_names = set(json.load(f).get("embeddings", {}))
    if image_names != emb_names:
        raise RuntimeError(
            "shipped COCO128 embeddings do not match the built dataset's images "
            f"(missing={sorted(image_names - emb_names)[:5]}, "
            f"extra={sorted(emb_names - image_names)[:5]})"
        )


def _dataset_meta(rec: dict) -> dict:
    return {
        "id": rec["id"],
        "name": rec["name"],
        "description": rec["description"],
        "source": rec["source"],
        "parent_source": rec["parent_source"],
        "row_count": rec["row_count"],
        "deletable": True,
        "has_embeddings": bool(rec["emb_source"]),
    }


def _cleanup_partial(stem: str, keep_images: bool = False) -> None:
    base = _stem_base(stem)
    for suffix in STEM_FILE_SUFFIXES:
        with contextlib.suppress(OSError):
            os.remove(base + suffix)
    if not keep_images:
        shutil.rmtree(base + STEM_IMAGES_SUFFIX, ignore_errors=True)


def _start_worker(job_id: str, target: Callable[[], None]) -> None:
    """Run ``target`` in a background thread that frees the build slot when it ends."""

    def worker() -> None:
        try:
            target()
        finally:
            _release_build_slot()

    threading.Thread(target=worker, daemon=True, name=f"dataset-build-{job_id[:8]}").start()


def _claim(name: str, kind: str, stem: str | None, base_slug: str = "") -> tuple[str, str]:
    """Take the build slot, reserve the output stem and record the job; return (job_id, stem).

    `stem` is a demo's fixed stem; otherwise a free stem derived from `base_slug` is reserved.
    Everything taken is given back if a later step fails.
    """
    _acquire_build_slot(name)
    try:
        if stem is None:
            stem = reserve_stem(base_slug)
        else:
            _reserve_demo_stem(stem)
        try:
            return _new_job(name, kind, stem), stem
        except Exception:
            _cleanup_partial(stem, keep_images=True)
            raise
    except Exception:
        _release_build_slot()
        raise


def _uploaded_size(upload: bytes | staging_mod.UploadedImage) -> int:
    if isinstance(upload, bytes):
        return len(upload)
    if upload.data is not None:
        return len(upload.data)
    if upload.source_path:
        try:
            return os.path.getsize(upload.source_path)
        except OSError as exc:
            raise BuildValidationError(f"uploaded file is unreadable: {upload.relative_path}") from exc
    raise BuildValidationError(f"uploaded file is missing content: {upload.relative_path}")


def _read_uploaded_bytes(upload: bytes | staging_mod.UploadedImage) -> bytes:
    if isinstance(upload, bytes):
        return upload
    if upload.data is not None:
        return upload.data
    if upload.source_path:
        try:
            with open(upload.source_path, "rb") as f:
                return f.read()
        except OSError as exc:
            raise BuildValidationError(f"uploaded file is unreadable: {upload.relative_path}") from exc
    raise BuildValidationError(f"uploaded file is missing content: {upload.relative_path}")


def _write_uploaded_file(upload: bytes | staging_mod.UploadedImage, dest: str) -> None:
    if isinstance(upload, bytes):
        with open(dest, "wb") as f:
            f.write(upload)
        return
    if upload.data is not None:
        with open(dest, "wb") as f:
            f.write(upload.data)
        return
    if upload.source_path:
        shutil.copyfile(upload.source_path, dest)
        return
    raise RuntimeError(f"uploaded file is missing content: {upload.relative_path}")


def _write_remapped_embeddings(
    upload: bytes | staging_mod.UploadedImage, dest: str, staged_names: dict[str, str]
) -> None:
    """Write the uploaded embeddings re-keyed to the staged image names; refuse a file that matches no image."""
    document = json.loads(_read_uploaded_bytes(upload))
    remapped = staging_mod.remap_embeddings(document["embeddings"], staged_names)
    if not remapped:
        raise BuildValidationError(
            "none of the names in the embeddings file match the uploaded images; key each vector by its image file name"
        )
    with open(dest, "w") as handle:
        json.dump({"embeddings": remapped}, handle)


def _cleanup_upload_sources(*groups: Any) -> None:
    incoming_root = (Path(DATA_ROOT) / USER_DATA_REL / INCOMING_UPLOAD_DIRNAME).resolve()
    cleanup_roots: set[Path] = set()
    for group in groups:
        items = group if isinstance(group, list) else [group]
        for item in items:
            if item is None or isinstance(item, bytes):
                continue
            source_path = getattr(item, "source_path", None)
            if not source_path:
                continue
            with contextlib.suppress(OSError, ValueError):
                source = Path(source_path).resolve()
                rel = source.relative_to(incoming_root)
                if rel.parts:
                    cleanup_roots.add(incoming_root / rel.parts[0])
    for root in cleanup_roots:
        shutil.rmtree(root, ignore_errors=True)


# ── Upload build ──────────────────────────────────────────────────────────────


def _validate_embeddings_file(data: bytes | staging_mod.UploadedImage) -> None:
    if _uploaded_size(data) > MAX_EMBEDDINGS_BYTES:
        raise BuildValidationError(f"embeddings file too large (max {MAX_EMBEDDINGS_BYTES // (1024 * 1024)}MB)")
    try:
        doc = json.loads(_read_uploaded_bytes(data))
    except json.JSONDecodeError as exc:
        raise BuildValidationError(f"embeddings file is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("embeddings"), dict):
        raise BuildValidationError('embeddings file must be JSON shaped {"embeddings": {"<basename>": [floats...]}}')


def start_upload_build(
    images: list,
    annotations: list,
    embeddings_file: bytes | staging_mod.UploadedImage | None,
    name: str,
    description: str,
) -> str:
    """Validate synchronously and start the build in a background thread.

    Returns a job_id immediately; raises BuildValidationError / BuildInProgressError
    without starting anything on a bad request.

    `images` is a list of staging_mod.UploadedImage (required, at least one).
    `annotations` is a list of staging_mod.UploadedImage (optional COCO-format JSON
    label files, may be empty) — each one whose original filename stem matches an
    image's gets real instance/coverage/class features instead of just quality/exposure
    features. `embeddings_file` is an optional pre-computed embeddings JSON (bring your
    own — this pipeline never computes embeddings itself); when given, the built
    dataset has embeddings, otherwise it doesn't.
    """
    if not name.strip():
        raise BuildValidationError("name is required")
    if not images:
        raise BuildValidationError("at least one image is required")
    if len(images) > MAX_IMAGES:
        raise BuildValidationError(f"too many images (max {MAX_IMAGES})")
    total_bytes = sum(_uploaded_size(i) for i in images) + sum(_uploaded_size(a) for a in annotations)
    if total_bytes > MAX_TOTAL_BYTES:
        raise BuildValidationError(f"upload too large (max {MAX_TOTAL_BYTES // (1024 * 1024)}MB)")
    if embeddings_file is not None:
        _validate_embeddings_file(embeddings_file)
    name, description = name.strip(), description.strip()
    job_id, stem = _claim(name, "upload", None, _slugify(name))

    def work() -> None:
        try:
            _run_upload_build(job_id, images, annotations, embeddings_file, name, description, stem)
        finally:
            _cleanup_upload_sources(images, annotations, embeddings_file)

    _start_worker(job_id, work)
    return job_id


def _precompute_projections(job_id: str, feature_csv_rel: str, emb_json_rel: str) -> None:
    """Reduce a freshly built dataset's embeddings to 3D and store them beside the CSV.

    Doing this here rather than on first view is the whole point of the sidecar: the 3D
    view then renders stored points instead of waiting on a reduction. A failure is
    logged into the job and otherwise ignored — the dataset is still perfectly usable,
    the viewer just falls back to computing the projection on demand.
    """
    _update_job(job_id, status="projecting", percent=92, message="Precomputing 3D projections…")
    csv_path = os.path.join(DATA_ROOT, feature_csv_rel)
    json_path = os.path.join(DATA_ROOT, emb_json_rel)
    try:
        matrix = embeddings_io.matrix_for_csv(csv_path, json_path)
        positions = projections.compute(matrix)
        if positions:
            projections.write(projections.sidecar_path(csv_path), matrix.shape[0], positions)
    except Exception as exc:  # a projection is a speed-up, never a reason to fail the build
        _update_job(job_id, message=f"Projections skipped: {exc}")


def _run_upload_build(
    job_id: str,
    images: list,
    annotations: list,
    embeddings_file: bytes | staging_mod.UploadedImage | None,
    name: str,
    description: str,
    stem: str,
) -> None:
    try:
        _update_job(job_id, status="staging", percent=15, message="Staging images…")
        stage_root = f"{USER_DATA_REL}/{stem}_images"
        staged_names: dict[str, str] = {}
        manifest_path = staging_mod.stage_images(images, stage_root, annotations=annotations, staged_names=staged_names)

        _update_job(job_id, status="extracting_features", percent=60, message="Extracting features…")
        feature_csv_rel = f"{USER_DATA_REL}/{stem}.csv"
        report = features_mod.extract(
            manifest_path,
            feature_csv_rel,
            device="cpu",
            skip_nima=True,
            image_mode="fullres",
            workers=_feature_workers(),
            drop_unreadable=True,
        )
        original_names = {staged: original for original, staged in staged_names.items()}
        skipped = _skipped_files(report, len(images), lambda path: original_names.get(os.path.basename(path), path))
        for path, _reason in report.unreadable:
            with contextlib.suppress(OSError):
                os.remove(path)

        emb_json_rel = None
        if embeddings_file is not None:
            emb_json_rel = f"{USER_DATA_REL}/{stem}.json"
            _write_remapped_embeddings(embeddings_file, emb_json_rel, staged_names)

        if emb_json_rel is not None:
            _precompute_projections(job_id, feature_csv_rel, emb_json_rel)

        _update_job(job_id, status="registering", percent=95, message="Registering dataset…")
        row_count = _count_csv_rows(feature_csv_rel)
        rec = db.create_dataset_record(name, description, feature_csv_rel, None, emb_json_rel, row_count)

        _finish(job_id, rec, skipped)
    except Exception as exc:
        _cleanup_partial(stem)
        _update_job(job_id, status="error", percent=None, message=str(exc), error=str(exc))


def _skipped_files(
    report: features_mod.FeatureReport, total: int, display_name: Callable[[str], str]
) -> list[dict[str, str]]:
    """Return ``[{file, reason}]`` for the images feature extraction could not read.

    Raises when none of them could be read, since an empty dataset is no use to anyone.
    """
    skipped = [{"file": display_name(path), "reason": reason} for path, reason in report.unreadable]
    if total and len(skipped) >= total:
        example = f" (for example {skipped[0]['file']}: {skipped[0]['reason']})" if skipped else ""
        raise RuntimeError(f"none of the {total} images could be read as an image{example}")
    return skipped


def _finish(job_id: str, rec: dict, skipped: list[dict[str, str]] | None = None) -> None:
    """Mark a build done, saying how many files were left out because they could not be read."""
    skipped = skipped or []
    message = "Done"
    if skipped:
        noun = "file" if len(skipped) == 1 else "files"
        message = f"Done — {len(skipped)} unreadable {noun} left out"
    _update_job(job_id, status="done", percent=100, message=message, dataset=_dataset_meta(rec), skipped=skipped)


# ── Local image_sets build ────────────────────────────────────────────────────


def _image_sets_root() -> Path:
    return Path(DATA_ROOT) / IMAGE_SETS_REL


def _local_images(folder_path: Path) -> list[Path]:
    """List every image under `folder_path`/images/, recursively, in a stable order."""
    images_dir = folder_path / "images"
    if not images_dir.is_dir():
        return []
    return sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in LOCAL_IMAGE_SUFFIXES)


def _local_label_count(folder_path: Path) -> int:
    labels_dir = folder_path / "labels"
    if not labels_dir.is_dir():
        return 0
    return sum(1 for p in labels_dir.rglob("*.json") if p.is_file())


def _local_stem(folder: str) -> str:
    """Dataset stem for a folder — derived from the folder name so a rebuild is detectable."""
    return _slugify(folder)


def list_local_folders() -> list[dict]:
    """Index the server-side image folders that a dataset can be built from.

    Returns
    -------
    list of dict
        One entry per directory directly under image_sets/ that contains a non-empty
        ``images/`` subdirectory, with its image count, its COCO label count, and
        whether a dataset has already been built from it. Directories without images
        are skipped rather than reported as errors.
    """
    root = _image_sets_root()
    if not root.is_dir():
        return []
    built_sources = {r["source"] for r in db.list_created_datasets()}
    folders = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        images = _local_images(entry)
        if not images:
            continue
        folders.append(
            {
                "folder": entry.name,
                "image_count": len(images),
                "label_count": _local_label_count(entry),
                "built": f"{USER_DATA_REL}/{_local_stem(entry.name)}.csv" in built_sources,
            }
        )
    return folders


def _resolve_local_folder(folder: str) -> Path:
    """Resolve `folder` to a directory directly under image_sets/, refusing anything else.

    Rejects separators, ``..`` and absolute paths up front, then confirms the resolved
    path is still a direct child of image_sets/ so a symlink cannot escape the root.
    """
    candidate = folder.strip()
    if not candidate or candidate in (".", "..") or "/" in candidate or "\\" in candidate or os.path.isabs(candidate):
        raise BuildValidationError("folder must be a single folder name directly under image_sets/")
    root = _image_sets_root()
    resolved = (root / candidate).resolve()
    if resolved.parent != root.resolve():
        raise BuildValidationError("folder must be a single folder name directly under image_sets/")
    if not resolved.is_dir():
        raise BuildValidationError(f"image set {candidate!r} not found under image_sets/")
    return resolved


def _write_local_manifest(images: list[Path], images_dir: Path, folder: str, manifest_path: str) -> None:
    """Write a feature-tool manifest whose rows point at the images where they already live."""
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "cluster", "cluster_l2"])
        writer.writeheader()
        for image in images:
            rel_within = image.relative_to(images_dir).as_posix()
            cluster, cluster_l2 = staging_mod.cluster_for(rel_within)
            writer.writerow(
                {
                    "image_path": f"{IMAGE_SETS_REL}/{folder}/images/{rel_within}",
                    "cluster": cluster,
                    "cluster_l2": cluster_l2,
                }
            )


def start_local_build(folder: str, name: str = "", description: str = "") -> str:
    """Validate synchronously and build a dataset from image_sets/<folder> in place.

    Unlike the upload path this never copies image data and never writes into the source
    folder, so it carries no image-count or byte cap — the folder can be arbitrarily
    large. The feature tool finds each image's COCO labels itself, at
    ``image_sets/<folder>/labels/<stem>.json``.

    Parameters
    ----------
    folder : str
        A single folder name directly under image_sets/, e.g. ``"MMDE-POC"``.
    name : str, default ""
        Dataset name; falls back to the folder name when blank.
    description : str, default ""
        Optional free-text description.

    Returns
    -------
    str
        The job id to poll; raises BuildValidationError / BuildInProgressError without
        starting anything on a bad request.
    """
    folder_path = _resolve_local_folder(folder)
    folder_name = folder_path.name
    images = _local_images(folder_path)
    if not images:
        raise BuildValidationError(f"image set {folder_name!r} has no images under images/")

    dataset_name = name.strip() or folder_name
    description = description.strip()
    job_id, stem = _claim(dataset_name, "local", None, _local_stem(folder_name))
    _start_worker(
        job_id, lambda: _run_local_build(job_id, images, folder_path, folder_name, dataset_name, description, stem)
    )
    return job_id


def _run_local_build(
    job_id: str,
    images: list[Path],
    folder_path: Path,
    folder_name: str,
    name: str,
    description: str,
    stem: str,
) -> None:
    # The manifest is scratch: it goes in the upload scratch area, never next to dataset CSVs,
    # where its name could belong to another dataset.
    scratch_rel = f"{USER_DATA_REL}/{INCOMING_UPLOAD_DIRNAME}/build-{job_id}"
    manifest_rel = f"{scratch_rel}/manifest.csv"
    prefix = f"{IMAGE_SETS_REL}/{folder_name}/"
    try:
        _update_job(job_id, status="staging", percent=10, message=f"Indexing {len(images)} images…")
        os.makedirs(os.path.join(DATA_ROOT, scratch_rel), exist_ok=True)
        _write_local_manifest(images, folder_path / "images", folder_name, manifest_rel)

        _update_job(job_id, status="extracting_features", percent=30, message="Extracting features…")
        feature_csv_rel = f"{USER_DATA_REL}/{stem}.csv"
        report = features_mod.extract(
            manifest_rel,
            feature_csv_rel,
            device="cpu",
            skip_nima=True,
            image_mode="fullres",
            workers=_feature_workers(),
            drop_unreadable=True,
        )
        skipped = _skipped_files(report, len(images), lambda path: path.removeprefix(prefix))

        _update_job(job_id, status="registering", percent=95, message="Registering dataset…")
        row_count = _count_csv_rows(feature_csv_rel)
        rec = db.create_dataset_record(name, description, feature_csv_rel, None, None, row_count)

        _finish(job_id, rec, skipped)
    except Exception as exc:
        # Only the derived artifacts are removed — image_sets/ is the user's own data.
        _cleanup_partial(stem)
        _update_job(job_id, status="error", percent=None, message=str(exc), error=str(exc))
    finally:
        shutil.rmtree(os.path.join(DATA_ROOT, scratch_rel), ignore_errors=True)


# ── Demo build ────────────────────────────────────────────────────────────────


def start_demo_build(demo_key: str) -> str:
    """Build one of the one-click demo datasets, using the same contract as start_upload_build.

    Any unknown key, or a disabled registry entry, raises BuildValidationError. Neither
    demo computes embeddings at build time (this pipeline never does) — COCO128
    attaches a DINOv2 embeddings file precomputed offline and shipped with the tool
    (see COCO128_EMBEDDINGS_ASSET); AgriStress-500 downloads its pre-computed SEED
    embeddings from the CDN.
    """
    demo = next((d for d in DEMOS if d["key"] == demo_key), None)
    if demo is None or not demo["enabled"]:
        raise BuildValidationError(f"unknown or disabled demo {demo_key!r}")
    if demo_key == COCO128_KEY:
        stem, worker_fn = COCO128_STEM, _run_demo_build
    elif demo_key == AGRISTRESS_KEY:
        stem, worker_fn = AGRISTRESS_STEM, _run_agristress_build
    else:
        # Unreachable today (every enabled registry entry maps to an implementation
        # above) but kept explicit so a future enabled-but-unimplemented entry fails
        # loudly rather than silently building the wrong thing.
        raise BuildValidationError(f"demo {demo_key!r} is not buildable yet")

    source = f"{USER_DATA_REL}/{stem}.csv"
    existing = db.get_created_dataset_by_source(source)
    if existing is not None:
        if os.path.isfile(os.path.join(DATA_ROOT, source)):
            raise BuildValidationError(f"the {demo['name']} demo dataset has already been built")
        # The records live in a persistent volume while data_user/ is a bind mount, so a fresh
        # checkout can keep a record whose CSV is gone. /api/datasets already hides such records.
        db.delete_created_dataset(existing["id"])
    job_id, _stem = _claim(demo["name"], f"demo:{demo_key}", stem)
    _start_worker(job_id, lambda: worker_fn(job_id))
    return job_id


def _run_demo_build(job_id: str) -> None:
    stem = COCO128_STEM
    try:
        _update_job(job_id, status="staging", percent=5, message="Downloading COCO128…")
        root = Path(DATA_ROOT).resolve()
        cache_dir = root / USER_DATA_REL / "_downloads"
        # Named after the actual URL so a source change (e.g. box-only coco128 ->
        # coco128-seg) can never silently reuse a stale cached zip from before.
        zip_name = Path(urllib.parse.urlparse(coco128.COCO128_URL).path).name
        zip_path = cache_dir / zip_name
        extract_dir = cache_dir / "extracted"
        coco128.download(coco128.COCO128_URL, zip_path)
        coco_root = coco128.extract(zip_path, extract_dir)

        _update_job(job_id, status="staging", percent=25, message="Staging COCO128 images…")
        args = argparse.Namespace(
            dataset_stem=stem,
            limit=None,
            stage_mode="copy",
            cluster_by="dominant-class",
            device="cpu",
            with_nima=False,
            no_embeddings=True,
            embedding_model="none",
        )
        stage_root = root / USER_DATA_REL / f"{stem}_images"
        manifest, _stats = coco128.stage_coco128(args, root, coco_root, stage_root=stage_root)

        _update_job(job_id, status="extracting_features", percent=65, message="Extracting features…")
        data_dir = root / USER_DATA_REL
        feature_csv, _embeddings_json = coco128.run_pipeline(args, root, manifest, data_dir=data_dir)

        _update_job(job_id, status="staging", percent=90, message="Attaching precomputed embeddings…")
        embeddings_json = data_dir / f"{stem}.json"
        shutil.copyfile(COCO128_EMBEDDINGS_ASSET, embeddings_json)
        _check_embeddings_match_csv(feature_csv, embeddings_json)

        feature_csv_rel = f"{USER_DATA_REL}/{stem}.csv"
        emb_rel = f"{USER_DATA_REL}/{stem}.json"
        _precompute_projections(job_id, feature_csv_rel, emb_rel)

        _update_job(job_id, status="registering", percent=95, message="Registering dataset…")
        row_count = _count_csv_rows(str(feature_csv))
        rec = db.create_dataset_record(COCO128_NAME, COCO128_DESCRIPTION, feature_csv_rel, None, emb_rel, row_count)

        _finish(job_id, rec)
    except Exception as exc:
        _cleanup_partial(stem)
        _update_job(job_id, status="error", percent=None, message=str(exc), error=str(exc))


# ── AgriStress-500 demo build ────────────────────────────────────────────────────


def _urlopen(url: str, timeout: float) -> Any:
    """Open a URL after checking its scheme, refusing anything but http(s).

    Every remote download in this module (the Hugging Face API listing, the per-file
    Hugging Face downloads, and the CDN embeddings download) funnels through this one
    guarded call point rather than calling urlopen() directly.
    """
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in ("http", "https"):
        raise ValueError(f"refusing to open URL with disallowed scheme {scheme!r}: {url}")
    return urllib.request.urlopen(url, timeout=timeout)


FEATURE_WORKER_BYTES = int(1.25 * 1024**3)  # one worker peaks near 0.9 GB on a 6016x4016 image, plus headroom
FEATURE_MEMORY_RESERVE_BYTES = 4 * 1024**3  # left for the OS, Docker and the API; WSL dies when this is exhausted
FEATURE_MAX_WORKERS = 6


def _available_memory_bytes() -> int | None:
    """Return the memory the OS reports as available, or None where it cannot be read."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def _feature_workers() -> int:
    """Return how many processes feature extraction may use.

    Bounded by CPU count (one core stays free for the API), a fixed cap, and available memory
    minus a reserve, because full-resolution images are decoded to float arrays and too many
    workers exhaust RAM (and take WSL down with it).
    """
    by_cpu = (os.cpu_count() or 2) - 1
    available = _available_memory_bytes()
    by_memory = (available - FEATURE_MEMORY_RESERVE_BYTES) // FEATURE_WORKER_BYTES if available is not None else 1
    return max(1, min(by_cpu, FEATURE_MAX_WORKERS, by_memory))


def _agristress_file_pairs() -> list[tuple[str, str, str]]:
    """List (basename, image_rel, mask_rel) for every image with a matching mask.

    Both live under images/<label>/ and masks/<label>/ in the Hugging Face repo with
    the same basename. One API call lists the whole repo tree (~4950 entries at last
    count, including the unrelated per-instance crops under instances/, which are
    ignored).
    """
    with _urlopen(AGRISTRESS_HF_API_URL, timeout=30) as resp:
        info = json.loads(resp.read())
    images_by_basename: dict[str, str] = {}
    masks_by_basename: dict[str, str] = {}
    for sib in info.get("siblings", []):
        rel = sib.get("rfilename", "")
        if rel.startswith("images/"):
            images_by_basename[os.path.basename(rel)] = rel
        elif rel.startswith("masks/"):
            masks_by_basename[os.path.basename(rel)] = rel
    basenames = sorted(set(images_by_basename) & set(masks_by_basename))
    return [(b, images_by_basename[b], masks_by_basename[b]) for b in basenames]


def _fetch_to_file(url: str, dest: Path) -> None:
    """Stream a URL to ``dest`` via a temp file, so an interrupted download is never mistaken for a finished one."""
    tmp = dest.with_name(dest.name + ".part")
    with _urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f, length=1024 * 1024)
    tmp.replace(dest)


def _download_agristress_pair(
    images_dir: Path, masks_dir: Path, thumbs_dir: Path, stem: str, image_rel: str, mask_rel: str, basename: str
) -> dict[str, str]:
    """Download one image/mask pair from Hugging Face and stage a local thumbnail.

    Files already staged by an earlier attempt are reused. Returns the manifest row
    (image_path, thumbnail_image, cluster, cluster_l2) for this pair. The thumbnail is
    generated locally since AgriStress-500's originals run tens of MB each (6016x4016).
    """
    label = image_rel.split("/")[1]  # images/<label>/<basename>
    image_file, mask_file = images_dir / basename, masks_dir / basename
    thumb_name = os.path.splitext(basename)[0] + ".jpg"
    thumb_file = thumbs_dir / thumb_name

    if not image_file.exists():
        _fetch_to_file(f"{AGRISTRESS_HF_RESOLVE_BASE}/{image_rel}", image_file)
    if not mask_file.exists():
        _fetch_to_file(f"{AGRISTRESS_HF_RESOLVE_BASE}/{mask_rel}", mask_file)
    if not thumb_file.exists():
        with Image.open(image_file) as source:
            thumb = source.convert("RGB")
        thumb.thumbnail((AGRISTRESS_THUMB_MAX, AGRISTRESS_THUMB_MAX), Image.Resampling.LANCZOS)
        thumb.save(thumb_file, "JPEG", quality=85)

    return {
        "image_path": f"{USER_DATA_REL}/{stem}_images/images/{basename}",
        "thumbnail_image": f"{USER_DATA_REL}/{stem}_images/thumbnails/{thumb_name}",
        "cluster": label,
        "cluster_l2": label,
    }


def _reattach_thumbnails(feature_csv_abs: str, manifest_rows: list[dict[str, str]]) -> None:
    """Re-attach the thumbnail_image column that features.run() drops.

    features.run() only passes a manifest's cluster/cluster_l2 through to the output
    (see tools/schema.py's CLUSTER list) — re-attach thumbnail_image by row index,
    since row order is preserved 1:1 between the manifest and the feature CSV it
    writes.
    """
    with open(feature_csv_abs, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [*(reader.fieldnames or []), "thumbnail_image"]
        feature_rows = list(reader)
    for row, manifest_row in zip(feature_rows, manifest_rows, strict=True):
        row["thumbnail_image"] = manifest_row["thumbnail_image"]
    with open(feature_csv_abs, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(feature_rows)


def _run_agristress_build(job_id: str) -> None:
    """Build the AgriStress-500 demo dataset from Hugging Face and the CDN.

    AgriStress-500's images and masks are downloaded from Hugging Face and processed
    entirely locally: thumbnails are generated here, and tools.features.run() computes
    the real feature CSV from the downloaded pixels (this pipeline never computes
    embeddings itself, per the module docstring — those still come from the CDN's
    pre-computed SEED-Embeddings.json, brought in as-is). Masks are staged as a
    masks/<basename> sibling to images/<basename>, which /api/annotation/mask +
    /api/annotation/overlay already know how to find and composite (see app.py's
    _sibling_raster_mask_bytes) — no extra wiring needed for them to show up.
    """
    stem = AGRISTRESS_STEM
    try:
        _update_job(job_id, status="staging", percent=2, message="Listing AgriStress-500 files…")
        pairs = _agristress_file_pairs()
        if not pairs:
            raise RuntimeError("no matching image/mask pairs found in the AgriStress-500 repo")

        root = Path(DATA_ROOT).resolve()
        stage_root = root / USER_DATA_REL / f"{stem}_images"
        images_dir, masks_dir, thumbs_dir = (stage_root / d for d in ("images", "masks", "thumbnails"))
        for d in (images_dir, masks_dir, thumbs_dir):
            d.mkdir(parents=True, exist_ok=True)

        total = len(pairs)
        with ThreadPoolExecutor(max_workers=AGRISTRESS_DOWNLOAD_WORKERS) as pool:
            futures = [
                pool.submit(
                    _download_agristress_pair, images_dir, masks_dir, thumbs_dir, stem, image_rel, mask_rel, basename
                )
                for basename, image_rel, mask_rel in pairs
            ]
            manifest_rows = []
            for i, future in enumerate(futures):
                manifest_rows.append(future.result())
                pct = 5 + int(55 * (i + 1) / total)
                _update_job(
                    job_id, status="staging", percent=pct, message=f"Downloading AgriStress-500 ({i + 1}/{total})…"
                )

        manifest_path = stage_root / f"{stem}_input.csv"
        with open(manifest_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["image_path", "thumbnail_image", "cluster", "cluster_l2"])
            w.writeheader()
            w.writerows(manifest_rows)

        _update_job(job_id, status="extracting_features", percent=65, message="Extracting features locally…")
        feature_csv_rel = f"{USER_DATA_REL}/{stem}.csv"
        features_mod.run(
            str(manifest_path.relative_to(root)),
            feature_csv_rel,
            device="cpu",
            skip_nima=True,
            image_mode="fullres",
            workers=_feature_workers(),
        )
        _reattach_thumbnails(os.path.join(DATA_ROOT, feature_csv_rel), manifest_rows)

        _update_job(job_id, status="staging", percent=90, message="Downloading SEED embeddings…")
        emb_rel = f"{USER_DATA_REL}/{stem}.json"
        with (
            _urlopen(AGRISTRESS_EMBEDDINGS_URL, timeout=60) as resp,
            open(os.path.join(DATA_ROOT, emb_rel), "wb") as f,
        ):
            f.write(resp.read())

        _precompute_projections(job_id, feature_csv_rel, emb_rel)

        _update_job(job_id, status="registering", percent=95, message="Registering dataset…")
        row_count = _count_csv_rows(feature_csv_rel)
        rec = db.create_dataset_record(
            AGRISTRESS_NAME, AGRISTRESS_DESCRIPTION, feature_csv_rel, None, emb_rel, row_count
        )

        _finish(job_id, rec)
    except Exception as exc:
        _cleanup_partial(stem, keep_images=True)
        _update_job(job_id, status="error", percent=None, message=str(exc), error=str(exc))
