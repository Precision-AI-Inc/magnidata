# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Orchestrates a "build a dataset from images" job: stage images -> extract features
-> register in the created_datasets table. Runs in a background thread; progress is
tracked in an in-process job dict polled by GET /api/datasets/build/<job_id>.

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

import csv
import io
import json
import os
import re
import shutil
import sys
import threading
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from types import SimpleNamespace

import db
from PIL import Image


# ── Make the precisionai.agriviz.{tools,scripts} package importable ─────────────
# This file's own location tells us where the package chain lives, regardless of
# the process's cwd or DATA_ROOT (which is purely a data-output location, below):
#   - Docker layout: this file is copied flat to /app/dataset_builder.py, and
#     /app/precisionai/... sits right next to it (0 levels up).
#   - Dev/checkout layout: this file lives at precisionai/agriviz/api/dataset_builder.py,
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

from precisionai.agriviz.scripts import prepare_coco128_dashboard as coco128  # noqa: E402
from precisionai.agriviz.tools import features as features_mod  # noqa: E402
from precisionai.agriviz.tools import staging as staging_mod  # noqa: E402

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

MAX_IMAGES = 500
MAX_TOTAL_BYTES = 200 * 1024 * 1024
MAX_EMBEDDINGS_BYTES = 100 * 1024 * 1024

COCO128_KEY = "coco128"
COCO128_STEM = "coco128"
COCO128_NAME = "COCO128"
COCO128_DESCRIPTION = "Ultralytics COCO128 sample, built in-app (embeddings precomputed offline and shipped with the tool)"

AGRISTRESS_KEY = "agristress500"
AGRISTRESS_STEM = "agristress500"
AGRISTRESS_NAME = "AgriStress-500"
AGRISTRESS_DESCRIPTION = ("451 field images + real segmentation masks, downloaded from "
                           "Hugging Face and processed locally (features computed here; "
                           "embeddings brought in from the public CDN)")
# Images + masks: Hugging Face (the full-resolution source; downloaded and processed
# locally — features are computed from these, not taken pre-built from anywhere).
AGRISTRESS_HF_API_URL = "https://huggingface.co/api/datasets/precisionaiinc/AgriStress-500"
AGRISTRESS_HF_RESOLVE_BASE = "https://huggingface.co/datasets/precisionaiinc/AgriStress-500/resolve/main"
# Embeddings: still the CDN's pre-computed SEED-Embeddings.json — this pipeline never
# computes embeddings itself (see the module docstring), so these are brought in as-is.
AGRISTRESS_EMBEDDINGS_URL = "https://d379glmvb7evte.cloudfront.net/internal/data/embeddings/SEED-Embeddings.json"
AGRISTRESS_THUMB_MAX = 720

# Static demo registry (GET /api/datasets/demos serves this verbatim).
DEMOS: list[dict] = [
    {"key": COCO128_KEY, "name": COCO128_NAME, "description": COCO128_DESCRIPTION, "enabled": True},
    {"key": AGRISTRESS_KEY, "name": AGRISTRESS_NAME, "description": AGRISTRESS_DESCRIPTION, "enabled": True},
]

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_build_lock = threading.Lock()


class BuildValidationError(ValueError):
    """A bad build request — reported synchronously, before any thread starts."""


class BuildInProgressError(RuntimeError):
    """Raised when a build is requested while another build is still running."""


def get_job(job_id: str) -> dict | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job is not None else None


def _new_job() -> str:
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {"status": "queued", "percent": 0, "message": "Queued…",
                          "dataset": None, "error": None}
    return job_id


def _update_job(job_id: str, **fields) -> None:
    with _jobs_lock:
        _jobs[job_id].update(fields)


def _slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_").lower()
    return s or "dataset"


def _unique_stem(base_slug: str) -> str:
    existing_sources = {r["source"] for r in db.list_created_datasets()}
    stem = base_slug
    n = 2
    while (f"{USER_DATA_REL}/{stem}.csv" in existing_sources
           or os.path.isfile(os.path.join(DATA_ROOT, USER_DATA_REL, f"{stem}.csv"))):
        stem = f"{base_slug}_{n}"
        n += 1
    return stem


def _count_csv_rows(path: str) -> int:
    with open(path, newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _check_embeddings_match_csv(feature_csv: Path, embeddings_json: Path) -> None:
    """COCO128's basenames are fixed (same 128 Ultralytics files every build), but this
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
        "id": rec["id"], "name": rec["name"], "description": rec["description"],
        "source": rec["source"], "parent_source": rec["parent_source"],
        "row_count": rec["row_count"], "deletable": True,
        "has_embeddings": bool(rec["emb_source"]),
    }


def _cleanup_partial(stem: str) -> None:
    base = os.path.join(DATA_ROOT, USER_DATA_REL, stem)
    for suffix in (".csv", ".csv.provenance.json", ".json", ".json.provenance.json"):
        try:
            os.remove(base + suffix)
        except OSError:
            pass
    shutil.rmtree(base + "_images", ignore_errors=True)


# ── Upload build ──────────────────────────────────────────────────────────────

def _validate_embeddings_file(data: bytes) -> None:
    if len(data) > MAX_EMBEDDINGS_BYTES:
        raise BuildValidationError(f"embeddings file too large (max {MAX_EMBEDDINGS_BYTES // (1024 * 1024)}MB)")
    try:
        doc = json.loads(data)
    except json.JSONDecodeError as exc:
        raise BuildValidationError(f"embeddings file is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("embeddings"), dict):
        raise BuildValidationError('embeddings file must be JSON shaped {"embeddings": {"<basename>": [floats...]}}')


def start_upload_build(images: list, annotations: list, embeddings_file: bytes | None,
                        name: str, description: str) -> str:
    """Validate synchronously and start the build in a background thread. Returns a
    job_id immediately; raises BuildValidationError / BuildInProgressError without
    starting anything on a bad request.

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
    total_bytes = sum(len(i.data) for i in images) + sum(len(a.data) for a in annotations)
    if total_bytes > MAX_TOTAL_BYTES:
        raise BuildValidationError(f"upload too large (max {MAX_TOTAL_BYTES // (1024 * 1024)}MB)")
    if embeddings_file is not None:
        _validate_embeddings_file(embeddings_file)
    if not _build_lock.acquire(blocking=False):
        raise BuildInProgressError("a dataset build is already in progress")

    job_id = _new_job()
    stem = _unique_stem(_slugify(name))
    name, description = name.strip(), description.strip()

    def worker():
        try:
            _run_upload_build(job_id, images, annotations, embeddings_file, name, description, stem)
        finally:
            _build_lock.release()

    threading.Thread(target=worker, daemon=True).start()
    return job_id


def _run_upload_build(job_id, images, annotations, embeddings_file, name, description, stem) -> None:
    try:
        _update_job(job_id, status="staging", percent=15, message="Staging images…")
        stage_root = f"{USER_DATA_REL}/{stem}_images"
        manifest_path = staging_mod.stage_images(images, stage_root, annotations=annotations)

        _update_job(job_id, status="extracting_features", percent=60, message="Extracting features…")
        feature_csv_rel = f"{USER_DATA_REL}/{stem}.csv"
        features_mod.run(manifest_path, feature_csv_rel, device="cpu", skip_nima=True,
                          image_mode="fullres")

        emb_json_rel = None
        if embeddings_file is not None:
            emb_json_rel = f"{USER_DATA_REL}/{stem}.json"
            with open(emb_json_rel, "wb") as f:
                f.write(embeddings_file)

        _update_job(job_id, status="registering", percent=95, message="Registering dataset…")
        row_count = _count_csv_rows(feature_csv_rel)
        rec = db.create_dataset_record(name, description, feature_csv_rel, None, emb_json_rel,
                                        row_count)

        _update_job(job_id, status="done", percent=100, message="Done", dataset=_dataset_meta(rec))
    except Exception as exc:
        _cleanup_partial(stem)
        _update_job(job_id, status="error", percent=None, message=str(exc), error=str(exc))


# ── Demo build ────────────────────────────────────────────────────────────────

def start_demo_build(demo_key: str) -> str:
    """Same contract as start_upload_build, for the one-click demo registry. Any
    unknown key, or a disabled registry entry, raises BuildValidationError. Neither
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

    existing = db.get_created_dataset_by_source(f"{USER_DATA_REL}/{stem}.csv")
    if existing is not None:
        raise BuildValidationError(f"the {demo['name']} demo dataset has already been built")
    if not _build_lock.acquire(blocking=False):
        raise BuildInProgressError("a dataset build is already in progress")

    job_id = _new_job()

    def worker():
        try:
            worker_fn(job_id)
        finally:
            _build_lock.release()

    threading.Thread(target=worker, daemon=True).start()
    return job_id


def _run_demo_build(job_id) -> None:
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
        args = SimpleNamespace(dataset_stem=stem, limit=None, stage_mode="copy",
                                cluster_by="dominant-class", device="cpu", with_nima=False,
                                no_embeddings=True, embedding_model="none")
        stage_root = root / USER_DATA_REL / f"{stem}_images"
        manifest, _stats = coco128.stage_coco128(args, root, coco_root, stage_root=stage_root)

        _update_job(job_id, status="extracting_features", percent=65, message="Extracting features…")
        data_dir = root / USER_DATA_REL
        feature_csv, _embeddings_json = coco128.run_pipeline(args, root, manifest, data_dir=data_dir)

        _update_job(job_id, status="staging", percent=90, message="Attaching precomputed embeddings…")
        embeddings_json = data_dir / f"{stem}.json"
        shutil.copyfile(COCO128_EMBEDDINGS_ASSET, embeddings_json)
        _check_embeddings_match_csv(feature_csv, embeddings_json)

        _update_job(job_id, status="registering", percent=95, message="Registering dataset…")
        row_count = _count_csv_rows(str(feature_csv))
        feature_csv_rel = f"{USER_DATA_REL}/{stem}.csv"
        emb_rel = f"{USER_DATA_REL}/{stem}.json"
        rec = db.create_dataset_record(COCO128_NAME, COCO128_DESCRIPTION, feature_csv_rel, None,
                                        emb_rel, row_count)

        _update_job(job_id, status="done", percent=100, message="Done", dataset=_dataset_meta(rec))
    except Exception as exc:
        _cleanup_partial(stem)
        _update_job(job_id, status="error", percent=None, message=str(exc), error=str(exc))


# ── AgriStress-500 demo build ────────────────────────────────────────────────────

def _agristress_file_pairs() -> list[tuple[str, str, str]]:
    """List (basename, image_rel, mask_rel) for every image that has a matching mask in
    the Hugging Face repo — both live under images/<label>/ and masks/<label>/ with the
    same basename. One API call lists the whole repo tree (~4950 entries at last count,
    including the unrelated per-instance crops under instances/, which are ignored).
    """
    with urllib.request.urlopen(AGRISTRESS_HF_API_URL, timeout=30) as resp:
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


def _run_agristress_build(job_id) -> None:
    """AgriStress-500's images and masks are downloaded from Hugging Face and processed
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

        manifest_rows = []
        total = len(pairs)
        for i, (basename, image_rel, mask_rel) in enumerate(pairs):
            pct = 5 + int(55 * i / total)
            _update_job(job_id, status="staging", percent=pct,
                        message=f"Downloading image {i + 1}/{total}…")
            label = image_rel.split("/")[1]   # images/<label>/<basename>

            with urllib.request.urlopen(f"{AGRISTRESS_HF_RESOLVE_BASE}/{image_rel}", timeout=60) as resp:
                image_bytes = resp.read()
            (images_dir / basename).write_bytes(image_bytes)

            with urllib.request.urlopen(f"{AGRISTRESS_HF_RESOLVE_BASE}/{mask_rel}", timeout=60) as resp:
                (masks_dir / basename).write_bytes(resp.read())

            # Locally computed thumbnail — the originals run tens of MB each (6016x4016).
            thumb = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            thumb.thumbnail((AGRISTRESS_THUMB_MAX, AGRISTRESS_THUMB_MAX), Image.LANCZOS)
            thumb_name = os.path.splitext(basename)[0] + ".jpg"
            thumb.save(thumbs_dir / thumb_name, "JPEG", quality=85)

            manifest_rows.append({
                "image_path": f"{USER_DATA_REL}/{stem}_images/images/{basename}",
                "thumbnail_image": f"{USER_DATA_REL}/{stem}_images/thumbnails/{thumb_name}",
                "cluster": label, "cluster_l2": label,
            })

        manifest_path = stage_root / f"{stem}_input.csv"
        with open(manifest_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["image_path", "thumbnail_image", "cluster", "cluster_l2"])
            w.writeheader()
            w.writerows(manifest_rows)

        _update_job(job_id, status="extracting_features", percent=65, message="Extracting features locally…")
        feature_csv_rel = f"{USER_DATA_REL}/{stem}.csv"
        features_mod.run(str(manifest_path.relative_to(root)), feature_csv_rel, device="cpu",
                          skip_nima=True, image_mode="fullres")

        # features.run() only passes a manifest's cluster/cluster_l2 through to the
        # output (see tools/schema.py's CLUSTER list) — re-attach thumbnail_image by
        # row index, since row order is preserved 1:1 between the manifest and the
        # feature CSV it writes.
        feature_csv_abs = os.path.join(DATA_ROOT, feature_csv_rel)
        with open(feature_csv_abs, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = [*(reader.fieldnames or []), "thumbnail_image"]
            feature_rows = list(reader)
        for row, manifest_row in zip(feature_rows, manifest_rows):
            row["thumbnail_image"] = manifest_row["thumbnail_image"]
        with open(feature_csv_abs, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(feature_rows)

        _update_job(job_id, status="staging", percent=90, message="Downloading SEED embeddings…")
        emb_rel = f"{USER_DATA_REL}/{stem}.json"
        with urllib.request.urlopen(AGRISTRESS_EMBEDDINGS_URL, timeout=60) as resp:
            with open(os.path.join(DATA_ROOT, emb_rel), "wb") as f:
                f.write(resp.read())

        _update_job(job_id, status="registering", percent=95, message="Registering dataset…")
        row_count = _count_csv_rows(feature_csv_rel)
        rec = db.create_dataset_record(AGRISTRESS_NAME, AGRISTRESS_DESCRIPTION, feature_csv_rel, None,
                                        emb_rel, row_count)

        _update_job(job_id, status="done", percent=100, message="Done", dataset=_dataset_meta(rec))
    except Exception as exc:
        _cleanup_partial(stem)
        _update_job(job_id, status="error", percent=None, message=str(exc), error=str(exc))
