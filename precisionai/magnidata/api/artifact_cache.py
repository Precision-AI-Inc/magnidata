# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Local cache of per-sample artifacts downloaded from MLflow.

Artifacts are immutable once logged, so a cached file never needs re-validation. The cache
lives under ``DATA_ROOT/data_user/.mlflow_cache`` (the one read-write volume) and is bounded:
least-recently-used run folders are evicted above ``MLFLOW_CACHE_MAX_MB``.
"""

import os
import re
import shutil
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path

from mlflow_store import MlflowNotFoundError, MlflowStore

from precisionai.magnidata.training import contract

DEFAULT_MAX_MB = 2048
CACHE_DIRNAME = ".mlflow_cache"
_lock = threading.Lock()
# MLflow run ids are 32 lowercase hex characters; anything else could name a folder outside the cache.
_RUN_ID = re.compile(r"[0-9a-f]{32}")


def cache_root() -> Path:
    """Return the cache folder, resolved from ``DATA_ROOT`` at call time."""
    return Path(os.environ.get("DATA_ROOT", ".")) / "data_user" / CACHE_DIRNAME


def _inside(base: Path, relative: str) -> Path:
    """Return ``base / relative`` resolved, refusing a path that would leave ``base``."""
    root = os.path.realpath(base)
    path = os.path.realpath(os.path.join(root, relative))
    if not path.startswith(root + os.sep):
        raise MlflowNotFoundError(f"artifact path outside the cache: {relative!r}")
    return Path(path)


def _run_dir(run_id: str) -> Path:
    """Return a run's cache folder; only real MLflow run ids are accepted, so it stays inside the cache."""
    if not _RUN_ID.fullmatch(run_id):
        raise MlflowNotFoundError(f"not an MLflow run id: {run_id!r}")
    return _inside(cache_root(), run_id)


def max_bytes() -> int:
    """Return the size bound in bytes (``MLFLOW_CACHE_MAX_MB``, default 2048)."""
    try:
        return int(float(os.environ.get("MLFLOW_CACHE_MAX_MB", DEFAULT_MAX_MB)) * 1024 * 1024)
    except ValueError:
        return DEFAULT_MAX_MB * 1024 * 1024


def _folder_size(folder: Path) -> int:
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())


def evict(root: Path, keep: str, limit: int) -> list[str]:
    """Delete least-recently-used run folders until the cache fits ``limit``; never ``keep``.

    Returns the evicted run ids.
    """
    folders = [d for d in root.iterdir() if d.is_dir() and d.name != keep]
    sizes = {d: _folder_size(d) for d in folders}
    kept = _inside(root, keep)
    total = sum(sizes.values()) + (_folder_size(kept) if kept.is_dir() else 0)
    evicted = []
    for folder in sorted(folders, key=lambda d: d.stat().st_mtime):
        if total <= limit:
            break
        shutil.rmtree(folder, ignore_errors=True)
        total -= sizes[folder]
        evicted.append(folder.name)
    return evicted


def _fetch(store: MlflowStore, run_id: str, artifact: str, run_dir: Path) -> Path | None:
    """Return the cached copy of ``artifact``, downloading it first if needed; None if the run lacks it."""
    target = _inside(run_dir, artifact)
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=run_dir) as tmp:
        try:
            downloaded = store.download_artifact(run_id, artifact, tmp)
        except MlflowNotFoundError:
            return None
        os.replace(downloaded, target)
    return target


def sync_per_sample(
    store: MlflowStore, run_id: str, epochs: Sequence[int] | None = None, with_split: bool = True
) -> list[tuple[int, Path]]:
    """Make the run's per-sample files available locally and return ``(epoch, path)`` pairs.

    Only epochs listed in the run's ``magnidata.scored_epochs`` tag (optionally narrowed to
    ``epochs``) that are not cached yet are downloaded. ``split.csv`` is fetched alongside
    unless ``with_split`` is false. Epochs whose file is missing in MLflow are left out.
    """
    scored = contract.parse_scored_epochs(store.get_run(run_id)["tags"])
    wanted = [e for e in scored if epochs is None or e in set(epochs)]
    root = cache_root()
    run_dir = _run_dir(run_id)
    with _lock:
        run_dir.mkdir(parents=True, exist_ok=True)
        files = []
        for epoch in wanted:
            path = _fetch(store, run_id, contract.per_sample_path(epoch), run_dir)
            if path is not None:
                files.append((epoch, path))
        if with_split:
            _fetch(store, run_id, contract.SPLIT_ARTIFACT, run_dir)
        os.utime(run_dir)
        evict(root, run_id, max_bytes())
    return files


def cached_split(run_id: str) -> Path | None:
    """Return the cached ``split.csv`` of a run, if present."""
    path = _run_dir(run_id) / contract.SPLIT_ARTIFACT
    return path if path.is_file() else None


def fetch_artifact(store: MlflowStore, run_id: str, artifact: str) -> Path | None:
    """Return a local copy of one run artifact (downloaded on first use), or None if the run lacks it."""
    run_dir = _run_dir(run_id)
    with _lock:
        run_dir.mkdir(parents=True, exist_ok=True)
        path = _fetch(store, run_id, artifact, run_dir)
        os.utime(run_dir)
        evict(cache_root(), run_id, max_bytes())
    return path
