# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the local per-sample artifact cache, against a real sqlite-backed MLflow store.

Run:  python -m pytest precisionai/magnidata/api/test_artifact_cache.py -q
"""

import os

import artifact_cache
import pytest
from conftest import seed_run

from precisionai.magnidata.training import contract

CSV = "image_name,split,loss_total\na.png,train,0.5\n"


@pytest.fixture(autouse=True)
def cache_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "root"))
    monkeypatch.delenv("MLFLOW_CACHE_MAX_MB", raising=False)


def _run(store, epochs, extra_tags=None):
    artifacts = {contract.per_sample_path(e): CSV for e in epochs}
    artifacts[contract.SPLIT_ARTIFACT] = "image_name,split\na.png,train\n"
    tags = {
        contract.TAG_PER_SAMPLE_SCHEMA: contract.SCHEMA_VERSION,
        contract.TAG_SCORED_EPOCHS: ",".join(map(str, epochs)),
        **(extra_tags or {}),
    }
    return seed_run(store, tags=tags, artifacts=artifacts)


def test_sync_downloads_only_new_epochs(mlflow_store, tmp_path, monkeypatch):
    run_id = _run(mlflow_store, [1, 2])
    first = artifact_cache.sync_per_sample(mlflow_store, run_id)
    assert [e for e, _ in first] == [1, 2]
    assert all(p.read_text() == CSV for _, p in first)
    assert artifact_cache.cached_split(run_id) is not None
    assert first[0][1].is_relative_to(tmp_path / "root" / "data_user" / ".mlflow_cache" / run_id)

    # a third epoch lands: log it and extend the tag
    client = mlflow_store._client
    client.log_text(run_id, CSV, contract.per_sample_path(3))
    mlflow_store.set_tag(run_id, contract.TAG_SCORED_EPOCHS, "1,2,3")
    mtimes = {e: p.stat().st_mtime_ns for e, p in first}
    downloads = []
    original = type(mlflow_store).download_artifact

    def counting(self, rid, path, dst):
        downloads.append(path)
        return original(self, rid, path, dst)

    monkeypatch.setattr(type(mlflow_store), "download_artifact", counting)
    second = artifact_cache.sync_per_sample(mlflow_store, run_id)
    assert [e for e, _ in second] == [1, 2, 3]
    assert downloads == [contract.per_sample_path(3)]
    assert {e: p.stat().st_mtime_ns for e, p in second if e in mtimes} == mtimes


def test_sync_can_narrow_epochs_and_skip_split(mlflow_store):
    run_id = _run(mlflow_store, [1, 2, 3])
    files = artifact_cache.sync_per_sample(mlflow_store, run_id, epochs=[3], with_split=False)
    assert [e for e, _ in files] == [3]
    assert artifact_cache.cached_split(run_id) is None


def test_scored_epoch_without_file_is_skipped(mlflow_store):
    run_id = _run(mlflow_store, [1], extra_tags={contract.TAG_SCORED_EPOCHS: "1,2"})
    assert [e for e, _ in artifact_cache.sync_per_sample(mlflow_store, run_id)] == [1]


def test_run_without_scored_epochs(mlflow_store):
    run_id = seed_run(mlflow_store)
    assert artifact_cache.sync_per_sample(mlflow_store, run_id) == []


def test_eviction_removes_least_recently_used_run(mlflow_store, monkeypatch):
    big = "image_name,split,loss_total\n" + "a.png,train,0.5\n" * 20_000
    runs = [
        seed_run(
            mlflow_store,
            tags={contract.TAG_SCORED_EPOCHS: "1"},
            artifacts={contract.per_sample_path(1): big},
        )
        for _ in range(3)
    ]
    root = artifact_cache.cache_root()
    for i, run_id in enumerate(runs[:2]):
        artifact_cache.sync_per_sample(mlflow_store, run_id, with_split=False)
        os.utime(root / run_id, (1000 + i, 1000 + i))  # runs[0] is the oldest
    size_mb = sum(f.stat().st_size for f in (root / runs[0]).rglob("*") if f.is_file()) / 1024 / 1024
    monkeypatch.setenv("MLFLOW_CACHE_MAX_MB", str(size_mb * 2.5))

    artifact_cache.sync_per_sample(mlflow_store, runs[2], with_split=False)
    assert not (root / runs[0]).exists()
    assert (root / runs[1]).exists()
    assert (root / runs[2]).exists()


def test_eviction_never_removes_the_run_in_use(mlflow_store, monkeypatch):
    run_id = _run(mlflow_store, [1])
    monkeypatch.setenv("MLFLOW_CACHE_MAX_MB", "0")
    files = artifact_cache.sync_per_sample(mlflow_store, run_id)
    assert files[0][1].is_file()


def test_invalid_size_setting_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("MLFLOW_CACHE_MAX_MB", "lots")
    assert artifact_cache.max_bytes() == artifact_cache.DEFAULT_MAX_MB * 1024 * 1024
