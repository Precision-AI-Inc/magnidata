# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for precomputed 3D projections.

The reduction itself is scikit-learn's, so these cover this module's own contract: that
each method returns one position per row, that oversized or undersized inputs are
skipped rather than raising, and — most importantly — that `read` refuses every stale or
malformed sidecar instead of handing the viewer positions that do not match its rows.

Run:  python -m pytest precisionai/magnidata/api/test_projections.py -q
"""

import json

import numpy as np
import projections
import pytest


@pytest.fixture
def clustered():
    """A [60, 8] matrix with three well-separated clusters, big enough for t-SNE's perplexity."""
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(3, 8)) * 5
    labels = np.repeat(np.arange(3), 20)
    return (centers[labels] + rng.normal(scale=0.2, size=(60, 8))).astype(np.float32)


@pytest.mark.parametrize("method", ["pca", "tsne", "lle"])
def test_project_returns_three_coordinates_per_row(clustered, method):
    positions = projections.project(clustered, method)

    assert len(positions) == len(clustered)
    assert all(len(p) == 3 for p in positions)


def test_project_rejects_an_unknown_method(clustered):
    with pytest.raises(ValueError, match="unknown projection method"):
        projections.project(clustered, "umap")


def test_compute_produces_every_precomputed_method(clustered):
    result = projections.compute(clustered)

    assert set(result) == set(projections.PRECOMPUTED_METHODS)
    assert all(len(v) == len(clustered) for v in result.values())


def test_compute_returns_nothing_for_too_few_rows():
    assert projections.compute(np.zeros((2, 8), dtype=np.float32)) == {}


def test_compute_returns_nothing_without_dimensions():
    assert projections.compute(np.zeros((10, 0), dtype=np.float32)) == {}


def test_compute_skips_tsne_above_the_row_cap(clustered, monkeypatch):
    """t-SNE is left to the on-demand path rather than stalling a build indefinitely."""
    monkeypatch.setattr(projections, "TSNE_MAX_ROWS", 10)

    result = projections.compute(clustered)

    assert "tsne" not in result
    assert "pca" in result


def test_sidecar_path_sits_next_to_the_csv():
    assert projections.sidecar_path("data_user/demo.csv") == "data_user/demo_projections.json"


# ── Sidecar round-trip and rejection ─────────────────────────────────────────────


def test_round_trip(tmp_path):
    path = str(tmp_path / "demo_projections.json")
    positions = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]

    projections.write(path, 2, {"pca": positions})

    assert projections.read(path, 2, "pca") == positions


def test_read_returns_none_without_a_sidecar(tmp_path):
    assert projections.read(str(tmp_path / "absent.json"), 2, "pca") is None


def test_read_returns_none_for_a_method_the_sidecar_lacks(tmp_path):
    path = str(tmp_path / "p.json")
    projections.write(path, 2, {"pca": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]})

    assert projections.read(path, 2, "tsne") is None


def test_read_rejects_a_sidecar_written_for_a_different_row_count(tmp_path):
    """A CSV rebuilt with more or fewer rows must not be rendered with the old positions."""
    path = str(tmp_path / "p.json")
    projections.write(path, 2, {"pca": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]})

    assert projections.read(path, 3, "pca") is None


def test_read_rejects_positions_that_do_not_match_the_row_count(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"row_count": 3, "projections": {"pca": [[0.0, 0.0, 0.0]]}}))

    assert projections.read(str(path), 3, "pca") is None


def test_read_rejects_malformed_json(tmp_path):
    path = tmp_path / "p.json"
    path.write_text("{not json")

    assert projections.read(str(path), 2, "pca") is None


def test_read_rejects_a_non_object_document(tmp_path):
    path = tmp_path / "p.json"
    path.write_text("[1, 2, 3]")

    assert projections.read(str(path), 2, "pca") is None


def test_written_sidecar_records_the_schema_version(tmp_path):
    path = tmp_path / "p.json"
    projections.write(str(path), 1, {"pca": [[0.0, 0.0, 0.0]]})

    assert json.loads(path.read_text())["schema_version"] == projections.PROJECTION_SCHEMA_VERSION


# ── Degenerate inputs ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("shape", [(2, 8), (8, 2), (8, 1), (1, 8), (3, 3)])
def test_project_always_returns_three_axes(shape):
    """Too few rows or dimensions for a 3D reduction must still yield x/y/z, not an error."""
    matrix = np.random.default_rng(0).normal(size=shape).astype(np.float32)

    positions = projections.project(matrix, "pca")

    assert len(positions) == shape[0]
    assert all(len(p) == 3 for p in positions)


def test_project_zero_pads_the_axes_a_low_rank_input_cannot_fill():
    matrix = np.random.default_rng(0).normal(size=(8, 2)).astype(np.float32)

    positions = projections.project(matrix, "pca")

    assert all(p[2] == 0.0 for p in positions)


def test_project_handles_a_matrix_with_no_dimensions():
    positions = projections.project(np.zeros((4, 0), dtype=np.float32), "pca")

    assert positions == [[0.0, 0.0, 0.0]] * 4
