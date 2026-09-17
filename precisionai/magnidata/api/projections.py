# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Precomputed 3D projections of a dataset's embeddings.

The 3D view used to ask the API to reduce a dataset's embeddings the first time it was
opened, which put the whole cost — parsing the embeddings file, then PCA or t-SNE over
it — inside a request the browser was waiting on. Projections are instead computed once,
while the dataset is being prepared, and written to a `<stem>_projections.json` sidecar
that the viewer renders directly.

Positions are aligned to the feature CSV's row order, so `positions[i]` belongs to row
`i` — the same contract as the on-demand endpoint's response. A sidecar recording a
different row count than the CSV it sits next to is stale and is ignored.

This module also owns the reduction itself, so the precomputed and on-demand paths
cannot drift apart.
"""

import json
import os
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding

PROJECTION_SCHEMA_VERSION = "dataviz-projections/1.0.0"

# Methods worth paying for up front. `lle` stays on-demand only: it is rarely opened and
# cheap enough next to t-SNE that precomputing it is not worth the build time.
PRECOMPUTED_METHODS = ("pca", "tsne")

# Every reduction the 3D view can ask for.
METHODS = ("pca", "tsne", "lle")

# Embeddings are PCA-reduced to this many dimensions before t-SNE/LLE, which is what
# makes those tractable on high-dimensional vectors.
PCA_PRE_DIMS = 30

# t-SNE cost grows steeply with row count, and a build that never finishes is worse than
# one that skips a projection. Above this many rows t-SNE is left to the on-demand path,
# where the user has explicitly asked for it and can see it running.
TSNE_MAX_ROWS = int(os.environ.get("PROJECTION_TSNE_MAX_ROWS", "60000"))


def sidecar_path(csv_path: str) -> str:
    """Return the projections sidecar path for a dataset's feature CSV."""
    return os.path.splitext(csv_path)[0] + "_projections.json"


def project(matrix: np.ndarray, method: str, pca_pre_dims: int = PCA_PRE_DIMS) -> list[list[float]]:
    """Reduce an [n, d] embedding matrix to 3D positions with `method`.

    Parameters
    ----------
    matrix : np.ndarray
        Embedding vectors, one row per dataset row.
    method : str
        One of `METHODS`.
    pca_pre_dims : int, optional
        Dimensionality the matrix is PCA-reduced to before ``tsne``/``lle``.

    Returns
    -------
    list[list[float]]
        One ``[x, y, z]`` per input row, in input order. A matrix with fewer than three
        rows or fewer than three dimensions cannot fill three axes, so it is reduced as
        far as it allows and the remaining axes are zero — callers always get x/y/z.

    Raises
    ------
    ValueError
        If `method` is not one of `METHODS`.
    """
    if method not in METHODS:
        raise ValueError(f"unknown projection method: {method}")
    n, d = matrix.shape
    if min(n, d) < 3:
        return _degenerate(matrix)
    if method == "pca":
        proj = PCA(n_components=3, random_state=0).fit_transform(matrix)
    else:
        pre = (
            PCA(n_components=min(pca_pre_dims, d), random_state=0).fit_transform(matrix) if d > pca_pre_dims else matrix
        )
        if method == "tsne":
            perplexity = max(5, min(30, (n - 1) // 3))
            proj = TSNE(n_components=3, init="pca", perplexity=perplexity, random_state=0).fit_transform(pre)
        else:
            proj = LocallyLinearEmbedding(n_components=3, n_neighbors=min(12, n - 1), random_state=0).fit_transform(pre)
    return np.asarray(proj, dtype=float).tolist()


def _degenerate(matrix: np.ndarray) -> list[list[float]]:
    """Place rows that cannot support a 3D reduction, padding the unused axes with zeros."""
    n, d = matrix.shape
    components = min(3, n, d)
    positions = np.zeros((n, 3), dtype=float)
    if components and n > 1:  # PCA needs at least two samples to have any variance to explain
        positions[:, :components] = PCA(n_components=components, random_state=0).fit_transform(matrix)
    return positions.tolist()


def compute(matrix: np.ndarray, methods: tuple[str, ...] = PRECOMPUTED_METHODS) -> dict[str, list[list[float]]]:
    """Reduce `matrix` with each requested method, skipping any that is out of range.

    A method is skipped rather than raising when the input is too small for a 3D
    reduction, or when t-SNE would exceed `TSNE_MAX_ROWS`; the result simply omits it.
    """
    n, d = matrix.shape
    if n < 3 or d < 1:
        return {}
    out: dict[str, list[list[float]]] = {}
    for method in methods:
        if method == "tsne" and n > TSNE_MAX_ROWS:
            continue
        out[method] = project(matrix, method)
    return out


def write(path: str, row_count: int, positions_by_method: dict[str, list[list[float]]]) -> None:
    """Write a projections sidecar for a dataset of `row_count` rows."""
    payload = {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "row_count": row_count,
        "projections": positions_by_method,
    }
    with open(path, "w") as f:
        json.dump(payload, f)


def read(path: str, row_count: int, method: str) -> list[list[float]] | None:
    """Return stored positions for `method`, or None when there are no usable ones.

    None covers every "fall back to computing" case: no sidecar, an unreadable or
    malformed one, a sidecar written for a different number of rows, and a sidecar that
    simply does not carry this method.
    """
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            payload: Any = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("row_count") != row_count:
        return None
    positions = (payload.get("projections") or {}).get(method)
    if not isinstance(positions, list) or len(positions) != row_count:
        return None
    return positions
