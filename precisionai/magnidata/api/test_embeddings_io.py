# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the streaming embeddings reader.

The parser exists because `json.load` on a multi-gigabyte embeddings file costs several
times the file size, so these cover the shapes the tooling writes plus the boundary
cases a chunked reader can get wrong — values split across reads, whitespace between
tokens, and truncated input.

Run:  python -m pytest precisionai/magnidata/api/test_embeddings_io.py -q
"""

import json

import embeddings_io
import numpy as np
import pytest


def test_reads_the_embeddings_wrapper(tmp_path):
    path = tmp_path / "wrapped.json"
    path.write_text(json.dumps({"embeddings": {"a.jpg": [1.0, 2.0], "b.jpg": [3.0, 4.0]}}))

    assert dict(embeddings_io.iter_entries(str(path))) == {"a.jpg": [1.0, 2.0], "b.jpg": [3.0, 4.0]}


def test_reads_a_bare_mapping(tmp_path):
    path = tmp_path / "bare.json"
    path.write_text(json.dumps({"a.jpg": [1.0], "b.jpg": [2.0]}))

    assert dict(embeddings_io.iter_entries(str(path))) == {"a.jpg": [1.0], "b.jpg": [2.0]}


def test_reduces_keys_to_basenames(tmp_path):
    path = tmp_path / "nested.json"
    path.write_text(json.dumps({"embeddings": {"some/deep/dir/a.jpg": [1.0]}}))

    assert list(embeddings_io.iter_entries(str(path))) == [("a.jpg", [1.0])]


def test_skips_non_vector_values(tmp_path):
    path = tmp_path / "meta.json"
    path.write_text(json.dumps({"model": "dinov2", "dim": 2, "embeddings": {"a.jpg": [1.0, 2.0]}}))

    assert list(embeddings_io.iter_entries(str(path))) == [("a.jpg", [1.0, 2.0])]


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 64, 4096])
def test_is_chunk_size_independent(tmp_path, chunk_size):
    """A value must decode identically however the reads split it — the buffer refills mid-value."""
    payload = {"embeddings": {f"img-{i}.jpg": [float(i) + 0.5, -float(i)] for i in range(12)}}
    path = tmp_path / "chunked.json"
    path.write_text(json.dumps(payload, indent=2))  # indentation puts whitespace between every token

    assert dict(embeddings_io.iter_entries(str(path), chunk_size=chunk_size)) == payload["embeddings"]


def test_rejects_a_non_object_document(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]")

    with pytest.raises(ValueError, match="must contain a JSON object"):
        list(embeddings_io.iter_entries(str(path)))


def test_rejects_a_truncated_document(tmp_path):
    path = tmp_path / "truncated.json"
    path.write_text('{"embeddings": {"a.jpg": [1.0, 2.0')

    with pytest.raises(ValueError, match=r"Expecting|Unterminated"):
        list(embeddings_io.iter_entries(str(path)))


def test_matches_json_load_on_a_realistic_file(tmp_path):
    """The streaming result must equal what the whole-file parse produced."""
    payload = {"embeddings": {f"img-{i}.jpg": list(np.random.default_rng(i).normal(size=16)) for i in range(50)}}
    path = tmp_path / "realistic.json"
    path.write_text(json.dumps(payload))

    with open(path) as f:
        reference = json.load(f)["embeddings"]

    assert dict(embeddings_io.iter_entries(str(path))) == reference


# ── CSV helpers ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (["image_path", "width"], "image_path"),
        (["width", "src_image_path"], "src_image_path"),
        (["width", "mask_path"], "mask_path"),
        (["width", "height"], "width"),
        ([], None),
    ],
)
def test_image_path_column_detection(header, expected):
    assert embeddings_io.image_path_column(header) == expected


def test_csv_image_paths_keeps_row_order(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text("image_path,width\nimages/b.jpg,10\nimages/a.jpg,20\n")

    assert embeddings_io.csv_image_paths(str(path)) == ["images/b.jpg", "images/a.jpg"]


# ── Matrix assembly ──────────────────────────────────────────────────────────────


def _write_embeddings(tmp_path, payload):
    path = tmp_path / "emb.json"
    path.write_text(json.dumps({"embeddings": payload}))
    return str(path)


def test_matrix_follows_the_given_name_order(tmp_path):
    path = _write_embeddings(tmp_path, {"a.jpg": [1.0, 2.0], "b.jpg": [3.0, 4.0]})

    matrix = embeddings_io.matrix_for_names(["b.jpg", "a.jpg"], path)

    np.testing.assert_array_equal(matrix, np.array([[3.0, 4.0], [1.0, 2.0]], dtype=np.float32))


def test_matrix_zero_fills_names_without_an_entry(tmp_path):
    path = _write_embeddings(tmp_path, {"a.jpg": [1.0, 2.0]})

    matrix = embeddings_io.matrix_for_names(["a.jpg", "missing.jpg"], path)

    np.testing.assert_array_equal(matrix, np.array([[1.0, 2.0], [0.0, 0.0]], dtype=np.float32))


def test_matrix_zero_fills_wrong_length_vectors(tmp_path):
    path = _write_embeddings(tmp_path, {"a.jpg": [1.0, 2.0], "b.jpg": [9.0]})

    matrix = embeddings_io.matrix_for_names(["a.jpg", "b.jpg"], path)

    np.testing.assert_array_equal(matrix, np.array([[1.0, 2.0], [0.0, 0.0]], dtype=np.float32))


def test_matrix_repeats_a_vector_across_duplicate_names(tmp_path):
    path = _write_embeddings(tmp_path, {"a.jpg": [1.0, 2.0]})

    matrix = embeddings_io.matrix_for_names(["a.jpg", "a.jpg"], path)

    np.testing.assert_array_equal(matrix, np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32))


def test_matrix_for_csv_aligns_to_the_csv(tmp_path):
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text("image_path\nimages/b.jpg\nimages/a.jpg\n")
    json_path = _write_embeddings(tmp_path, {"a.jpg": [1.0], "b.jpg": [2.0]})

    matrix = embeddings_io.matrix_for_csv(str(csv_path), json_path)

    np.testing.assert_array_equal(matrix, np.array([[2.0], [1.0]], dtype=np.float32))
