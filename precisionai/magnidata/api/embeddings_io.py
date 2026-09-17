# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Reading embeddings sidecars into a row-aligned matrix.

An embeddings file maps an image name to that image's vector. `json.load` costs several
times the file's size once every number has become a Python float inside a list — enough
to get the API OOM-killed on a multi-gigabyte file — so entries are streamed one at a
time and written straight into a preallocated array.

Shared by the request path (`app.py`) and the dataset-preparation path
(`dataset_builder.py`), which must agree on row order for a precomputed projection to
line up with the feature CSV it is served alongside.
"""

import csv
import json
import os
from collections.abc import Iterator, Sequence
from typing import Any, TextIO

import numpy as np


class _JsonStreamReader:
    """Incremental JSON reader: decodes one value at a time from a chunk-by-chunk file read."""

    def __init__(self, fh: TextIO, chunk_size: int) -> None:
        self._fh = fh
        self._chunk_size = chunk_size
        self._decoder = json.JSONDecoder()
        self._buf = ""
        self._pos = 0

    def _fill(self) -> bool:
        """Drop the consumed prefix and append another chunk. False at end of file."""
        if self._pos:
            self._buf = self._buf[self._pos :]
            self._pos = 0
        chunk = self._fh.read(self._chunk_size)
        if not chunk:
            return False
        self._buf += chunk
        return True

    def peek(self) -> str:
        """Return the next non-whitespace character without consuming it ("" at end of file)."""
        while True:
            while self._pos < len(self._buf) and self._buf[self._pos].isspace():
                self._pos += 1
            if self._pos < len(self._buf):
                return self._buf[self._pos]
            if not self._fill():
                return ""

    def skip(self) -> None:
        """Consume the character the last `peek` returned."""
        self._pos += 1

    def decode(self) -> Any:
        """Decode the next JSON value, reading more input for as long as it is truncated."""
        self.peek()  # raw_decode does not skip leading whitespace
        while True:
            try:
                value, end = self._decoder.raw_decode(self._buf, self._pos)
            except ValueError:
                if not self._fill():
                    raise
                continue
            self._pos = end
            return value


def _next_object_key(reader: _JsonStreamReader) -> str | None:
    """Consume the next `"key":` of a JSON object, or return None at its closing brace."""
    char = reader.peek()
    while char == ",":
        reader.skip()
        char = reader.peek()
    if char in ("}", ""):
        return None
    if char != '"':
        raise ValueError("malformed embeddings file")
    key = str(reader.decode())
    if reader.peek() != ":":
        raise ValueError("malformed embeddings file")
    reader.skip()
    return key


def iter_entries(path: str, chunk_size: int = 1 << 20) -> Iterator[tuple[str, list[float]]]:
    """Yield (image basename, vector) pairs from an embeddings JSON, one entry at a time.

    Both shapes the tooling writes are accepted: a bare mapping of name → vector, and a
    {"embeddings": {...}} wrapper. Values that are not vectors — metadata sitting
    alongside the wrapper — are skipped.
    """
    with open(path) as fh:
        reader = _JsonStreamReader(fh, chunk_size)
        if reader.peek() != "{":
            raise ValueError("embeddings file must contain a JSON object")
        reader.skip()

        descended = False
        while (key := _next_object_key(reader)) is not None:
            if not descended and key == "embeddings" and reader.peek() == "{":
                reader.skip()  # step into the wrapper and read its entries instead
                descended = True
                continue
            value = reader.decode()
            if isinstance(value, list):
                yield os.path.basename(key), value


def image_path_column(cols: Sequence[str]) -> str | None:
    """Return the column holding each row's image path, or None for an empty header."""
    for c in cols:
        lc = c.lower()
        if lc == "image_path" or ("image" in lc and "path" in lc):
            return c
    return next((c for c in cols if "path" in c.lower()), cols[0] if cols else None)


def csv_image_paths(csv_path: str) -> list[str]:
    """Return the image-path value of every row, in the CSV's own order."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = list(reader)
    col = image_path_column(cols)
    return [r.get(col, "") if col else "" for r in rows]


def matrix_for_names(names: Sequence[str], json_path: str) -> np.ndarray:
    """Stream an embeddings JSON into an [n, d] array aligned to `names`.

    Names without an entry — and entries whose vector does not match the first vector's
    length — are left as zero rows, so the result always has one row per name.
    """
    rows_of_name: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        rows_of_name.setdefault(os.path.basename(str(name)), []).append(i)

    matrix: np.ndarray | None = None
    dim = 0
    for name, vec in iter_entries(json_path):
        rows = rows_of_name.get(name)
        if matrix is None and vec:
            dim = len(vec)
            matrix = np.zeros((len(names), dim), dtype=np.float32)
        if rows and matrix is not None and len(vec) == dim:
            matrix[rows] = vec
    return matrix if matrix is not None else np.zeros((len(names), 0), dtype=np.float32)


def matrix_for_csv(csv_path: str, json_path: str) -> np.ndarray:
    """Stream an embeddings JSON into an [n, d] array aligned to a feature CSV's row order."""
    return matrix_for_names(csv_image_paths(csv_path), json_path)
