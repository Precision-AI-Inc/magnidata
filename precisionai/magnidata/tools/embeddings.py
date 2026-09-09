# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Batch embeddings tool — CSV (image_path) in -> embeddings JSON out.

For each image listed in the input CSV, resolve it to a readable file
(full-res, falling back to a thumbnail — same convention as features.py), run
it through a CNN embedding model, and write one entry per successfully-read
image to the output JSON: {"embeddings": {"<basename>": [floats...]}}. This is
exactly the format api/app.py already reads for /api/embedding/* and the
Graph3D "Compare" overlay. A provenance sidecar is written alongside.

Naming convention (by --output path, not enforced here): the first embeddings
file for a dataset -> data/<stem>.json (the dashboard's default embeddings);
a second model run against the same dataset -> data/<stem>_<variant>.json
(shows up as a "Compare" option).

Usage:
    python -m precisionai.magnidata.tools.embeddings \
        --input in.csv --output data/datalake_4k_dinov2.json \
        [--model dinov2] [--device cpu] [--limit N] [--image-source auto]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
from PIL import Image

from . import image_source, provenance
from .embedding_models import MODELS, EmbeddingModel

# tqdm is part of the optional `[ml]` extra (see pyproject.toml), so it may not be
# installed even when this module is imported — fall back to periodic stderr counts.
try:
    from tqdm import tqdm

    _TQDM_AVAILABLE = True
except ImportError:
    tqdm = None
    _TQDM_AVAILABLE = False


def _progress(iterable: Iterable[Any], total: int) -> Iterable[Any]:
    """Wrap with a tqdm bar if available; else fall back to periodic stderr counts."""
    if _TQDM_AVAILABLE and tqdm is not None:
        return tqdm(iterable, total=total, unit="img", desc="embedding", file=sys.stderr)

    def gen() -> Iterator[Any]:
        for i, x in enumerate(iterable):
            if (i + 1) % 50 == 0 or (i + 1) == total:
                print(f"  {i + 1}/{total}", file=sys.stderr)
            yield x

    return gen()


class AllRowsFailedError(RuntimeError):
    """Every input row failed to embed, so no output was written.

    Raised instead of writing ``{"embeddings": {}}``, which would silently clobber a
    good pre-existing embeddings file at the same path (e.g. after a misconfigured
    DATALAKE_ROOT made every image unresolvable).
    """


def embed_row(
    image_path: str, model_entry: EmbeddingModel, handle: Any, image_mode: str = "auto"
) -> tuple[str, np.ndarray | None, str, str]:
    """Embed one image. Returns (basename, vector, source, error_str); never raises.

    ``vector`` is None when the image could not be resolved or read, and then
    ``error_str`` says why — mirroring features.py::process_row's convention of
    returning an error string that is empty on success, which run() aggregates into
    an end-of-run stderr summary. ``source`` is image_source.resolve()'s label
    ('fullres' | 'thumb720' | 'thumb64' | 'missing'); run() tallies it into the
    provenance sidecar so full-res and thumbnail-derived vectors (which are not
    directly comparable) can be told apart after the fact.
    """
    basename = os.path.basename(image_path)
    resolved, source = image_source.resolve(image_path, image_mode)
    if source == "missing":
        return basename, None, source, "image:not_found (full-res and thumbnails)"
    try:
        with open(resolved, "rb") as f:
            img = Image.open(f).convert("RGB")
        rgb = np.asarray(img, dtype=np.uint8)
        vec = model_entry.embed(handle, rgb)
    except Exception as exc:
        return basename, None, source, f"embed:{exc}"
    return basename, vec, source, ""


def run(
    input_csv: str,
    output_json: str,
    model: str = "dinov2",
    device: str = "cpu",
    limit: int | None = None,
    image_mode: str = "auto",
) -> int:
    """Embed every image in `input_csv` with `model` and write the embeddings JSON.

    Loads the requested model once, embeds each resolvable image, writes
    ``{"embeddings": {...}}`` to `output_json` plus a provenance sidecar, and
    returns the number of images successfully embedded. Raises
    `AllRowsFailedError` instead of writing an empty output when every row of a
    non-empty input failed, so a good pre-existing file at `output_json` survives
    a misconfigured run.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; available: {sorted(MODELS)}")
    model_entry = MODELS[model]

    with open(input_csv, newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        path_col = image_source.find_path_column(fields)
        rows = list(reader)
    if limit:
        rows = rows[:limit]

    print(f"Loading {model} on {device} ...", file=sys.stderr)
    handle = model_entry.load(device)

    embeddings: dict[str, list[float]] = {}
    errored: list[tuple[str, str]] = []
    source_counts: dict[str, int] = {}
    observed_dim: int | None = None
    for r in _progress(rows, len(rows)):
        image_path = (r.get(path_col) or "").strip()
        name, vec, source, err = embed_row(image_path, model_entry, handle, image_mode)
        source_counts[source] = source_counts.get(source, 0) + 1
        if vec is None:
            errored.append((image_path, err))
            continue
        values = [round(float(x), 6) for x in vec]
        if observed_dim is None:
            observed_dim = len(values)
        embeddings[name] = values

    if errored:
        print(f"WARNING: {len(errored)}/{len(rows)} rows had errors:", file=sys.stderr)
        for path, err in errored[:20]:
            print(f"  {os.path.basename(path)}: {err}", file=sys.stderr)
        if len(errored) > 20:
            print(f"  … and {len(errored) - 20} more", file=sys.stderr)

    # Nothing usable came out of a non-empty input: refuse to write, so a good
    # pre-existing embeddings file at this path survives a misconfigured run. (An
    # input CSV with zero rows is not a failure — it writes an empty file as usual.)
    if rows and not embeddings:
        raise AllRowsFailedError(
            f"all {len(rows)} rows failed to embed — nothing written to {output_json} "
            f"(an existing file there was left untouched). See the per-row errors above; "
            f"a common cause is DATALAKE_ROOT / thumbnail roots pointing somewhere the "
            f"input CSV's images do not exist."
        )

    with open(output_json, "w") as f:
        json.dump({"embeddings": embeddings}, f)

    # Record what the run actually did, not just what was configured: the observed
    # vector dimension (which may differ from the registry's declared `dim`) and the
    # per-source breakdown, since thumbnail-derived vectors are not comparable to
    # full-res ones.
    params = {
        "model": model,
        "dim": model_entry.dim,
        "image_mode": image_mode,
        "stats": {
            "rows": len(rows),
            "embedded": len(embeddings),
            "skipped": len(errored),
            "sources": source_counts,
            "observed_dim": observed_dim,
        },
    }
    # Best-effort checkpoint hashing (like nima.py's weight_paths()) is left for a
    # follow-up — timm/HF cache layout varies by download backend and model.
    prov = provenance.build(
        input_csv,
        output_json,
        params,
        f"dataviz-embeddings/{model}",
        weight_paths=[],
        version_modules=("numpy", "PIL", "torch", "timm"),
        determinism_notes=[
            "Vectors come from the configured embedding model's own preprocessing (which "
            "may resize/crop the input image) — not directly comparable to full-resolution "
            "deterministic pixel features.",
            "Per-row image resolution (full-res vs. thumbnail fallback) affects what the "
            "model actually sees; see this sidecar's 'stats' for this run's source breakdown.",
        ],
    )
    provenance.write(prov, output_json + ".provenance.json")

    print(f"Wrote {len(embeddings)} embeddings -> {output_json}", file=sys.stderr)
    return len(embeddings)


def main(argv: list[str] | None = None) -> None:
    """Parse CLI args and run the embeddings generator, exiting 1 on total failure."""
    ap = argparse.ArgumentParser(description="Dataviz embeddings generator (CSV -> JSON).")
    ap.add_argument("--input", required=True, help="input CSV with an image_path column")
    ap.add_argument("--output", required=True, help="output embeddings JSON path")
    ap.add_argument("--model", default="dinov2", choices=sorted(MODELS), help="embedding model to use")
    ap.add_argument("--device", default="cpu", help="torch device (cpu/cuda)")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N rows")
    ap.add_argument(
        "--image-source",
        dest="image_mode",
        choices=["auto", "fullres", "thumbnail"],
        default="auto",
        help="auto: full-res then thumbnail fallback (default); fullres: full-res only; thumbnail: prefer thumbnails",
    )
    args = ap.parse_args(argv)
    try:
        run(args.input, args.output, args.model, args.device, args.limit, args.image_mode)
    except AllRowsFailedError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
