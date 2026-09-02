# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Batch feature-extraction tool — CSV (image_path) in -> dashboard-ready CSV out.

For each image listed in the input CSV, derive its sibling COCO label and metadata
from the on-disk layout (``<DATASET>/images/<STEM>.png`` ->
``<DATASET>/labels/<STEM>.json``, ``<DATASET>/metadata/images_metadata.csv``),
compute the deterministic feature set, run NIMA, join metadata, and write one output
row per image in the dashboard schema. A provenance sidecar is written alongside.

Usage:
    python -m precisionai.agriviz.tools.features --input in.csv --output out.csv
        [--device cpu] [--limit N] [--skip-nima]
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys

import numpy as np
from PIL import Image

from . import coco_labels, image_source, instance_features, metadata_join, pixel_features, provenance, quality_features
from .schema import CLUSTER, FEATURE_SCHEMA_VERSION, OUTPUT_COLUMNS, blank_row

Image.MAX_IMAGE_PIXELS = None        # these are legitimately large images

ROUND_DECIMALS = 6

# ── Difficulty composite (transparent, reproducible; recorded in provenance) ───────
# complexity = STRUCT_MIX * structural_difficulty + QUALITY_MIX * capture_quality_penalty
#
# Structural difficulty (always available from COCO + pixels): how hard the SCENE is to
# segment — entanglement, look-alike-class colour similarity, class mixing.
# Capture-quality penalty (0 good .. 1 bad): how DEGRADED the image is. Prefers the
# learned IQA scores (NIQE/BRISQUE technical + NIMA aesthetic) now that we have them,
# and falls back to deterministic proxies (blur + noise) when --skip-nima is used.
STRUCT_MIX, QUALITY_MIX = 0.60, 0.40
STRUCT_WEIGHTS = {"overlap_ratio": 0.45, "interclass_color_sim": 0.35, "class_mix": 0.20}
QUALITY_WEIGHTS_LEARNED = {"niqe": 0.35, "brisque": 0.25, "nima_poor": 0.20, "exposure": 0.20}
QUALITY_WEIGHTS_DET = {"blur_soft": 0.45, "noise": 0.35, "exposure": 0.20}

# normalizers mapping each metric to 0..1
BLUR_REF = 500.0                  # laplacian variance treated as "fully sharp"
NOISE_REF = 12.0                  # noise sigma treated as "very noisy"
NIQE_LO, NIQE_HI = 3.0, 8.0       # niqe lower=better
BRISQUE_REF = 80.0                # brisque lower=better (0..100)
NIMA_LO, NIMA_HI = 3.0, 6.5       # nima higher=better; poorness = (HI-nima)/(HI-LO)

PARAMS = {
    "bg_threshold_sum": 30,
    "class_map_offset": coco_labels.CLASS_MAP_OFFSET,
    "min_instance_px": instance_features.MIN_INSTANCE_PX,
    "luma_coeff": list(pixel_features.LUMA_COEFF),
    "overexpose_t": pixel_features.OVEREXPOSE_T,
    "underexpose_t": pixel_features.UNDEREXPOSE_T,
    "shadow_band": [pixel_features.SHADOW_LO, pixel_features.SHADOW_HI],
    "nadir_deg_tol": metadata_join.NADIR_DEG_TOL,
    "complexity": {
        "struct_mix": STRUCT_MIX, "quality_mix": QUALITY_MIX,
        "struct_weights": STRUCT_WEIGHTS,
        "quality_weights_learned": QUALITY_WEIGHTS_LEARNED,
        "quality_weights_deterministic": QUALITY_WEIGHTS_DET,
        "blur_ref": BLUR_REF, "noise_ref": NOISE_REF,
        "niqe_lo_hi": [NIQE_LO, NIQE_HI], "brisque_ref": BRISQUE_REF,
        "nima_lo_hi": [NIMA_LO, NIMA_HI],
    },
    "round_decimals": ROUND_DECIMALS,
}


def _progress(iterable, total):
    """Wrap with a tqdm bar if available; else fall back to periodic stderr counts."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, unit="img", desc="extracting", file=sys.stderr)
    except ImportError:
        def gen():
            for i, x in enumerate(iterable):
                if (i + 1) % 50 == 0 or (i + 1) == total:
                    print(f"  {i + 1}/{total}", file=sys.stderr)
                yield x
        return gen()


def _round(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):   # NaN/Inf -> blank
            return ""
        return round(v, ROUND_DECIMALS)
    return v


def _derive_paths(image_path: str):
    images_dir = os.path.dirname(image_path)
    dataset_dir = os.path.dirname(images_dir)
    stem, ext = os.path.splitext(os.path.basename(image_path))
    label_path = os.path.join(dataset_dir, "labels", stem + ".json")
    mask_path = os.path.join(dataset_dir, "masks", stem + ".png")
    return dataset_dir, stem, ext, label_path, mask_path


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _complexity(row: dict) -> tuple[float, int, str]:
    """Return (complexity_score 0..1, category 1..10, basis).

    basis is 'learned' when NIMA/NIQE/BRISQUE are present, else 'deterministic'.
    """
    def g(k):
        v = row.get(k, "")
        return float(v) if isinstance(v, (int, float)) else 0.0

    # structural (scene) difficulty — always available
    cc = g("class_count")
    class_mix = _clip01(g("class_entropy") / np.log2(cc)) if cc > 1 else 0.0
    struct = (STRUCT_WEIGHTS["overlap_ratio"] * _clip01(g("overlap_ratio"))
              + STRUCT_WEIGHTS["interclass_color_sim"] * _clip01(g("interclass_color_sim"))
              + STRUCT_WEIGHTS["class_mix"] * class_mix)

    exposure = _clip01(g("overexpose_ratio") + g("underexpose_ratio"))

    # capture-quality penalty — prefer learned IQA, fall back to deterministic proxies
    learned = all(isinstance(row.get(k), (int, float)) for k in ("niqe", "brisque", "nima_ava"))
    if learned:
        niqe_n = _clip01((g("niqe") - NIQE_LO) / (NIQE_HI - NIQE_LO))
        bris_n = _clip01(g("brisque") / BRISQUE_REF)
        nima_poor = _clip01((NIMA_HI - g("nima_ava")) / (NIMA_HI - NIMA_LO))
        w = QUALITY_WEIGHTS_LEARNED
        quality = (w["niqe"] * niqe_n + w["brisque"] * bris_n
                   + w["nima_poor"] * nima_poor + w["exposure"] * exposure)
        basis = "learned"
    else:
        blur_soft = 1.0 - _clip01(g("blur_laplacian") / BLUR_REF)
        noise_n = _clip01(g("noise_sigma") / NOISE_REF)
        w = QUALITY_WEIGHTS_DET
        quality = w["blur_soft"] * blur_soft + w["noise"] * noise_n + w["exposure"] * exposure
        basis = "deterministic"

    score = _clip01(STRUCT_MIX * struct + QUALITY_MIX * quality)
    category = int(min(10, max(1, int(score * 10) + 1)))
    return score, category, basis


def process_row(image_path: str, scorer=None, image_mode: str = "auto") -> tuple[dict, str]:
    """Compute one output row. Returns (row, error_str); never raises.

    ``error_str`` is empty on success. It is no longer an output column (the curated
    schema dropped it) — run() aggregates errors into an end-of-run stderr summary.
    """
    row = blank_row()
    dataset_dir, stem, ext, label_path, mask_path = _derive_paths(image_path)
    row["image_path"] = image_path
    row["stem"] = stem
    row["extension"] = ext.lstrip(".")
    row["batch"] = os.path.basename(dataset_dir)
    if os.path.isfile(mask_path):
        row["mask_path"] = mask_path
    errors = []

    # Resolve to a readable file (full-res, falling back to a thumbnail if missing).
    resolved, source = image_source.resolve(image_path, image_mode)
    row["image_source"] = source
    if source == "missing":
        errors.append("image:not_found (full-res and thumbnails)")

    # ── RGB + NIMA ────────────────────────────────────────────────────────────────
    rgb = None
    luma = None   # stays None if the try block below fails before computing it
    if source != "missing":
        try:
            with open(resolved, "rb") as f:
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
            rgb = np.asarray(img, dtype=np.uint8)
            row["width"], row["height"] = int(img.width), int(img.height)
            row.update(pixel_features.white_balance(rgb))
            luma = pixel_features.to_luma(rgb)
            row.update(pixel_features.illumination(luma, np.ones(luma.shape, dtype=bool)))
            row.update(pixel_features.focus(luma))
            row.update(quality_features.compute(rgb, luma))
        except Exception as exc:
            errors.append(f"image:{exc}")

    if scorer is not None and rgb is not None:
        try:
            row.update(scorer.score(resolved))
        except Exception as exc:
            errors.append(f"nima:{exc}")

    # ── COCO labels: coverage + instances + look-alike-class colour ────────────────
    if rgb is not None and os.path.isfile(label_path):
        try:
            labels = coco_labels.parse(label_path)
            # Rasterize at the loaded image's resolution (handles thumbnails too).
            class_map = coco_labels.rasterize_class_map(labels, target_wh=(rgb.shape[1], rgb.shape[0]))
            fg = coco_labels.foreground_from_class_map(class_map)
            bg = coco_labels.background_from_class_map(class_map)
            row.update(pixel_features.coverage(fg, rgb))
            row.update(instance_features.instance_metrics(labels, rgb, class_map))
            row["bg_coverage"] = float(bg.sum()) / float(bg.size) if bg.size else 0.0
            # refine illumination shadow proxy to the actual foreground (only if the
            # RGB block above actually got far enough to compute luma)
            if luma is not None:
                row.update(pixel_features.illumination(luma, fg))
        except Exception as exc:
            errors.append(f"coco:{exc}")
    elif rgb is not None:
        errors.append("coco:label_not_found")

    # ── Metadata join ─────────────────────────────────────────────────────────────
    try:
        meta, _matched = metadata_join.lookup(dataset_dir, stem)
        row.update(meta)
    except Exception as exc:
        errors.append(f"meta:{exc}")

    # ── Composite difficulty ──────────────────────────────────────────────────────
    if rgb is not None:
        row["complexity_score"], row["category"], _basis = _complexity(row)

    out = {k: _round(row.get(k, "")) for k in OUTPUT_COLUMNS}
    return out, "; ".join(errors)


def run(input_csv: str, output_csv: str, device: str = "cpu",
        limit: int | None = None, skip_nima: bool = False,
        image_mode: str = "auto") -> int:
    with open(input_csv, newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        path_col = image_source.find_path_column(fields)
        rows = list(reader)
    if limit:
        rows = rows[:limit]

    # Cluster labels to carry through from the input CSV (if it has them), so the 3D view
    # can auto-cluster on them downstream.
    cluster_cols = [c for c in CLUSTER if c in fields]

    scorer = None
    if not skip_nima:
        from .nima import NimaScorer
        print(f"Loading learned IQA models (NIMA/NIQE/BRISQUE) on {device} ...", file=sys.stderr)
        scorer = NimaScorer(device=device)

    out_rows, errored = [], []
    for r in _progress(rows, len(rows)):
        image_path = (r.get(path_col) or "").strip()
        row, err = process_row(image_path, scorer, image_mode)
        for c in cluster_cols:                       # passthrough cluster labels (verbatim)
            row[c] = (r.get(c) or "")
        out_rows.append(row)
        if err:
            errored.append((image_path, err))

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(out_rows)

    if errored:
        print(f"WARNING: {len(errored)}/{len(out_rows)} rows had errors:", file=sys.stderr)
        for path, err in errored[:20]:
            print(f"  {os.path.basename(path)}: {err}", file=sys.stderr)
        if len(errored) > 20:
            print(f"  … and {len(errored) - 20} more", file=sys.stderr)

    params = dict(PARAMS, image_mode=image_mode)
    if scorer is not None:
        params.update(scorer.proc_params())
    prov = provenance.build(
        input_csv, output_csv, params, FEATURE_SCHEMA_VERSION,
        weight_paths=scorer.weight_paths() if scorer else [])
    provenance.write(prov, output_csv + ".provenance.json")

    print(f"Wrote {len(out_rows)} rows -> {output_csv}", file=sys.stderr)
    return len(out_rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Agriviz image feature extractor (CSV -> CSV).")
    ap.add_argument("--input", required=True, help="input CSV with an image_path column")
    ap.add_argument("--output", required=True, help="output CSV path")
    ap.add_argument("--device", default="cpu", help="torch device for NIMA (cpu/cuda)")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N rows")
    ap.add_argument("--skip-nima", action="store_true",
                    help="skip all learned IQA models (NIMA/NIQE/BRISQUE); pixel/COCO/meta only")
    ap.add_argument("--image-source", choices=["auto", "fullres", "thumbnail"], default="auto",
                    help="auto: full-res then thumbnail fallback (default); fullres: full-res only; "
                         "thumbnail: prefer thumbnails (fast). Thumbnails are downscaled JPEGs — "
                         "image-derived features won't match full-res.")
    args = ap.parse_args(argv)
    run(args.input, args.output, args.device, args.limit, args.skip_nima, args.image_source)


if __name__ == "__main__":
    main()
