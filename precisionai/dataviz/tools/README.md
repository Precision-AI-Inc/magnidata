# Dataviz feature extractor

Batch tool: a CSV of image paths in → one dashboard-ready CSV out (schema-compatible
with `data/datalake_4k.csv`, plus NIMA quality columns). Reproducible: deterministic
feature math, pinned model weights, and a provenance sidecar.

## Inputs (derived per image from the on-disk layout)

From each `image_path` (`<DATASET>/images/<STEM>.png`) the tool finds its siblings:

| Source | Path | Provides |
|--------|------|----------|
| RGB image | the `image_path` itself (full res) | coverage, blur, illumination, white balance, NIMA |
| COCO labels | `<DATASET>/labels/<STEM>.json` (polygon segmentation) | instances, entanglement/overlap, look-alike colour similarity |
| Metadata | `<DATASET>/metadata/images_metadata.csv` (join by stem) | GSD, camera angle, sensor/model, focal length, growth stage, … |
| Mask PNG | `<DATASET>/masks/<STEM>.png` (optional) | passthrough `mask_path` for dashboard overlay |

## Usage

```bash
pip install -r precisionai/dataviz/requirements.txt   # torch CPU build: see file note
python -m precisionai.dataviz.tools.features \
    --input images.csv --output features.csv          # full run (with NIMA)
python -m precisionai.dataviz.tools.features \
    --input images.csv --output features.csv --skip-nima   # fast, pixel/COCO/meta only
# flags: --device cpu|cuda   --limit N   --image-source auto|fullres|thumbnail
```

### Image source / thumbnail fallback

The CSV holds full-resolution `.png` paths under `DATALAKE_ROOT`. Pre-generated
thumbnails live at a parallel root (`datalake-clone-thumbnails/{720,64}`) with the same
relative layout but a **`.jpg`** extension. `--image-source` controls resolution:

- `auto` (default): full-res, falling back to a thumbnail if the full-res file is absent.
- `fullres`: full-res only (rows whose image is missing get `image_source=missing` + an error).
- `thumbnail`: prefer thumbnails (fast pass).

The source used per image is recorded in the `image_source` column
(`fullres`/`thumb720`/`thumb64`/`missing`). Roots are env-overridable via
`DATALAKE_ROOT`, `THUMB_ROOT_720`, `THUMB_ROOT_64`. COCO masks are rasterized at the
loaded image's resolution, so coverage/instance metrics stay valid on thumbnails — but
**image-derived quality features (blur, noise, illumination, NIMA) are not comparable
between full-res and thumbnail rows** (thumbnails are downscaled JPEGs).

Outputs `features.csv` (one row per input image) and `features.csv.provenance.json`
(git commit, library versions, NIMA weight hashes, all params, input hash).

## Output columns (curated schema v2)

Grouped by concept, in this order:

- **Identity**: `image_path`, `width`, `height`
- **Cluster passthrough** (from the input CSV, blank if absent): `cluster`, `cluster_l2`
- **Coverage** (RGB+COCO): `annotation_ratio`, `annotated_px_count`, `fg_green_mean`, `bg_coverage`
- **Instances / entanglement** (COCO): `instance_count`, `real_instance_count`, `mean_instance_area_ratio`, `instance_area_std`, `overlap_ratio`
- **Class / look-alike colour** (COCO+colour): `class_count`, `smallest_class_ratio`, `class_entropy`, `mean_pairwise_color_dist`, `interclass_color_sim`
- **Exposure**: `overexpose_ratio`, `underexpose_ratio`, `shadow_edge_ratio`
- **Focus**: `blur_laplacian`, `tenengrad`
- **Quality panel**: `noise_sigma`, `rms_contrast`, `dynamic_range`, `luma_entropy`, `colorfulness`
- **White balance**: `wb_r_gain`, `wb_b_gain`, `wb_rb_ratio`
- **Learned IQA** (pyiqa): `nima_ava`, `nima_vgg16_ava`, `niqe`, `brisque`
- **Composite**: `complexity_score`, `category`
- **Camera metadata** (from `images_metadata.csv`): `gsd`, `camera_angle` ∈ {`nadir`, `oriented`, `missing`}
- **Domain-specific passthrough** (not computed): `domain_metric`

Notes:
- `camera_angle` is derived from `camera_view` (primary) and the numeric off-nadir angle
  (fallback, nadir if ≤ `NADIR_DEG_TOL`°).
- `complexity_score` = `0.6·structural_difficulty + 0.4·capture_quality_penalty`. The
  quality term uses the learned IQA scores (NIQE/BRISQUE/NIMA) when present, and falls
  back to deterministic proxies (blur + noise) under `--skip-nima` — so scores are most
  comparable within a single run mode. All weights/normalizers are recorded in provenance.
- Legacy dashboard columns and verbose identity/diagnostic fields (`mask_path`, `stem`,
  `error`, …) were dropped. Per-row failures are no longer a column — they're printed as
  an end-of-run stderr summary instead.

## Tests

```bash
python -m pytest precisionai/dataviz/tools/test_features.py -q
```
(Synthetic in-memory COCO + PNG; no datalake, no torch required.)
