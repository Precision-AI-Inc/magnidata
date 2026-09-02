# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Output schema for the feature-extraction tool — a curated, grouped column set.
Domain-agnostic by design: every computed column describes generic objects/instances
in an image, not any particular subject matter.

Columns are ordered by concept. Everything here is produced by the tool: image/COCO
features, learned IQA scores, a composite difficulty, two trimmed camera-metadata
fields, and a few optional domain-specific passthroughs. (Legacy dashboard columns and
verbose identity/diagnostic fields were intentionally dropped.)

`FEATURE_SCHEMA_VERSION` is recorded in the provenance sidecar so any output can be
tied back to the exact column contract that produced it. Bumped to 2.0.0 for the
curated schema.
"""

FEATURE_SCHEMA_VERSION = "agriviz-features/2.1.0"

# ── Identity ──────────────────────────────────────────────────────────────────────
IDENTITY = ["image_path", "width", "height"]

# ── Cluster labels carried through from the input CSV (enables 3D auto-clustering) ──
# Populated from the input row when present (e.g. agri-benc's `cluster` / `cluster_l2`);
# blank otherwise. The dashboard auto-detects `cluster` to colour/partition the 3D view.
CLUSTER = ["cluster", "cluster_l2"]

# ── Foreground coverage / segmentation (RGB + COCO) ─────────────────────────────────
COVERAGE = ["green_annotation_ratio", "annotated_px_count", "green_mass", "bg_coverage"]

# ── Instances & entanglement (COCO) ─────────────────────────────────────────────────
INSTANCES = ["instance_count", "real_instance_count", "mean_instance_area_ratio",
             "instance_area_std", "overlap_ratio"]

# ── Class composition & look-alike colour (COCO + colour) ──────────────────────────
CLASSES = ["class_count", "smallest_class_ratio", "class_entropy",
           "mean_pairwise_color_dist", "interclass_color_sim"]

# ── Exposure / illumination (RGB) ───────────────────────────────────────────────────
EXPOSURE = ["overexpose_ratio", "underexpose_ratio", "shadow_edge_ratio"]

# ── Focus / sharpness (RGB) ─────────────────────────────────────────────────────────
FOCUS = ["blur_laplacian", "tenengrad"]

# ── Noise / contrast / colour panel (RGB) ───────────────────────────────────────────
QUALITY_PANEL = ["noise_sigma", "rms_contrast", "dynamic_range", "luma_entropy", "colorfulness"]

# ── White balance / colour cast (RGB gray-world) ───────────────────────────────────
WHITE_BALANCE = ["wb_r_gain", "wb_b_gain", "wb_rb_ratio"]

# ── Learned image-quality (pyiqa) ───────────────────────────────────────────────────
LEARNED_IQA = ["nima_ava", "nima_vgg16_ava", "niqe", "brisque"]

# ── Composite difficulty ────────────────────────────────────────────────────────────
COMPOSITE = ["complexity_score", "category"]

# ── Camera metadata (trimmed; from images_metadata.csv) ─────────────────────────────
# camera_angle is a category derived from the off-nadir angle + camera_view.
CAMERA_META = ["gsd", "camera_angle"]   # camera_angle ∈ {nadir, oriented, missing}

# ── Optional domain-specific metadata passthrough (not computed) ───────────────────
DOMAIN_METADATA = ["weed_density"]

OUTPUT_COLUMNS = (IDENTITY + CLUSTER + COVERAGE + INSTANCES + CLASSES + EXPOSURE + FOCUS
                  + QUALITY_PANEL + WHITE_BALANCE + LEARNED_IQA + COMPOSITE
                  + CAMERA_META + DOMAIN_METADATA)


def blank_row() -> dict:
    """A fresh output row with every column present and empty."""
    return {c: "" for c in OUTPUT_COLUMNS}
