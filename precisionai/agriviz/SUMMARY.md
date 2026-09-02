# Feature Extraction Summary

Reference for the feature set that should drive dataset filtering/curation. Every feature is tagged with a tier:

- **Atomic** — deterministic pixel/metadata math, no model inference required.
- **Model-based** — requires running a trained/pretrained model (segmentation, embedding, learned IQA).
- **Compound** — derived by combining two or more atomic and/or model-based features into a new signal.

Features marked *(existing)* already exist in `precisionai/agriviz/tools/`. Features marked *(new)* still need to be built.

---

## Data Integrity Gates

These should run first, since a bad verdict here corrupts every color-dependent feature downstream.

### Has Annotations
**Tier:** Atomic *(new)*
**Source:** check at ingestion time whether a COCO annotation/mask file exists for the image, using the same sibling-file lookup convention `coco_labels.py` already relies on.
**Answers:** whether this image has ground-truth labels at all. Gates which annotation-dependent features (class breakdown, `green_annotation_ratio`, instance stats) are valid versus which fall back to model-predicted equivalents.

### Channel-Order Signals (RGB/BGR swap)
**Tier:** Atomic *(existing — `rb_swap_check.py`, not yet wired into the pipeline)*
**Source:** `median(R-B)` over soil pixels, `median(R-B)` over vegetation pixels, `frac(R>B)` over all pixels — segmented via the swap-invariant `ExG = 2G - R - B`.
**Answers:** whether an image's color channels were likely swapped during the historical `.raw.tif → .png` conversion. Soil is the primary, most reliable signal (brown ground should have R>B); vegetation is a secondary vote, demoted because it legitimately flips sign in dusk/blue-heavy white balance.

### Trained Swap Classifier
**Tier:** Model-based *(proposed, not built)*
**Source:** logistic regression (or similarly small model) trained on real images plus synthetically R/B-swapped copies, using the color-stat features above as inputs.
**Answers:** the same question as the heuristic above, but self-calibrated to this dataset's actual soil/vegetation palette instead of a hand-picked global threshold — higher precision on borderline/ambiguous cases.

### Channel-Order Verdict (gate)
**Tier:** Compound *(existing detector, remediation missing)*
**Source:** combines soil/vegetation `median(R-B)`, `frac(R>B)`, and optionally the trained classifier and/or embedding-space outlier status as corroboration.
**Answers:** swapped / likely-swapped / ambiguous / likely-ok / ok. Drives whether an image gets auto-corrected, excluded, or flagged for manual review before any other color feature computes.

---

## Vegetation & Segmentation

### Vegetation Indices (ExG, VARI, GLI, NGRDI)
**Tier:** Atomic *(new)*
**Source:** pixel math directly on RGB, no model or labels required.
**Answers:** a cheap, label-free proxy for "how much of this image is plant vs. background" — usable as a fallback segmenter or a sanity check against the model-based mask.

### Crop/Weed/Soil Segmentation Mask
**Tier:** Model-based *(new — user-supplied model)*
**Source:** the crop/weed classification-for-segmentation model, run per image; mask stored as a sidecar file, same pattern as COCO annotation sidecars today.
**Answers:** per-pixel class (crop / weed / soil-background), the foundation every vegetation-related compound feature below is built from — including on images with no ground-truth annotations.

### Green-on-Brown Ratio (Model-Based)
**Tier:** Compound *(new)*
**Source:** (crop-pixels + weed-pixels) / soil-background-pixels, from the segmentation mask above.
**Answers:** vegetation-to-background ratio for *every* image, labeled or not — the general-purpose version of this feature.

### Green-on-Brown Ratio (Annotation-Based)
**Tier:** Compound *(new)*
**Source:** (crop-annotated px + weed-annotated px) / (image area − annotated vegetation px), from COCO ground-truth polygons. Only populated when `has_annotations` is true.
**Answers:** the same ratio computed from human-labeled ground truth instead of model prediction — a trustworthy reference wherever labels exist, and directly comparable to the model-based version to catch model failures or mislabeled data.

### `green_annotation_ratio` / `green_mass`
**Tier:** Atomic, label-dependent *(existing — `pixel_features.py`)*
**Source:** mean pixel coverage/intensity over the COCO foreground mask.
**Answers:** coarse annotated-foreground coverage; superseded in intent by the two green-on-brown ratios above, kept for backward compatibility.

### Weed-Density-vs-Crop Ratio
**Tier:** Compound *(new)*
**Source:** weed-pixel-count vs. crop-pixel-count, same segmentation mask.
**Answers:** "is this a weed-heavy image" — falls out of the mask almost for free.

### Canopy Cover %
**Tier:** Compound *(new)*
**Source:** (crop + weed pixels) / total pixels, same mask.
**Answers:** overall vegetative cover, independent of the crop/weed split.

### Stress / Chlorosis Score
**Tier:** Compound *(future)*
**Source:** vegetation-index drop + leaf hue shift (HSV/Lab histogram) + GLCM texture irregularity.
**Answers:** whether the visible canopy shows signs of nutrient deficiency or disease stress.

---

## Image Quality & Composition

### Focus / Sharpness
**Tier:** Atomic *(existing — `pixel_features.py`, `blur_laplacian`)*
**Answers:** is the image in focus.

### Illumination
**Tier:** Atomic *(existing — `pixel_features.py`: `overexpose_ratio`, `underexpose_ratio`, `shadow_edge_ratio`)*
**Answers:** exposure problems — blown highlights, crushed shadows, harsh shadow edges.

### White Balance
**Tier:** Atomic *(existing — `pixel_features.py`: `wb_r_gain`, `wb_b_gain`, `wb_rb_ratio`, gray-world estimate)*
**Answers:** color-cast/white-balance error, which also matters as a cross-check against the channel-order gate.

### Noise & Signal Quality
**Tier:** Atomic *(existing — `quality_features.py`: `noise_sigma`, `tenengrad`, `rms_contrast`, `dynamic_range`, `luma_entropy`, `colorfulness`)*
**Answers:** general signal quality — grain/noise level, contrast, tonal range, information content, and color vividness.

### GLCM Texture
**Tier:** Atomic *(new)*
**Source:** gray-level co-occurrence matrix contrast/homogeneity/entropy/correlation over the canopy region.
**Answers:** canopy structure and density independent of color — useful where color-based vegetation indices saturate or are ambiguous.

### Color Histograms (HSV/Lab)
**Tier:** Atomic *(new)*
**Answers:** color distribution shape, feeding hue-shift-based stress detection and general dataset diversity analysis.

### Resolution / Aspect Ratio
**Tier:** Atomic *(new)*
**Answers:** basic image geometry — flags inconsistent capture settings across a dataset.

### Learned Image Quality (NIMA, NIQE, BRISQUE)
**Tier:** Model-based *(existing — `nima.py`, via `pyiqa`)*
**Answers:** a learned aesthetic/perceptual-quality assessment that captures issues hand-rolled pixel math misses.

### Complexity Score
**Tier:** Compound *(existing — `features.py`, hand-tuned)*
**Source:** weighted blend of `overlap_ratio`, `interclass_color_sim`, `class_entropy`, exposure, and NIQE/BRISQUE/NIMA (or blur/noise fallback).
**Answers:** a single "how hard/messy is this scene" score. Candidate to reimplement in a declarative atomic/compound registry rather than hard-coded weights.

### Image Usability Gate
**Tier:** Compound *(new)*
**Source:** blur + exposure + noise + resolution, combined into a pass/fail.
**Answers:** should this image be excluded from training entirely on quality grounds, independent of content.

---

## Embedding & Similarity

### DINOv3 Embedding Vector
**Tier:** Model-based *(new)*
**Source:** DINOv3 inference on the raw image, stored once per image (sidecar vector or vector index).
**Answers:** a general-purpose semantic representation every similarity/novelty/dedup feature below is computed from.

### Novelty / Out-of-Distribution Score
**Tier:** Compound *(new)*
**Source:** embedding distance (kNN or centroid) to a reference/training set.
**Answers:** "how different is this image from what I already have" — directly answers hypotheses like "find images unlike my current training set."

### Near-Duplicate Score
**Tier:** Compound *(new)*
**Source:** embedding cosine similarity, optionally paired with a perceptual hash for a two-stage exact+semantic check.
**Answers:** redundant/near-identical images worth deduplicating before training.

### Reverse Image Search ("More Like This")
**Tier:** Compound, query mechanism rather than a static column *(new)*
**Source:** kNN lookup over the DINOv3 embedding index.
**Answers:** "show me more images like this one" — the core interaction for hypothesis-driven exploration, not just a filter.

---

## Classes & Annotation Taxonomy

### Has-Class Flags
**Tier:** Atomic *(new)*
**Source:** `has_<class>` boolean per class in the taxonomy, flattened from COCO categories (one column per class, since a multi-valued list column doesn't work as a parallel-coordinates axis).
**Answers:** which specific classes are present in an image — the raw dimension the current aggregate stats (`class_count`, `class_entropy`) discard.

### Class Area Ratio
**Tier:** Atomic *(new)*
**Source:** `class_area_ratio__<class>` — annotated pixel area of that class / total image area.
**Answers:** richer than presence alone — "images that are >40% weed by area," for example.

### Class Instance Count
**Tier:** Atomic *(new)*
**Source:** `class_instance_count__<class>`.
**Answers:** how many objects of each class appear in the image.

### Dominant Class
**Tier:** Atomic/Compound *(new)*
**Source:** the class with the largest area or instance count.
**Answers:** a single categorical summary column, compatible with existing categorical-column handling in the frontend.

### Predicted Class Breakdown
**Tier:** Model-based *(new)*
**Source:** `pred_has_<class>`, `pred_class_area_ratio__<class>` — same breakdown as above, but from the crop/weed/soil segmentation model's mask instead of COCO annotations.
**Answers:** the same class-membership questions on images with no ground truth at all — extends class-based filtering to unlabeled/incoming imagery.

### Annotation-vs-Prediction Disagreement Score
**Tier:** Compound *(new)*
**Source:** compares ground-truth class breakdown to model-predicted breakdown where both exist.
**Answers:** flags either an annotation error or a model failure case on that image — a QA signal in the same spirit as the channel-order gate.

---

## Instance & Label Aggregates *(existing)*

### Instance Counts & Geometry
**Tier:** Atomic *(existing — `instance_features.py`: `instance_count`, `real_instance_count`, `mean_instance_area_ratio`, `instance_area_std`, `overlap_ratio`)*
**Answers:** how many labeled objects, how large relative to the frame, how much they overlap.

### Class Distribution Aggregates
**Tier:** Atomic *(existing — `class_count`, `smallest_class_ratio`, `class_entropy`)*
**Answers:** how many distinct classes and how balanced they are, without saying which classes.

### Cross-Class Color Similarity
**Tier:** Atomic *(existing — `mean_pairwise_color_dist`, `interclass_color_sim`, via RGB→Lab)*
**Answers:** how visually distinguishable the labeled classes are from each other in this image — relevant to how hard the scene is for a classifier.

---

## Metadata

### Existing Camera/Flight Metadata
**Tier:** Atomic *(existing — `metadata_join.py`: `gsd`, `camera_angle`)*
**Answers:** ground sample distance and capture angle, joined from `images_metadata.csv`.

### Extended EXIF / Flight Metadata
**Tier:** Atomic *(new)*
**Source:** GPS, timestamp (→ time-of-day flag), altitude, camera/sensor model, EXIF `Software` tag.
**Answers:** capture context (lighting conditions via time-of-day, per-flight GSD normalization) and, via the `Software` tag specifically, a provenance signal that can corroborate or resolve the channel-order gate directly instead of guessing from pixel statistics alone.

---

## Pipeline Sequencing

1. **`has_annotations`** and the **channel-order gate** run first — nothing color-dependent should compute on a flagged-swapped image, and every annotation-dependent feature needs to know upfront whether it's even applicable.
2. **Segmentation model** and **DINOv3 embedding** inference run next, producing the model-based atomic outputs (mask, vector).
3. **All compound features** — green-on-brown (both variants), weed-density ratio, canopy cover, novelty/dedup scores, disagreement scores — derive from the atomic and model-based outputs of steps 1–2.
