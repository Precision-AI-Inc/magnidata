# MagniData feature columns

A dataset CSV has one row per image. The columns come from `precisionai/magnidata/tools/schema.py`
(`FEATURE_SCHEMA_VERSION = "dataviz-features/2.1.0"`), in this order. A dataset may hold a subset, or extra columns
such as `mask_path` and `thumbnail_path`. An empty cell means the input was not available, for example no COCO
labels, no camera metadata, or learned-quality models switched off. It does not mean zero.

Columns marked *labels* need COCO polygon labels (`labels/<stem>.json`). Without labels they stay empty.

| Group | Column | Meaning | Higher means |
|---|---|---|---|
| Identity | `image_path` | Image path relative to the data root; use it with `/api/image` | — |
| | `width`, `height` | Image size in pixels | — |
| Cluster | `cluster` | Ground-truth group: the image's folder relative to the dataset's image root | — |
| | `cluster_l2` | First component of `cluster` | — |
| Coverage | `annotation_ratio` | Share of the image covered by labelled objects (*labels*) | more of the image labelled |
| | `annotated_px_count` | Labelled pixels (*labels*) | — |
| | `fg_green_mean` | Mean green-channel value inside the labelled area, 0–255 (*labels*) | brighter foreground |
| | `bg_coverage` | Share of the image that is background (*labels*) | emptier image |
| Instances | `instance_count` | Object polygons, including specks (*labels*) | — |
| | `real_instance_count` | Objects after dropping specks; the count to trust (*labels*) | more objects |
| | `mean_instance_area_ratio` | Mean object area relative to the image (*labels*) | bigger objects |
| | `instance_area_std` | Spread of object sizes (*labels*) | mixed sizes |
| | `overlap_ratio` | How much objects touch or overlap (*labels*) | more entangled, harder |
| Classes | `class_count` | Distinct classes in the image (*labels*) | — |
| | `smallest_class_ratio` | Share of the rarest class (*labels*) | more balanced |
| | `class_entropy` | Evenness of the class mix (*labels*) | more even mix |
| | `mean_pairwise_color_dist` | Colour spread inside single objects (*labels*) | tangled, mixed objects |
| | `interclass_color_sim` | Colour similarity between classes (*labels*) | look-alike classes, harder |
| Exposure | `overexpose_ratio` | Share of blown-out pixels | more washed out |
| | `underexpose_ratio` | Share of crushed dark pixels | more too-dark area |
| | `shadow_edge_ratio` | Share of shadow edges | more shadow lines |
| Focus | `blur_laplacian` | Variance of the Laplacian (unbounded) | sharper |
| | `tenengrad` | Mean squared Sobel gradient (unbounded) | sharper |
| Quality panel | `noise_sigma` | Estimated Gaussian noise | noisier |
| | `rms_contrast` | Standard deviation of luminance, 0–1 | more contrast |
| | `dynamic_range` | 1st–99th percentile luminance range, 0–1 | wider range |
| | `luma_entropy` | Entropy of the luminance histogram, in bits | more tonal detail |
| | `colorfulness` | Hasler–Süsstrunk colourfulness | more vivid |
| White balance | `wb_r_gain`, `wb_b_gain` | Gray-world red and blue gains | stronger correction needed |
| | `wb_rb_ratio` | Red-to-blue gain ratio | colour-cast indicator |
| Learned quality | `nima_ava`, `nima_vgg16_ava` | NIMA aesthetic score (empty unless built with NIMA) | better looking |
| | `niqe` | NIQE no-reference quality (empty unless built with pyiqa) | worse quality |
| | `brisque` | BRISQUE no-reference quality (empty unless built with pyiqa) | worse quality |
| Composite | `complexity_score` | Overall difficulty, 0–1, mixing scene structure and capture quality | harder image |
| | `category` | `complexity_score` bucketed into 1–10 | harder image |
| Camera | `gsd` | Ground sampling distance, from `metadata/images_metadata.csv` | coarser resolution |
| | `camera_angle` | `nadir`, `oriented` or `missing` | — |
| Domain | `domain_metric` | Optional passthrough from the metadata file | — |

Useful rankings:

- Blurriest images: lowest `blur_laplacian` or `tenengrad`.
- Badly exposed images: highest `overexpose_ratio + underexpose_ratio`.
- Hardest images overall: highest `complexity_score`.
- Hardest to segment: highest `overlap_ratio` or `mean_pairwise_color_dist`. These need labels.
- Outliers: images far from their cluster in `/api/embedding/projection`, or alone in a small
  `/api/embedding/clusters` group.

Columns are scored on whatever resolution the build read, so compare images within one dataset rather than across
datasets built differently. A `<csv>.provenance.json` sidecar records the code version and settings of a build.
