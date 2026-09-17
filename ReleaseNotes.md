# MagniData: Feature-First Visual Analysis for Global Dataset Understanding in Computer Vision

*Official release notes and technical report · Precision AI · September 2026*

MagniData is an image-dataset exploration tool for teams that need to understand what
is inside a large visual dataset before they train, label, clean, or ship with it. It
turns images into a dashboard-ready feature table, then makes those features usable as
parallel coordinates, 3D views, histograms, table filters, and image previews.

Parallel coordinates is the main working view: each axis is an extracted signal, and
brushing several axes together becomes a combined feature-extraction query.

<p align="center">
  <img src="assets/preview.png" alt="MagniData Dashboard Preview" width="720"/>
</p>


In data-centric computer vision, a recurring bottleneck is incomplete understanding of the training distribution. Large image collections are often uncategorized, sparsely labeled, or annotated under inconsistent protocols. Under these conditions, architectural changes and additional annotation are frequently undertaken before the dominant visual statistics, coverage gaps, and quality structure of the data are known. The practical questions are straightforward, but sample-centric viewers do not make them easy to answer:

- Which photometric and geometric properties account for most of the observed variance?
- How are object and scene diversity distributed across the collection?
- Is quality degradation, class imbalance, or geometric bias concentrated in identifiable subpopulations?

MagniData is a visual analysis pipeline organized around interpretable, dataset-wide features rather than individual samples. It couples three coordinated views: (i) parallel coordinates for interactive multivariate and compound querying; (ii) a three-dimensional multiparametric or embedding view with clustering, to expose global structure and outliers; and (iii) a table for exact filtering and image, mask, and overlay inspection. Feature construction is automatic. From images, the system derives color, exposure, sharpness, noise, compression, and learned image-quality measurements. When instance masks or boxes are available, it also computes geometry-aware quantities such as annotation coverage, relative object area, aspect ratio, instance count, overlap, and spatial layout. The result is a global view of both what the dataset contains and how its important properties co-vary.

This feature-first workflow differs from other instance-centric tools because the feature space is the starting point. Many engineered signals are visible together, can be brushed in combination, and remain linked to the 3D structure, distributions, and underlying images. MagniData therefore turns feature extraction into an interactive method for understanding the collection, rather than treating global analysis as a separately configured extension of sample browsing.

## Motivation

Standard inspection tools are useful for examining individual images, labels, tracks, or embedding neighborhoods. They are less effective as a default interface for seeing compound regularities—for example, the co-occurrence of low quality, small object scale, and a restricted hue distribution. Without that global structure, it is difficult to decide whether underperformance is attributable to model capacity, annotation coverage, or a mismatch between the labeled slice and the unlabeled remainder. MagniData treats that mismatch as an empirical object: selections in feature space update the 3D view, distributions, and table together, so underrepresented regions can be measured rather than inferred from anecdotal browsing.

### Two operating regimes are distinguished.

- *Unlabeled or weakly labeled collections:* Analysis relies on photometric statistics, quality measures, resolution, and optional embeddings. The intended use is to identify tails, near-duplicates, and visually homogeneous regimes.

- *Partially annotated collections:* The same axes are extended with mask- or box-derived geometry and, where present, class composition. The intended use is to test whether the annotated subset is representative of the full set.

Proposed masks, when used, are treated as derived features for exploration. They are not presented as ground truth.

## Relation to instance-centric tools

Instance-centric tools are optimized for tasks such as frame inspection, label and prediction review, multimodal playback, similarity search, and dataset engineering. They can also expose custom fields, histograms, embeddings, and linked plots. The difference is the default analytical priority. In MagniData, parallel coordinates are the primary query surface: many engineered variables can be brushed at once, while the effect on global geometry, distributions, and selected samples is immediate. MagniData organizes the workflow around simultaneous, interpretable, multivariate constraints and a structural view of the dataset.

The two classes of tool are complementary. MagniData is aimed at early dataset characterization, global coverage analysis, and subset construction. Instance-centric tools remain appropriate for detailed label validation, temporal alignment, neighborhood search, and downstream pipeline integration.

## An analysis step

A typical query combines a quality threshold, an object-area ratio, and a photometric band. The user brushes one or more axes to build a combined feature query. The 3D view reveals the location and structure of the resulting subpopulation; the distributions show how it differs from the full dataset; and the table provides exact rows and previews. The subset can be exported for targeted labeling or reserved as a robustness split. The same mechanism applies to any conjunction of computed axes.

## Methodological constraints

Engineered features support explanation; embeddings support neighborhood structure. Both should be available, with engineered quantities on the parallel-coordinate axes and embeddings (or a joint projection) in the spatial view. Parallel coordinates scale poorly with undifferentiated high-dimensional input; axes therefore require grouping, ranking by variance or association with the current selection, and optional suggested brushes. Clustering over heterogeneous units (photometry, quality, geometry) requires explicit normalization and a means of attributing cluster membership to a small set of driving dimensions. These are interface and estimation problems, not afterthoughts.

## MagniData Offerings

- Find dataset outliers, low-quality images, and difficult samples faster.
- Use parallel-coordinate brushes as a human-readable feature extraction layer.
- Compare multiparametric features with embedding-space structure.
- Build and save curated subsets for training, review, or downstream analysis.
- Bring your own images, build demo datasets, or load an existing feature CSV.


## Scope

MagniData does not replace annotation tools, training frameworks, or embedding-based retrieval. It does not, by itself, establish that additional model capacity is unnecessary. Its function is focused: to make the distributional properties of raw and partially labeled image data globally observable, so subsequent modeling and labeling decisions can be justified from measured structure rather than unexamined assumptions.

## Benchmark

The current reference environment is a WSL2 Linux instance on an AMD Ryzen AI 9 365
(10 cores / 20 threads), with 15 GiB of system memory and an NVIDIA GeForce RTX 5070
Laptop GPU with 8 GiB of VRAM. The API and web containers share the host's 15 GiB
container memory limit. These results were collected on 17 September 2026 using the
47,281-row `Stress-47k` dataset, with a 19.0 MB feature CSV and a 1.1 GB, 1024-dimensional
embedding sidecar.

### Rendering architecture

Graph3D uses Three.js `WebGLRenderer` with custom shader materials, `Float32Array`
buffers, and an `EffectComposer` bloom pass. The 47k points are therefore submitted to
the browser's WebGL implementation and can be rendered by the GPU when the browser has
hardware-accelerated WebGL enabled. The Docker web container does not render the scene
and does not use CUDA. CSV parsing, JavaScript object creation, client-side
multiparametric PCA, and interaction bookkeeping run on the CPU. Embedding projection
and embedding-space clustering run in the API's Python process on the CPU; changing
those algorithms to GPU implementations would require a separate CUDA-compatible
backend and would not accelerate the browser's WebGL drawing itself.

### Measured timings and memory

| Stage | Measured result |
|---|---:|
| CSV download from the API | 0.042 s, 19.9 MB |
| Browser-equivalent CSV parse (Node/Papa Parse, 42 columns) | 0.618 s; approximately 157 MB additional heap |
| Cold embedding-matrix load after API restart | approximately 15 s; streams and parses the 1.1 GB JSON sidecar |
| Precomputed PCA response | 0.073 s, 2.95 MB |
| Precomputed t-SNE response | 0.066 s, 2.79 MB |
| Precomputed LLE response | 0.082 s, 3.30 MB |
| Uncached LLE computation | 13.27 s |
| Uncached spherical K-means, `k=21` | 13.40 s |
| API container after embedding work | approximately 630 MiB resident |

The projection timings include the API request and response but not browser parsing,
GPU buffer upload, animation, or the first painted frame. An end-to-end browser paint
time and FPS measurement was not collected in this WSL session because no browser
automation or WebGL profiler is installed; those should be measured separately with the
target browser and hardware. The timings nevertheless identify the present bottlenecks:
the initial CSV parse and JavaScript row representation consume substantially more
memory than the point buffers, while the first embedding-matrix load and uncached LLE
or K-means dominate compute latency. The 1.1 GB embedding JSON is parsed by the API,
not downloaded by the browser, and the current streaming parser reports no intermediate
progress during that phase. Once the matrix and result are cached, the API response is
effectively sub-second; repeated K changes are primarily limited by whether a new
clustering result must be computed.

## Usage

```bash
git clone git@github.com:Precision-AI-Inc/magnidata.git
cd magnidata
docker compose up --build
```

Open the dashboard at `http://localhost:5175`.

From the landing screen, open MagniData and choose a prepared demo, build from a local
folder under `image_sets/`, upload your own images, or load a CSV that follows the
MagniData data contract.


### Feature Extraction

```bash
python -m precisionai.magnidata.tools.features --input images.csv --output features.csv
```


### Feature Format

```csv
image_path,annotation_ratio,blur_laplacian,noise_sigma,colorfulness,complexity_score,camera_angle
image_sets/field/images/img_001.jpg,0.421,318.7,4.8,27.1,0.63,oriented
```

Minimal requirements are Docker for the full application or Python for the feature
extractor. Embeddings are optional; when present, they unlock the embedding-space 3D
view.

## Contact and collaboration

For bug reports, feature requests, and collaboration on MagniData, contact the main
project collaborator, Reinier, at [reinier@precision.ai](mailto:reinier@precision.ai).
