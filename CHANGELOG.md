# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- The trainer has 16 GB of shared memory instead of 1 GB, so full-size reference training (for example LR-ASPP
  at 798×532) no longer hangs when its data-loader workers run out of room for prefetched batches.
- Run detail and run compare plot every metric of a run instead of only the first 20 keys, which left most
  charts of an AgriTune run (about 140 per-class metrics) empty.
- *Input resolution: Original size* no longer refuses datasets with more than one image size (SEEDPLANT mixes
  798×532 and 810×540 tiles). It trains at the size most images share: AgriTune still encodes every image as it
  is and resizes the other masks to that size (as AgriTune's own `geometric.resize` does), the reference trainer
  resizes their images and masks, and the run log says how many were resized.
- AgriTune runs launched from the portal use the hosted encoder (`https://embeddings.precision.ai/v1`) by default,
  with the key read from AgriTune's own `../agritune/.env`, instead of silently falling back to the fake encoder,
  whose features cannot teach the decoder anything but the most common class. The run log names the encoder in use;
  `AGRITUNE_ENCODER_BASE_URL=` (empty) still selects the offline fake encoder.
- AgriTune runs prepare their data in parallel (one process per spare core, at most 8) and log progress while they scan
  mask colours and stage resized images, instead of working through a large dataset in one process for hours with an
  empty log. The worker also starts trainers with unbuffered output, so a failure such as a full disk is recorded in
  the run log rather than ending as a bare *trainer exited with code 120*.
- Building a dataset from a server folder or a browser upload now extracts features with the same worker pool as the
  AgriStress-500 build (up to 6 processes, bounded by CPU and memory) instead of a single process, so large folders
  build several times faster.
- A dataset build no longer shows *Lost track of the build — Failed to fetch* after one dropped status request while
  it keeps running on the server: the portal retries with backoff for about three minutes, keeping the last progress,
  and says plainly when an API restart has lost the build.
- The User Guide's AgriTune setup adds the AgriTune file to an existing `COMPOSE_FILE` instead of appending a second
  line that silently dropped the GPU file. The trainer's *AgriTune is required for this preset* error now names the
  missing AgriTune module.
- AgriTune runs now train on the device named by `MAGNIDATA_TRAIN_DEVICE` (default `auto`, so the GPU when one is
  available) instead of always using the CPU, and fail with a clear message when `cuda` is requested without a GPU.
- The Train dialog no longer sends reference-only settings for AgriTune presets, which the API rejected with a 400.
- Cancelling an AgriTune run stops it within seconds instead of at the end of the current epoch.

### Added

- AgriTune ASPP, Pyramid Pooling, Segmenter and MaskFormer presets, so the portal offers all six AgriTune decoders,
  with each decoder's options (atrous rates, pool sizes, attention heads, MLP ratio, mask queries, widths and
  layers) under the Train dialog's advanced Decoder section; left empty, they keep that decoder's defaults.
- The Train dialog has an *Advanced AgriTune settings* section exposing AgriTune's whole training configuration:
  optimizer, learning-rate schedule, precision, early stopping, checkpointing, class weights, decoder options, image
  and feature augmentation, and data loading. Each setting is declared once (`training/agritune_settings.py`); the
  API validates it, the dialog renders it, and the trainer applies it to AgriTune's config.
- The reference trainer (Tiny UNet, LR-ASPP) can train at the images' own size with *Input resolution: Original
  size* (`input_resolution: native`) instead of a square `image_size`; every image must share one size, and the
  checkpoint records it so on-demand inference resizes to the same width and height.
- AgriTune runs can encode images at their original size (*Encoder input: Original image size*, `input_resolution:
  native`) instead of squaring them to the image size, which for 798×532 tiles gives the encoder a 57×38 patch grid
  instead of 18×18; masks are then trained at their own size. They can also read a feature store already built by
  `agritune features build` in place (`AGRITUNE_EXISTING_MANIFEST`/`AGRITUNE_EXISTING_STORE` on the trainer) when
  every image matches that store's manifest, keeping its sample IDs and split. New settings: gradient accumulation
  and clipping (AgriTune) and weight decay (both engines); the batch size limit is now 256.
- AgriTune presets default to a new *Encode once, then train from the store* feature provider (`cached`): every
  image is encoded into the feature store before training, skipped on later runs over the same images, and the
  features and masks can be preloaded into RAM or GPU memory (`feature_preload`, half precision, shuffled each
  epoch), so epochs no longer wait on per-batch feature reads. The API rejects preload without `cached` and
  augmentation with it; choosing another provider turns preload off.
- AgriTune is now an optional portal training engine. Presets carry an `engine` (`reference` or `agritune`); the new
  MLP Probe and TokenFPN presets expose a frozen-feature provider, augmentation and loss objective, and the existing
  worker dispatches the same MLflow queue to the engine named by the run's `magnidata.engine` tag, so progress, losses,
  split and per-image files, hard-sample predictions, checkpoints and provenance all arrive through the usual contract.
  Enable it by cloning AgriTune beside this repository and adding `docker-compose.agritune.yml` to `COMPOSE_FILE`; the
  default build is unchanged. The Train dialog adapts to the selected engine and becomes a phone-width bottom sheet. A
  new User Guide chapter, **Train with AgriTune**, covers setup, the offline fake encoder, the hosted encoder,
  the settings and troubleshooting, with screenshots from a real run. AgriTune reports every metric once per epoch
  (including the loss and learning rate) instead of once per optimizer step, so MLflow stays small.
- The API rejects training settings the chosen engine cannot honour (for example `loss_terms` or `split_mode: column` on
  an AgriTune preset, or `feature_provider` on a reference preset) instead of silently ignoring them, and refuses
  on-demand inference for AgriTune runs with an explanation.
- The User Guide gained a chapter, **Add a model or a metric**, explaining how extensions reach the UI: registering a model
  preset and a per-image loss term (which become Train dialog options and Explorer axes), logging an epoch metric (which
  becomes a chart and a column choice), and logging runs from a trainer of your own with plain MLflow. Each worked example
  was run through the real trainer, with screenshots from the portal, and the chapter lists the limits and troubleshooting.
- The guide gained two chapters: **Bring your own data** (a folder of images and masks via Prepare From Server Folder, the
  browser upload and its limits, an existing CSV, and a permanent catalog entry) and **Add embeddings** (what they enable,
  computing them with the embeddings tool, the file format, ways to attach them, inheritance, and comparing models).
- **Help & Setup** is now the **User Guide**: a self-contained, chaptered guide from `docker compose up` to a curated, retrained dataset
  on AgriStress-500: how the services and data fit together, installing and resetting the stack, loading the demo,
  partitioning from the command line, the workspace, the Visual Explorer, training, and closing the loop, plus a
  reference and troubleshooting chapter. It has copyable commands, an inline architecture diagram, "what you should
  see" checks and 19 screenshots from the real UI (`dashboard/scripts/capture-guide.mjs` regenerates them).
- `python -m precisionai.magnidata.scripts.partition_dataset` tiles a dataset into fixed-size crops, splits them into
  balanced sub-datasets (by source image by default), builds each with the portal's server-folder pipeline and
  registers it under the source in the lineage graph. It supports `--dry-run`, `--limit-images` and safe re-runs.
- `POST /api/datasets/<id>/parents` replaces a dataset's lineage parents (validated, cycle-checked).
- Datasets can now be created from the Visual Explorer's current selection (**Save as dataset**, next to Export
  selection) and by merging datasets in the workspace: Shift/Ctrl/Cmd-click several lineage cards, name the result,
  and `POST /api/datasets/merge` writes the de-duplicated union with every source as a parent. Both appear in the
  lineage graph with no experiments.
- Lineage cards carry an experiment badge beside the dataset icon: yellow (pulsing) while a run is queued or
  training, gray otherwise, with run counts in its tooltip. `GET /api/datasets/experiment-counts` aggregates MLflow
  runs per dataset in one request and returns `{}` while MLflow is unreachable.
- The Workspace and Datasets pages are now one lineage workspace: a storage-wide summary strip, the interactive
  lineage graph, and a detail panel for the selected dataset with statistics, sample images, annotation coverage,
  recent runs, and lineage links. The panel's **Open in Visual Explorer** and **View experiments** actions open the
  explorer or the Experiments view filtered to that dataset. The selection highlights its ancestors and descendants,
  can be driven with the keyboard (arrows, Enter, Escape), and survives a reload.
- `GET /api/datasets/summary` returns per-dataset row and column counts, numeric distributions, sample images,
  mask coverage, and partition counts; configured datasets now report `row_count` in `GET /api/datasets`.
- Dataset lineage is now an interactive, read-only canvas with drag panning, wheel and pinch zoom, fit-view controls,
  and a pannable minimap. Its deterministic left-to-right layout expands for deep or branching dataset histories while
  preserving dataset loading and deletion actions.
- A trainable **Cloud Dataset** now combines AgriStress500-mini1 and mini2 and appears with all derived datasets in
  a selectable lineage graph. Filtered samples can create a top-K feature-similar complement from Cloud, optionally
  applying image removals staged directly in the preview; complemented datasets retain both parent edges while
  removal-only datasets retain the current dataset as their sole parent.
- The dashboard remembers the latest server dataset and restores it after refresh, with brief retries for a busy
  API and a recoverable error screen if loading still fails. Large parallel-coordinate plots render a deterministic
  3,000-line representation while brushes, counts, tables, exports, and selections continue to use every row.
  Training status now distinguishes data preparation, mask indexing, GPU batches, validation, and scoring, with
  coarse percentages refreshed every two seconds.
- A calmer Material-style dashboard shell with floating navigation, a breadcrumb context header, at-a-glance
  workspace cards, focused recent datasets and experiments, compact task shortcuts, and collapsible service
  readiness. Dataset views now live beneath **Visual Explorer** in the left navigation, while the connected
  parallel-coordinates, 3D, and results workspace keeps the full available height. Training, run overlays,
  appearance, field selection, exports and curation actions are organized in a collapsible right setup panel;
  dataset identity and sample counts sit in the page header beside the title. Responsive and dark-theme behavior
  are preserved.
- Missing model predictions can be generated from the image preview. The API queues the request for the trainer,
  which loads the run's latest checkpoint and stores the resulting mask with the MLflow run; the modal follows
  progress, displays the result when ready, and reuses that artifact the next time the sample is opened.
- The train/val partition shows across the dashboard: an overlay that brings a split adds `split` as the first
  categorical axis (listed under **Training** in Fields) and, unless colouring was chosen by hand, colours by it with
  fixed train/val/test colours in Parallel Coordinates, the 3D view and histograms. Histograms can **stack by split**,
  the 3D view can group by any categorical column with 12 or fewer values (`split` first), the data table pins `split`
  beside the image column with filter chips, and the toolbar shows `train 355 · val 96`.
- The Run picker lists every run linked to the dataset, newest first, with status, device, scored epochs and a reason
  when a run cannot be picked yet ("queued", "training · first losses after epoch 1 scoring", "split only"); it refreshes
  every 10 s while a run is queued or training, when opened, and right after launching. A launched (or newest running)
  run is overlaid automatically once it has a split or losses, and Parallel Coordinates shows `loss_total`,
  `loss_total_mean` and `hardness` next to the photometric axes. A **High-loss samples** button in the Fields panel's
  Training group brushes `hardness ≥ 0.6` (or the top fifth of mean loss).
- Per-sample data from the start of a run: the trainer tags `magnidata.has_split` as soon as `split.csv` is logged and
  always scores epoch 1 and the last epoch, whatever `score_every` is. Run summaries gain `has_split`;
  `GET /api/runs` takes `per_sample=any|split|scored`; `GET /api/runs/<id>/columns` answers `image_name,split`
  (header `X-Magnidata-Scored-Epochs: 0`) before the first scored epoch instead of 409.
- Faster training input: images and masks are decoded once, resized to the training size and cached under
  `data_user/.train_cache/` (bounded by `TRAIN_CACHE_MAX_MB`, default 4096, least recently used evicted first); the
  palette is built from that cache, loaders use parallel worker processes (`num_workers`, default one per spare core
  up to 8, `shm_size: 1gb` for the trainer container), pinned memory and, on CUDA, mixed precision (`amp`, bfloat16
  where supported; per-sample losses are still computed in float32). Both are launch parameters in the Train dialog.
  On AgriStress-500 at 256 px on an RTX 5070 an epoch drops from ~31 min on CPU to ~2 s once the cache is warm.
- GPU training: `docker-compose.gpu.yml` builds the trainer with CUDA torch and reserves the NVIDIA device; enable it
  with `COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml` in the git-ignored `.env`. torch 2.14.0 is only
  published on the cu130 wheel index (cu128 stops at 2.11), so the override uses cu130 (driver CUDA 13+, RTX 50xx
  needs cu128 or newer). `MAGNIDATA_TRAIN_DEVICE=cuda` makes a missing GPU fail the run instead of silently
  training on the CPU. The trainer now prints the device and per-batch/epoch/scoring progress (flushed, uploaded
  to MLflow every 30 s), logs `perf/epoch_seconds` and `perf/samples_per_sec`, and sets the `magnidata.device` and
  `magnidata.eta_seconds` tags, shown as a GPU/CPU badge and ETA in the Experiments table and run detail.

- Close the loop: after saving a curated dataset, **Train on it** opens the Train dialog with the overlaid run's
  settings and records the run as its parent (shown as lineage in run detail and an optional "Parent run" column).
  **Export relabel queue** downloads the selected images with their losses, hardness and notes, hardest first.
  **Compare on dataset** (two runs in the compare view) overlays each image's hardness and loss change between
  the runs, drawn on a diverging scale centred on zero. New API: `GET /api/runs/compare-samples`.
- Epoch scrubber for an overlaid run (slider, "latest" toggle, play) that re-colours the dataset views as you move
  through the scored epochs. The image preview gains a Training section with the image's loss curves per term,
  its hardness and rank, and a Prediction view of the model's mask at the viewed epoch (only the hardest
  samples of each epoch have one; the view falls back to the nearest earlier epoch). New API:
  `GET /api/runs/<id>/samples/<image>` and `GET /api/runs/<id>/prediction`.
- Train from the dashboard: a **Train** button (tab bar and Experiments tab) opens a dialog to launch a run on the
  loaded dataset, with the preset's tunable parameters validated against the server's bounds. The Header shows
  "Training · epoch 3/20" while it runs, a toast reports the outcome, run detail can cancel it, and an overlaid
  run's columns refresh as each epoch is scored. Train is disabled, with the reason, when MLflow or the trainer
  worker is unavailable or another run is active.
- Overlay a training run on a dataset: pick a run in the new **Run** dropdown (or "Overlay on dataset" in the
  Experiments tab) and its per-sample losses, ranks, trends and `hardness` become regular columns in the
  parallel coordinates, 3D colour-by, charts, table and Schema tab, grouped under "Training". A run that was
  not launched from MagniData can be linked to the open dataset after checking how many images match.
- Compare 2–10 runs in the Experiments tab: overlaid metric curves, a final/best summary with the better
  run highlighted, and a params table that shows only the settings that differ. The comparison is kept in
  the URL, so a reload restores it.
- Run detail panel in the Experiments tab: loss curves with train and val overlaid per term, mIoU/IoU and
  learning-rate charts (step or relative time, smoothing, log scale, per-group toggles), searchable params
  and tags, an artifact tree with image previews, and live updates while a run is scheduled or running.
- Experiments tab (also reachable from the header without a dataset): browse MLflow experiments and runs
  with status, dataset and name/filter controls, sortable metric and param columns you can choose, "Load more"
  paging, links into the MLflow UI, and a banner when MLflow is unreachable.
- `trainer` service in `docker-compose.yml` (`Dockerfile.trainer`, CPU torch by default, optional CUDA build
  arg): a worker that claims `SCHEDULED` runs launched from the dashboard, trains them in a subprocess and
  fails runs a crashed worker left behind. `docker compose up --build` now starts the portal, API, MLflow
  and the trainer.
- Launch and cancel training through the API: `GET /api/training/presets` (presets, bounds, trainer
  and MLflow availability), `POST /api/training/runs` (validated request becomes a `SCHEDULED` MLflow
  run that a trainer worker claims; one UI-launched run at a time) and `POST /api/runs/<id>/cancel`.
  The trainer also gained a `split_mode` parameter (`random` or `column`).
- `GET /api/runs/<id>/columns` serves per-sample loss axes derived from a run's MLflow artifacts: the
  loss at an epoch plus its mean, improvement, trend and within-split rank over the scored epochs, and a
  `hardness` score for samples that stay in the worst fifth. `GET /api/runs/<id>/match` reports how
  many of a run's images exist in a dataset. Downloaded artifacts are cached under
  `data_user/.mlflow_cache` (bounded by `MLFLOW_CACHE_MAX_MB`, default 2048).
- API endpoints to browse MLflow: `GET /api/experiments`, `GET /api/runs` (filter by experiment, dataset,
  status or an allow-listed MLflow filter), run detail, metric histories (downsampled), multi-run metrics
  for comparison, artifact listing and download, and `POST /api/runs/<id>/link` to attach an external run
  to a dataset.
- Reference segmentation trainer (`python -m precisionai.magnidata.training.train`): trains a small
  U-Net or LRASPP model on a MagniData dataset with masks and logs params, loss/mIoU curves,
  per-sample loss files, hardest-sample predictions and a checkpoint to an MLflow run. Install its
  dependencies with `pip install -e ".[training]"`.
- `LossRecorder` and `python -m precisionai.magnidata.training.attach`: record per-sample loss terms
  per epoch in the training-contract CSV format and log them to an MLflow run, including for runs
  that already exist. Torch-free, so external trainers can use the same writer.
- Local MLflow tracking server (`mlflow` service in `docker-compose.yml`, UI on :5052, data in the
  `magnidata-mlflow` volume) and an API adapter around it. `GET /api/mlflow/status` reports
  whether the server is reachable.
- Training contract (`precisionai.magnidata.training.contract`): the MLflow tags, metric keys,
  artifact layout and per-sample loss columns shared by the trainer, API and external trainers.
  Documented in `docs/training-contract.md`. Imports without torch or mlflow.
- PCA and t-SNE are now computed while a dataset is being prepared, not the first time
  its 3D view is opened. Preparation writes a `<stem>_projections.json` sidecar holding
  one 3D position per CSV row, and the viewer renders those points directly. On a
  47k-row dataset with 1024-d embeddings this takes the first PCA view from 27 s to
  0.8 s and makes t-SNE usable at all (it needs ~4 minutes to compute). Datasets without
  a sidecar — and the LLE projection, which stays on-demand — are unaffected: the API
  still computes those per request. A sidecar whose row count no longer matches its CSV
  is ignored rather than trusted.
- Build a dataset from a server-side folder in `image_sets/`: drop a folder holding
  `images/` (and optionally matching COCO `labels/`) on the host, and prepare it from
  the portal's "Bring Your Own Data" dialog. The folders are indexed by
  `GET /api/datasets/build/local-folders`, and the build runs in place — no upload, no
  copy of the image data, and none of the browser-upload size caps.
- Initial release: the `precisionai.magnidata` visual-query dashboard — a deterministic
  feature-extraction/embeddings toolkit (`tools/`), a Flask API (`api/`), a React/Vite/
  Three.js dashboard (`dashboard/`), and Docker Compose deployment.
- `docker compose up` now prints where to open the tool (`portal -> http://localhost:5175`,
  `api -> http://localhost:5051/api/health`) in the API's startup log, driven by the
  `MAGNIDATA_PORTAL_URL` / `MAGNIDATA_API_URL` variables in `docker-compose.yml`.

### Fixed

- An embeddings file uploaded with a new dataset never matched its images: uploads are renamed when staged
  (`<name>-<hash>.png`) but the file was stored keyed by the original names, so the Embeddings view had nothing to
  draw. The portal now re-keys the vectors to the staged names and rejects a file that matches no image.
- A dataset created in the portal (server-folder build, upload, tiles) had no way to gain embeddings afterwards. It now
  picks up a `<csv stem>.json` placed beside its CSV, the same rule configured datasets already follow.
- The image preview's ground-truth caption was unreadable (white text on a light bar); lineage card names now wrap
  to two lines instead of truncating, and selecting a dataset frames its whole lineage instead of centring on it.
- The image preview's **Prediction** tab showed the dataset's ground-truth overlay whenever the run had no saved
  prediction for that image, so ground truth could be mistaken for model output. Each tab now maps to its own
  source with no fallback (a missing prediction shows "No saved inference" and the Run inference button), and the
  Mask, Overlay and Prediction views carry a caption naming their source.
- Dataset cards in the lineage graph could not be clicked, so datasets could neither be opened in the Visual
  Explorer nor deleted: React Flow disabled pointer events on its non-interactive nodes.

### Changed

- Renamed the project to MagniData throughout: the distribution is now `magnidata`
  (was `pai-dataviz`), the import path is `precisionai.magnidata` (was
  `precisionai.dataviz`), and the Compose project/containers are `magnidata-*`. The
  `dataviz-features/…` and `dataviz-embeddings/…` feature-schema identifiers are
  deliberately unchanged — they are a data contract recorded in provenance sidecars, so
  renaming them would misreport the schema version of already-generated datasets.
- Human-facing text now spells the product "MagniData" everywhere (READMEs, docs, CLI
  help, browser tab title "Precision AI - MagniData"); package and import paths stay
  lowercase `magnidata`. The README and `pyproject.toml` descriptions now match, and the
  root README no longer shows PyPI/Python-version badges.
- Landing page cards are wider, and the light- and dark-theme MagniData artwork now share
  one crop and scale, so both themes show the same logo at the same size.
- AgriStress-500 demo progress reads "Downloading AgriStress-500…" next to the overall
  percentage, instead of a per-image counter beside a different percentage.
- Inline errors in the Bring Your Own Data dialog are larger, bold, and boxed with an icon.
- Bumped Pillow to 12.3.0, torch to 2.14.0, torchvision to 0.29.0 and tqdm to 4.70.0
  (Dependabot #9–#12), and brought the bundled `precisionai/magnidata/requirements.txt`
  and the recipe's CPU install command in line. Pillow 12.3 slightly shifts some
  pixel/COCO feature values (on a 201-image test set, `annotated_px_count` by up to
  125 px and `mean_pairwise_color_dist` by up to 0.012), so datasets built before and
  after this change are not byte-comparable; the NIMA/NIQE/BRISQUE columns are unchanged.
- Releases are published only as GitHub Releases, with the sdist and wheel attached: the
  release workflow no longer has a PyPI publishing job (it failed on every tag, since the
  package isn't registered on PyPI).

### Fixed

- Resized training-cache eviction now uses the main cache entry's access time instead of the newer timestamp from
  its immutable colour-count companion, so least-recently-used ordering is deterministic and retains recent entries.
- Loading a dataset could fail in the browser with a bare "Failed to fetch", console-only
  and with no indication of the cause. `/api/dataset/csv` and `/api/dataset/embeddings`
  serve a dataset's CSV/embeddings from a path that can be rebuilt in place (re-running a
  demo build, regenerating a BYOD dataset, or — as happened repeatedly during this
  session — replacing a stress-test dataset's files while iterating), and Werkzeug's
  `send_file` defaults to conditional responses: an ETag/Last-Modified pair plus Range
  support, cached by the browser under `Cache-Control: no-cache`. If the browser had
  cached or partially cached that URL from before the file was rebuilt, it could later
  issue its own Range "resume" request against bytes that no longer matched, which
  Chrome surfaces as `net::ERR_FAILED` right after a `206 Partial Content` — exactly the
  failure this reproduced. Both endpoints now serve with `conditional=False` and
  `Cache-Control: no-store`, so the response is never cached, range-resumed, or
  revalidated by the client; a Range header is simply ignored and the current file is
  always returned in full. Existing browser tabs still hold their old cache until a hard
  reload.
- Only one dataset build can run at a time, but the UI didn't say so: the Bring Your Own
  Data dialog now names the dataset being prepared and disables Prepare/Create until it
  finishes, a rejected request (HTTP 409) names the running build, and a rejected start no
  longer briefly replaces the running build's progress badge with an error.
- `SECURITY.md` listed the wrong report address and a template package name; it now points
  to reinier@precision.ai and `magnidata`.
- Portal served a blank page after a rebuild: `index.html` is now sent with
  `Cache-Control: no-cache` and missing `/assets/` bundles return 404 instead of
  falling back to `index.html`, which browsers refused to execute as a module script.
- The API could be OOM-killed by a large embeddings file, taking the whole 3D view down
  with a "failed to fetch". Two causes: the embeddings JSON was read with `json.load`,
  which costs several times the file size once every number is a Python float in a list
  (a 1 GiB file peaked at 2.9 GB); and `_base_matrix` only populated its cache after
  parsing, so every request arriving during a parse started its own, making peak memory
  scale with how many happened to overlap. The file is now streamed into the matrix one
  entry at a time (same 1 GiB file: 467 MB peak), and concurrent requests for the same
  dataset share a single parse.

[Unreleased]: https://github.com/Precision-AI-Inc/magnidata/compare/0.0.3...HEAD
