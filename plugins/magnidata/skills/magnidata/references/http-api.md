# MagniData HTTP API reference

All routes are under `/api` on the API's base URL: `http://localhost:5051` for the compose stack,
`http://localhost:5050` for `python app.py`, and `https://precisionaiinc-magnidata.hf.space` for the read-only
public demo. Responses are JSON unless noted. Errors are `{"error": "<message>"}` with a 4xx or 5xx status.
There is no authentication. CORS is open.

`source` is always a dataset CSV path relative to the server's data root, as returned by `GET /api/datasets`.
Image paths (`path`, `img`, `mask`) are the values in a dataset's `image_path` and `mask_path` columns.
URL-encode both.

## Service

| Route | Returns |
|---|---|
| `GET /api/health` | `{"status": "ok", "local": {...}}` |
| `GET /api/config` | `{"demo_mode": bool}`. When true, the routes marked **write** below answer `403`. |

## Datasets (read)

| Route | Parameters | Returns |
|---|---|---|
| `GET /api/datasets` | none | List of `{name, description, source, row_count, has_embeddings, parent_sources, deletable, file_size_bytes}`. Datasets created in the app also carry `id` and `parent_source`. |
| `GET /api/datasets/summary` | `source` | `{row_count, column_count, numeric_column_count, numeric_columns: [{name, min, max, mean, histogram}], image_column, sample_images, mask_coverage: {checked, with_masks}, truncated, partitions_committed, has_embeddings}`. `404` unknown dataset, `409` Git LFS pointer. |
| `GET /api/dataset/csv` | `source` | The feature table as `text/csv`, one row per image. `409` when the file is a Git LFS pointer. |
| `GET /api/dataset/embeddings` | `source`, optional `variant` | `{"embeddings": {"<image basename>": [float, ...]}}`. `404` when the dataset has none. |
| `GET /api/dataset/embeddings/variants` | `source` | `[{name, variant}]` for comparison files `<stem>_<variant>.json`. Ignore an entry named `projections`. |
| `GET /api/datasets/demos` | none | `[{key, name, description, enabled}]`, the one-click demo datasets. |
| `GET /api/datasets/build/limits` | none | `{max_images, max_upload_bytes, max_embeddings_bytes, max_request_bytes}` for upload builds. |
| `GET /api/datasets/build/local-folders` | none | `[{folder, image_count, label_count, built}]` for each folder under `image_sets/` that holds a non-empty `images/`. |
| `GET /api/datasets/build/<job_id>` | none | `{status, percent, message, name, dataset, error}`. `status` ends as `done` (then `dataset` holds the new entry) or `error`. |
| `GET /api/partitions` | `source` | `{"partitions": [...]}`, the committed cluster id per image. |
| `GET /api/notes` | optional `stem` | Notes attached to images. |

## Embeddings

| Route | Parameters | Returns |
|---|---|---|
| `GET /api/embedding/projection` | `source`, `method` = `pca` (default), `tsne` or `lle`, optional `variant` | `{"positions": [[x, y, z], ...]}` in CSV row order. Cached per dataset and method. |
| `GET /api/embedding/clusters` | `source`, `k` (1–50, default 20), optional `variant` | `{"labels": [int, ...]}` in CSV row order: spherical (cosine) k-means on the full vectors. |
| `GET /api/embedding/compare` | `source`, `variant` (required) | `{nbr, w, origIdx}`: for each image in the comparison embeddings, its nearest neighbours there, mapped to row indices of the base layout. Used to overlay a second model's point cloud. |
| `POST /api/embedding/analysis` | JSON `{source, variant?, k?, reduction?}` | `202 {"job_id"}`. Background projection plus clusters; poll `GET /api/embedding/analysis/<job_id>`. The demo answers `429` when four are already running. Prefer the two cached `GET` routes above. |

## Images

All take an optional `max_size` (longest side in pixels, default 1600). Ask for 256–512 when you only need to look.

| Route | Parameters | Returns |
|---|---|---|
| `GET /api/image` | `path` | JPEG of the image. |
| `GET /api/mask` | `path` (a mask PNG) | PNG of the mask with a transparent background. |
| `GET /api/overlay` | `img`, `mask` | JPEG of the mask composited over the image. |
| `GET /api/annotation/mask` | `path` (the image) | PNG (transparent background) of the image's COCO polygon labels from `labels/<stem>.json`. |
| `GET /api/annotation/overlay` | `path` (the image), optional `alpha` (0–1, default 0.5) | JPEG of those labels over the image. |

A mask is the row's `mask_path` column when present, otherwise `<dataset>/masks/<stem>.png` next to
`<dataset>/images/<stem>.<ext>`. On the demo, paths must stay inside the data folder (`400 invalid path`
otherwise).

## Writes (local stack only)

Each of these answers `403` on the demo.

| Route | Body | Effect |
|---|---|---|
| `POST /api/datasets/build` | multipart form, `source_type` = `demo`, `local` or `upload` | Starts a build and returns `202 {"job_id"}`. `demo` needs `demo_key`. `local` needs `folder` (from `local-folders`), with optional `name` and `description`. `upload` needs `name` and one or more `images` files, with optional `annotations` (COCO JSON files) and `embeddings` (one JSON). `409` while another build runs. See the `magnidata-build-dataset` skill. |
| `POST /api/datasets` | JSON `{name, parent_source, image_names: [...], description?}` | Creates a child dataset holding the parent's rows whose image basename is in `image_names`. It inherits the parent's embeddings. Returns `201` with the new entry. |
| `POST /api/datasets/merge` | JSON `{name, sources: [two or more], description?}` | Creates the de-duplicated union. It keeps embeddings only when every source shares one embeddings file. |
| `POST /api/datasets/<id>/parents` | JSON `{parent_sources: [...]}` | Replaces a created dataset's lineage parents. |
| `DELETE /api/datasets/<id>` | none | Deletes a created dataset and its files. Datasets listed in `confi.yaml` cannot be deleted. Destructive: confirm with the user first. |
| `POST /api/partitions/commit` | JSON `{source, entries: [{image_name, cluster_id}]}` | Stores a cluster id per image, overwriting an earlier commit. |
| `POST /api/notes`, `PUT /api/notes/<id>`, `DELETE /api/notes/<id>` | JSON `{stem, note, image_path?}`, or `{note}` for `PUT` | Image notes. |

## Training and experiments (local stack with MLflow)

These routes read and launch training runs stored in the stack's MLflow. Start with
`GET /api/training/presets` (presets, bounds, and whether the trainer and MLflow are reachable),
`GET /api/runs` and `GET /api/runs/<run_id>`. `POST /api/training/runs` queues a run; it is a long, compute-heavy job,
so confirm with the user before launching one. Per-sample losses become dataset axes through
`GET /api/runs/<run_id>/columns`. Tags, metric keys and artifacts are specified in the repository's
`precisionai/magnidata/docs/training-contract.md`.
