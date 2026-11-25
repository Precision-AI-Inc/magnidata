# MagniData HTTP API reference

All routes are under `/api` on the API's base URL: `http://localhost:5051` for the compose stack,
`http://localhost:5050` for `python app.py`, and `https://precisionaiinc-magnidata.hf.space` for the read-only
public demo. Responses are JSON unless noted. Errors are `{"error": "<message>"}` with a 4xx or 5xx status.
There is no authentication. The demo allows any CORS origin; the compose stack allows only its portal, and listens
on 127.0.0.1 only. Paths (`source`, `path`, `img`, `mask`) must lie inside `data/`, `data_user/` or `image_sets/`.

`source` is always a dataset CSV path relative to the server's data root, as returned by `GET /api/datasets`.
Image paths (`path`, `img`, `mask`) are the values in a dataset's `image_path` and `mask_path` columns.
URL-encode both.

## Service

| Route | Returns |
|---|---|
| `GET /api/health` | `{"status": "ok", "local": {...}}` |
| `GET /api/config` | `{"demo_mode": bool}`. When true, the routes marked **write** below answer `403`. |
| `GET /api/openapi.json` | The OpenAPI 3.1 document for the read (`GET`) endpoints, as JSON. |

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
| `GET /api/datasets/build/current` | none | `{"job": {...}}` for the build that is running now (same shape as below), or `{"job": null}`. Use it to pick up a build you did not start or lost track of. |
| `GET /api/datasets/build/<job_id>` | none | `{job_id, kind, status, percent, message, name, dataset, error, skipped}`. `status` ends as `done` (then `dataset` holds the new entry) or `error`. `skipped` lists `{file, reason}` for images that could not be read and were left out. Jobs survive an API restart; one cut off by it ends as `error`. |
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

These routes read and launch training runs stored in the stack's MLflow. They answer `503` while MLflow is
unreachable and `404` for an unknown run. Start with `GET /api/training/presets`, `GET /api/runs` and
`GET /api/runs/<run_id>`. Tags, metric keys and artifacts are specified in the repository's
`precisionai/magnidata/docs/training-contract.md`.

| Route | Parameters or body | Returns |
|---|---|---|
| `GET /api/experiments` | optional `query`, `max_results`, `page_token` | `{experiments, next_page_token}`. |
| `GET /api/runs` | optional `experiment_ids` (comma-separated), `dataset`, `status`, `per_sample` (`any`, `scored`, `split`), `filter`, `order_by`, `max_results`, `page_token` | `{runs, next_page_token}`. |
| `GET /api/datasets/experiment-counts` | none | `{"<source>": {total, active}}` run counts per linked dataset; `{}` while MLflow is unreachable. |
| `GET /api/runs/<run_id>` | none | One run: params, tags, `metric_keys`, `scored_epochs`, `artifact_uri` and top-level `artifacts`. |
| `GET /api/runs/<run_id>/metrics` | optional `keys` (comma-separated) | `{run_id, metrics, truncated}`, the metric histories (all keys, capped, when omitted). |
| `POST /api/runs/metrics` | JSON `{run_ids: [...], keys: [...]}` | `{"runs": {"<run_id>": {...}}}`, histories of several runs for comparison. Read-only. |
| `GET /api/runs/<run_id>/artifacts` | optional `path` | `{path, artifacts}`, one directory level. |
| `GET /api/runs/<run_id>/artifact` | `path` | The file: images inline, anything else as a download. `413` above 50 MB. |
| `GET /api/runs/<run_id>/columns` | optional `epoch` (`latest` or a number) | CSV of per-sample loss axes per image, to join to the dataset as extra axes. `409` when the run logged none. |
| `GET /api/runs/<run_id>/match` | `dataset` (a `source`) | `{matched, run_only, dataset_only, run_only_examples, dataset_only_examples}`: image names shared with the dataset. |
| `GET /api/runs/<run_id>/samples/<image_name>` | none | One image's per-term loss across scored epochs, plus `prediction_epochs`. |
| `GET /api/runs/<run_id>/prediction` | `image`, optional `epoch`, `alpha`, `max_size` | JPEG of the dataset image with the run's predicted mask over it. |
| `GET /api/runs/<run_id>/inference` | `image` | `{status, image, epoch}` of on-demand inference for that image at the latest scored epoch. |
| `GET /api/runs/compare-samples` | `a`, `b` (run ids), optional `epoch` | CSV comparing the two runs image by image (hardness and loss change from `a` to `b`). |
| `GET /api/training/presets` | none | The presets and their bounds, plus whether the trainer (`backend`) and `mlflow` are reachable. |

These start work or change runs; each answers `403` on the demo.

| Route | Body | Effect |
|---|---|---|
| `POST /api/training/runs` | JSON `{preset, dataset_source, params?, name?, parent_run_id?}` | Queues a `SCHEDULED` run and returns `201` with it. `params` must stay within the preset's bounds. `409` while the trainer is busy. A long, compute-heavy job: confirm with the user first. |
| `POST /api/runs/<run_id>/cancel` | none | Kills a scheduled run, or asks a running one to stop. `409` when it cannot be cancelled. |
| `POST /api/runs/<run_id>/link` | JSON `{dataset_source}` | Attaches an external run to a dataset by tagging it. |
| `POST /api/runs/<run_id>/inference` | JSON `{image}` | Queues inference for one image with the run's latest scored checkpoint; poll the `GET` form. |
