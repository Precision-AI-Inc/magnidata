---
name: magnidata
description: >-
  Install, run and use MagniData, Precision AI's open-source explorer for image datasets: per-image quality and
  complexity metrics, segmentation masks and overlays, and embeddings projected to 3D (PCA, t-SNE, LLE) with k-means
  clusters, served by a Flask HTTP API and a React dashboard. Use when the user mentions MagniData, wants to explore,
  audit or visualize an image or segmentation dataset, find outliers, blurry, over-exposed or hard images, compare
  embeddings, or call the MagniData API (locally or on the public Hugging Face demo). Also use when MagniData's MCP
  server (magnidata-mcp) is connected. To turn a folder of images into a MagniData dataset, use magnidata-build-dataset.
license: Apache-2.0
compatibility: >-
  The public demo needs only HTTPS access (curl or any HTTP client). A local stack needs git, Docker with Compose and
  free ports 5175, 5051 and 5052.
---

# MagniData

MagniData turns an image dataset into one table (one row per image, about 40 numeric and categorical columns of
coverage, instance, exposure, focus, noise, colour and learned-quality metrics) plus optional embeddings, and serves
both over an HTTP API that a React/Three.js dashboard reads. Repository: https://github.com/Precision-AI-Inc/magnidata
(Apache 2.0).

Use it to answer questions such as "which images are blurry or under-exposed", "how are the classes and masks
distributed", "which images cluster together in embedding space", or "show me the overlay for this image".
It is not a labelling tool and it does not train general-purpose models. The training feedback loop in the repository
is a separate, optional feature of the local stack.

## 1. Pick a target

| Target | Base URL | When |
|---|---|---|
| Public demo | `https://precisionaiinc-magnidata.hf.space` | No install. COCO128 and AgriStress-500 are preloaded. Read-only. |
| Local stack | `http://localhost:5051` (API), `http://localhost:5175` (dashboard) | The user's own data, building datasets, curation, training. |

Check which one answers before doing anything else:

```bash
BASE=https://precisionaiinc-magnidata.hf.space   # or http://localhost:5051
curl -s "$BASE/api/health"    # {"status":"ok",...}
curl -s "$BASE/api/config"    # {"demo_mode":true} on the demo
```

The demo is a free CPU Space that sleeps when idle, so the first request after a pause can take a minute.
With `demo_mode` true, every route that writes data, builds a dataset or trains answers `403 {"error":"Disabled for
the Demo"}`, and image paths must stay inside the data folder. Do not retry those calls; switch to a local stack.

## 2. Run it locally

```bash
git clone https://github.com/Precision-AI-Inc/magnidata.git
cd magnidata
docker compose up --build -d
curl -s http://localhost:5051/api/health
```

That starts four services: `web` (dashboard, 5175), `api` (Flask, 5051), `mlflow` (5052) and `trainer` (CPU worker).
The first build takes several minutes. The dataset catalog starts empty. Load a demo dataset with one call (needs
internet, takes a few minutes):

```bash
curl -s -X POST -F source_type=demo -F demo_key=coco128 http://localhost:5051/api/datasets/build
# -> {"job_id":"..."}; poll GET /api/datasets/build/<job_id> until "status" is "done" or "error"
```

`GET /api/datasets/demos` lists the demo keys (`coco128`, `agristress500`).
To bring the user's own images, follow the `magnidata-build-dataset` skill.

Without Docker, only the Python feature-extraction and embeddings tools install as a package. The API must run from
inside `precisionai/magnidata/api/` (`pip install -r requirements.txt && python app.py`, port 5050 by default),
because its modules use flat imports. Check `pip index versions magnidata` for a PyPI release; until there is one,
install the wheel attached to each GitHub release (`gh release view --repo Precision-AI-Inc/magnidata` lists it).

## 3. Use the MCP server when it is connected

If a `magnidata-mcp` MCP server is connected in this session, prefer its tools; otherwise use the HTTP API below.

## 4. Explore a dataset over HTTP

Every dataset is identified by its `source`: the CSV path relative to the server's data root, exactly as
`GET /api/datasets` returns it (for example `data_user/coco128.csv`). Pass it URL-encoded wherever a route takes
`source`.

1. **List datasets.** `GET /api/datasets` returns `name`, `description`, `source`, `row_count`, `has_embeddings`
   and lineage (`parent_sources`).
2. **Summarize one.** `GET /api/datasets/summary?source=...` returns `row_count`, `column_count`, `image_column`,
   up to six `numeric_columns` with `min`, `max`, `mean` and a histogram, eight `sample_images` and `mask_coverage`.
3. **Pull the table.** `GET /api/dataset/csv?source=...` streams the full feature CSV. Load it with pandas (or the
   `csv` module) and filter or rank there; the API has no query endpoint. Column meanings are in `references/columns.md`.
4. **Embeddings.** When `has_embeddings` is true:
   - `GET /api/embedding/projection?source=...&method=pca|tsne|lle` returns `{"positions": [[x, y, z], ...]}`.
   - `GET /api/embedding/clusters?source=...&k=20` returns `{"labels": [...]}` from spherical k-means on the full
     vectors (`k` is clamped to 1–50).
   - Both lists are in CSV row order, so row `i` of the CSV is `positions[i]` and `labels[i]`.
   - `GET /api/dataset/embeddings?source=...` returns the raw vectors, keyed by image basename.
5. **Look at images.** Take the path from the CSV's `image_path` column:
   - `GET /api/image?path=...&max_size=512` returns a JPEG.
   - `GET /api/overlay?img=...&mask=...` returns the image with its mask composited on top. The mask is the row's
     `mask_path` column when present, otherwise `<dataset>/masks/<stem>.png` beside `<dataset>/images/<stem>.<ext>`.
   - `GET /api/annotation/overlay?path=...` draws COCO polygon labels instead.
   Keep `max_size` small (256–512) unless the user needs full resolution.

A worked example against the demo:

```bash
BASE=https://precisionaiinc-magnidata.hf.space
SRC=data_user/coco128.csv
curl -s "$BASE/api/datasets"
curl -s "$BASE/api/dataset/csv?source=$SRC" -o coco128.csv
curl -s "$BASE/api/embedding/clusters?source=$SRC&k=8" -o clusters.json
python - <<'PY'
import json
import pandas as pd

df = pd.read_csv("coco128.csv")
df["cluster_k8"] = json.load(open("clusters.json"))["labels"]
print(df.nsmallest(5, "blur_laplacian")[["image_path", "blur_laplacian", "complexity_score"]])
print(df.groupby("cluster_k8")["complexity_score"].describe())
PY
curl -s "$BASE/api/image?path=data_user/coco128_images/images/000000000009.jpg&max_size=512" -o sample.jpg
```

`references/http-api.md` lists every endpoint (except an unused debug route and the dashboard's MLflow probe) with
its parameters, responses and errors, including the write routes of a local stack (create a child dataset from a
selection, merge, partition, notes) and the training and experiment routes.

## 5. Point the user at the dashboard

The API answers questions; the dashboard is where people look. After an analysis, tell the user where to see it:
the demo at https://huggingface.co/spaces/precisionaiinc/magnidata, or `http://localhost:5175` locally. Its tabs
are Visual Explorer (parallel coordinates, histograms, table, image/mask/overlay preview, 3D embeddings) and
Experiments (training runs, local stack only). A **User Guide** inside the dashboard covers each view.

## Gotchas

- A `409` with "Git LFS pointer" means the CSV or JSON on that server was never pulled; run `git lfs pull` in the
  checkout and restart the stack.
- `/api/dataset/embeddings/variants` can list a `projections` entry. That is the precomputed-projection sidecar,
  not a model; ignore it when choosing a `variant`.
- The demo allows at most four background embedding analyses at once (`POST /api/embedding/analysis` answers `429`
  beyond that). The `GET` projection and cluster routes above are cached and do not count.
- Empty quality columns (`nima_ava`, `niqe`, `brisque`) mean the dataset was built without the learned-quality
  models, not that the images scored zero.
- The column schema, embeddings JSON format and naming rules are in the repository at
  `precisionai/magnidata/docs/data-contract.md`.
