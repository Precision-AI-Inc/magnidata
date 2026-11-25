---
name: magnidata-build-dataset
description: >-
  Turn a folder of images (optionally with COCO polygon labels, segmentation masks or precomputed embeddings) into a
  MagniData dataset: a per-image feature CSV plus optional DINOv2 embeddings, registered so the MagniData dashboard
  and API can explore it. Use when the user wants to load, import, index or prepare their own images for MagniData,
  compute per-image quality and complexity metrics or DINOv2 embeddings for an image folder, or bring an existing
  CSV or embeddings file into MagniData. For exploring a dataset that is already loaded, use the magnidata skill.
license: Apache-2.0
compatibility: >-
  Needs a local MagniData checkout (git, Docker with Compose). The embeddings route also needs Python 3.10+ and
  PyTorch; it runs on CPU, faster on a GPU. Not possible on the public read-only demo.
---

# Build a MagniData dataset from images

The public demo refuses builds, so this needs a local stack. If none is running, start one from a checkout:

```bash
git clone https://github.com/Precision-AI-Inc/magnidata.git
cd magnidata
docker compose up --build -d
curl -s http://localhost:5051/api/health
```

Run every command below from the repository root. `API=http://localhost:5051`.

## Choose a route

| The user has | Route | Result |
|---|---|---|
| A folder of images on this machine, with or without COCO labels | A. Build from `image_sets/` | Feature CSV, no embeddings, no size cap |
| Route A's dataset and wants the 3D embeddings view | A, then B | Adds DINOv2 embeddings to it |
| Images elsewhere and only HTTP access to the API | C. Upload | Feature CSV, plus embeddings if they bring a JSON |
| Already-computed features or embeddings | D. Bring your own files | Whatever they bring |
| Wants checksums, provenance and a permanent catalog entry | The full recipe | See the end of this skill |

Ask before copying large image folders; prefer hard links or moving files when the user agrees.

## A. Build from `image_sets/` (recommended)

1. Lay the images out under the repository's `image_sets/` folder, which is mounted read-only into the API
   container:

   ```text
   image_sets/<NAME>/
     images/            # .jpg .jpeg .png .bmp .webp .tif .tiff, subfolders allowed
     labels/            # optional: one COCO polygon JSON per image, same stem; flat images/ only (below)
   ```

   Subfolders of `images/` become the `cluster` and `cluster_l2` ground-truth groups
   (`images/siteA/batch1/x.png` gives `cluster=siteA/batch1`, `cluster_l2=siteA`). A flat folder builds as one
   cluster, `uncategorized`. The folder is listed on each request, so a running stack sees it without a restart.

   Labels in `labels/` are only used when `images/` is flat. The feature extractor looks for each label in a
   `labels/` folder beside the image's parent (`images/x.png` -> `labels/x.json`), so with subfolders it misses them: the build still
   finishes, but `instance_count` and coverage come out empty and only the API log mentions it (an info line counting images with no label file),
   even though `local-folders` counts the labels. To keep labels, use a flat `images/` folder (one cluster), or
   use route C, which matches annotations by stem wherever the images sit.

2. Confirm the API sees it, then start the build:

   ```bash
   curl -s $API/api/datasets/build/local-folders
   # [{"folder":"<NAME>","image_count":...,"label_count":...,"built":false}]
   curl -s -X POST -F source_type=local -F folder=<NAME> -F name="<Display name>" $API/api/datasets/build
   # -> {"job_id":"..."}
   ```

3. Poll until `status` is `done` or `error`. Only one build runs at a time; a second one answers `409`, and
   `GET /api/datasets/build/current` returns the running build (with its `job_id`) if you lost track of it.

   ```bash
   curl -s $API/api/datasets/build/<job_id>
   # {"status":"extracting_features","percent":30,...} ... {"status":"done","dataset":{"source":"data_user/<stem>.csv",...}}
   ```

   Files that cannot be read as images (a text file renamed `.png`, a truncated JPEG) are left out of the dataset;
   the finished job lists each one in `skipped` as `{file, reason}` and its `message` gives the count. A build cut
   off by an API restart ends as `error` and its partial output is removed; start it again.

4. The dataset now appears in `GET /api/datasets` and in the dashboard at http://localhost:5175. The finished
   job's `dataset.source` identifies it: `data_user/<stem>.csv`, where `<stem>` is the folder name lower-cased, with
   each run of characters other than letters, digits, `_` and `-` replaced by `_`, plus `_2`, `_3`, … when files
   under that name already exist. On the host that CSV is
   `precisionai/magnidata/data_user/<stem>.csv`. The build skips the slow learned-quality models (NIMA, NIQE,
   BRISQUE), so those columns are empty and `complexity_score` uses the deterministic quality terms.

Images without a matching label still build. They get the exposure, focus and colour columns but not the instance,
class and coverage ones.

## B. Add embeddings to a built dataset

The in-app builds never compute embeddings. Generate DINOv2 vectors on the host and save them beside the CSV with
the same stem; the API finds them by filename:

```bash
python -m venv .venv && source .venv/bin/activate
# From the repository root. CPU-only machine: first install the CPU build of the pinned torch and torchvision:
#   pip install torch torchvision -c constraints.txt --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -m precisionai.magnidata.tools.embeddings \
  --input  precisionai/magnidata/data_user/<stem>.csv \
  --output precisionai/magnidata/data_user/<stem>.json \
  --model dinov2 --device cpu            # --device cuda on a GPU; --limit 20 for a quick test
```

The CSV's `image_path` values (`image_sets/...`) resolve from the repository root, so run it there. The default
model is `vit_small_patch14_dinov2.lvd142m` (384 dimensions), downloaded on first use. After it finishes,
`GET /api/datasets` reports `has_embeddings: true` and the 3D view's Embeddings mode turns on. A second file named
`<stem>_<variant>.json` (another model, another seed) shows up as a Compare option.

## C. Upload over HTTP

When the images are not on the API's machine, upload them with a multipart request. Each file's name carries its
relative path, which sets the cluster:

```bash
curl -s $API/api/datasets/build/limits    # max_images, max_upload_bytes, max_embeddings_bytes
curl -s -X POST $API/api/datasets/build \
  -F source_type=upload -F name="My upload" -F description="..." \
  -F "images=@photos/siteA/a.jpg;filename=siteA/a.jpg" \
  -F "images=@photos/siteA/b.jpg;filename=siteA/b.jpg" \
  -F "annotations=@labels/a.json;filename=a.json" \
  -F "embeddings=@my_embeddings.json"
```

Uploads are capped (500 images by default, see `limits`), so route A is the better choice for anything large.
Annotations match images by filename stem. The optional embeddings JSON is stored as is and must follow the format
in section D.

## D. Bring your own files

- **Feature CSV.** Needs an `image_path` column with paths the API can read, relative to its data root
  (`image_sets/...` or `data_user/...`). Any subset of MagniData's columns works; the dashboard detects types per
  column. Register it in `precisionai/magnidata/confi.yaml`:

  ```yaml
  datasets:
  - name: "My Dataset"
    description: "..."
    source: data/my_dataset.csv
  ```

  The compose stack mounts the repository's `data/` read-only at the API's `data/`, and `confi.yaml` is read on
  each request, so no restart is needed. Datasets listed there cannot be deleted from the app.

- **Embeddings JSON.** One vector per image, keyed by image basename, all the same length:

  ```json
  {"embeddings": {"img_001.jpg": [0.0123, -0.045, 0.31]}}
  ```

  Name it `<csv stem>.json` beside the CSV. Image basenames must be unique within a dataset.

## Check the result

```bash
curl -s "$API/api/datasets/summary?source=data_user/<stem>.csv"   # row_count, columns, mask coverage
curl -s "$API/api/embedding/projection?source=data_user/<stem>.csv&method=pca" | head -c 200
```

Then explore it with the `magnidata` skill or in the dashboard.

## The full recipe

For reproducible, publishable datasets (staged copies with deterministic names, a manifest, SHA-256 checksums,
the full learned-quality panel, provenance sidecars and a `confi.yaml` entry), follow
`precisionai/magnidata/docs/recipes/from-images-to-dashboard.md` in the repository. For the COCO128 sample in one
command: `python -m precisionai.magnidata.scripts.prepare_coco128_dashboard --device cpu`.
The file formats are specified in `precisionai/magnidata/docs/data-contract.md`.
