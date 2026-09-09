# magnidata — release bundle

Self-contained copy of the Precision AI visual-query stack plus the feature-extraction
tool, so the whole thing can be built and run from this folder. This is the only copy
of the stack in the repo — the original top-level `dashboard/`, `confi.yaml`, and
`docker-compose.yml` have been retired.

```
magnidata/
  api/         Flask API (images/masks/overlays, datasets, embedding compute)
  dashboard/   React + Vite portal, served by nginx
  tools/       Feature extractor (CSV/images -> CSV, see tools/README.md) + embeddings
               generator (CSV/images -> JSON, see docs/data-contract.md)
  docs/        data-contract.md — the multiparametric CSV + embeddings JSON contracts
  confi.yaml   Dataset registry (which CSVs the portal lists)
  requirements.txt   deps for the tools/ feature extractor + embeddings generator (numpy/torch/timm/pyiqa)
  SUMMARY.md   Feature taxonomy reference for tools/ (atomic/model-based/compound tiers)
```

`docker-compose.yml`, `Dockerfile.api`, and `Dockerfile.dashboard` live at the **repo
root**, not in this folder — the compose file's build contexts point back into `api/`
and `dashboard/` here, so moving them doesn't require any change to either Dockerfile.

## Run the dev stack (portal + API)

```bash
cd <repo root>
docker compose up --build
#   portal -> http://localhost:5175
#   api    -> http://localhost:5051/api/health
```

`confi.yaml` ships with an empty dataset list — this is a clean install, not preloaded with
any sample data. From the portal's "Bring Your Own Data" dialog, either:

- **Load Demo** → prepare COCO128 or AgriStress-500. Both are self-contained: images (and,
  for COCO128, precomputed embeddings shipped in `scripts/`) are fetched and built entirely
  by the API, no external data mount required.
- **Prepare From Server Folder** → build from a folder already placed in `./image_sets`
  (mounted read-only at `/app/image_sets`). Lay it out as `image_sets/<NAME>/images/` plus an
  optional `image_sets/<NAME>/labels/` holding one COCO-format JSON per image, matched by
  filename stem. The dialog indexes every such folder with its image and label counts, and the
  build runs **in place** — the image data is never uploaded or copied, so this route has no
  size cap. Subfolders under `images/` become the `cluster`/`cluster_l2` grouping.
- **Create Dataset** → upload your own images from the browser (optionally with COCO-format
  annotations and/or a pre-computed embeddings JSON) — see
  [`docs/data-contract.md`](docs/data-contract.md).

Docker accepts BYOD image/annotation uploads up to 16 GiB by default, plus a 100 MB
embeddings JSON. For larger uploads, raise both `DATASET_BUILD_MAX_TOTAL_BYTES` on
the API service and `dashboard/nginx.conf`'s `client_max_body_size`, then rebuild the
web image so nginx picks up the change — or sidestep the upload entirely and use the
`image_sets/` route above, which has no limit.

To instead register your own pre-built CSV/embeddings pair as a permanent catalog entry, add
it under `./data` (mounted read-only into the API container) and list it in `confi.yaml`.
User-created child datasets are written to `./data_user` (the only read-write mount).

If you run the React dev server directly with `npm run dev`, it proxies `/api` to
`http://localhost:5051` by default to match Docker Compose. For a locally-run API
on another port, set `VITE_API_PROXY_TARGET`, for example:

```bash
VITE_API_PROXY_TARGET=http://localhost:5050 npm run dev
```

## Run the feature extractor

```bash
pip install -r requirements.txt
python -m precisionai.magnidata.tools.features --input data/agri-benc-0.0.3.csv --output out.csv
```

See `tools/README.md` for the full column schema, flags (`--skip-nima`, `--image-source`,
`--device`, `--limit`), and reproducibility/provenance details.

## Run the embedding generator

```bash
pip install -r requirements.txt
python -m precisionai.magnidata.tools.embeddings --input data/agri-benc-0.0.3.csv --output data/agri-benc-0.0.3.json
```

Default backbone is DINOv2 (see `tools/embedding_models.py` to add another). See
[`docs/data-contract.md`](docs/data-contract.md) for the full output format and how
the base-vs-comparison-model naming convention (`<stem>.json` vs `<stem>_<variant>.json`)
maps to the dashboard's "Compare" overlay.

## Data contract

[`docs/data-contract.md`](docs/data-contract.md) documents both file formats the
dashboard consumes — the multiparametric CSV and the embeddings JSON — including how
to generate each with the tools above, and how to bring your own that conforms.

## Recipe: start from raw images

Use [`docs/recipes/from-images-to-dashboard.md`](docs/recipes/from-images-to-dashboard.md)
to turn any local folder of images into a dashboard-ready dataset with folder-derived
ground-truth clusters, extracted features, DINOv2 embeddings, provenance, and the
commands needed to reproduce the run.

For a public end-to-end smoke dataset, the COCO128 helper downloads Ultralytics
COCO128, converts its YOLO labels, extracts features, writes DINOv2 embeddings,
and registers the dashboard dataset:

```bash
python -m precisionai.magnidata.scripts.prepare_coco128_dashboard --device cpu
```
