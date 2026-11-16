<p align="center">
  <img src="assets/magnidata.png" alt="MagniData Logo" width="512"/>
</p>

# Precision AI MagniData

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.md)
[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Live%20demo-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/precisionaiinc/magnidata)

---

Precision AI Open Source Data Visualization Tool for exploring image datasets: a
deterministic feature-extraction pipeline, a Flask API, and a React/Three.js dashboard
for browsing images, segmentation masks, and embeddings — including a live 3D
PCA/t-SNE/cluster view over a dataset's embedding space.

**Try it without installing anything:** the [live demo on Hugging Face](https://huggingface.co/spaces/precisionaiinc/magnidata)
has COCO128 and AgriStress-500 already loaded. Its HTTP API answers read-only requests at
`https://precisionaiinc-magnidata.hf.space/api`: see [Use it from a script or an agent](#use-it-from-a-script-or-an-agent).
The current version is **0.1.1**; releases, with a wheel and sdist attached, are on [GitHub Releases](https://github.com/Precision-AI-Inc/magnidata/releases).

- **`precisionai/magnidata/tools/`** — a domain-agnostic feature extractor: turns a folder of
  images (optionally with COCO-format segmentation annotations) into a single CSV of
  per-image complexity/quality/coverage metrics, plus an optional embeddings generator
  (DINOv2 by default) producing a matching JSON.
- **`precisionai/magnidata/api/`** — a Flask API serving images/masks/overlays, dataset
  listing and CSV/embeddings access, server-side PCA/t-SNE/LLE projection and clustering
  over embeddings, and an in-app dataset-build pipeline (upload your own images, or
  one-click "demo" builds of COCO128 / AgriStress-500).
- **`precisionai/magnidata/dashboard/`** — the React + Vite + Three.js portal: parallel
  coordinates, histograms, a data table, an image/mask/overlay preview, and the 3D
  embeddings view, plus an Experiments tab for training runs.
- **Training feedback loop** — train segmentation models from the dashboard, store every run in
  a local MLflow, and turn per-image losses into dataset axes so persistently hard images can
  be found, curated into a new dataset and trained on again. See the
  [recipe](precisionai/magnidata/docs/recipes/training-feedback-loop.md) and the
  [training contract](precisionai/magnidata/docs/training-contract.md).

Coding standards, naming conventions, and tooling configuration for the Python side are
governed by [CLAUDE.md](CLAUDE.md). This repository was migrated from an existing,
working application rather than started from a blank template — see
[CONTRIBUTING.md](CONTRIBUTING.md#known-gaps) for the specific places it doesn't yet meet
every standard there.

---

## Quickstart — run the full stack

```bash
git clone https://github.com/Precision-AI-Inc/magnidata.git
cd magnidata
docker compose up --build
#   portal -> http://localhost:5175
#   api    -> http://localhost:5051/api/health
#   mlflow -> http://localhost:5052   (experiment tracking UI)
```

**New here?** Open **User Guide** in the portal. It is a self-contained, illustrated guide that takes you from
`docker compose up` through loading AgriStress-500, partitioning it into sub-datasets from the command line,
exploring, training and curating a better dataset, with every command and screenshot included.
Its **Train with AgriTune** chapter explains the optional frozen-feature decoder presets, hosted encoder
configuration, the offline fake-encoder smoke test, and how component losses become Explorer axes.

The API container also prints those local access URLs during startup. The stack has four
services:

| Service | Port | Role |
|---|---|---|
| `web` | 5175 | the dashboard (React, served by nginx) |
| `api` | 5051 | Flask API; proxies MLflow, never runs training |
| `mlflow` | 5052 | MLflow tracking server and UI; data in the `magnidata-mlflow` volume |
| `trainer` | — | worker that trains runs scheduled from the dashboard (CPU by default; on a GPU host add `docker-compose.gpu.yml` — see below) |

To train on an NVIDIA GPU (needs the NVIDIA container toolkit), put
`COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml` in the git-ignored `.env`, then run
`docker compose up --build` as usual. Check it with
`docker compose exec trainer python -c "import torch; print(torch.cuda.is_available())"`. The trainer's
log (first line `device=…`) is uploaded to the run's `logs/` artifacts while it trains.

To run the reference trainer outside Docker: `pip install -e ".[training]"`, then
`python -m precisionai.magnidata.training.train --dataset data_user/<name>.csv --data-root precisionai/magnidata`.
The AgriTune presets are optional. To build them into the trainer image, clone
`git@github.com:Precision-AI-Inc/agritune.git` next to this repository (so `../agritune` exists) and add
`docker-compose.agritune.yml` to `COMPOSE_FILE` in the git-ignored `.env`
(`COMPOSE_FILE=docker-compose.yml:docker-compose.agritune.yml`; with the GPU file, list all three on that one line,
since a second `COMPOSE_FILE` line overrides the first), then run
`docker compose up --build`. Without it the stack builds and runs as before. AgriTune runs use the hosted encoder
(`https://embeddings.precision.ai/v1`) with the key from AgriTune's own `../agritune/.env`
(`AGRITUNE_ENCODER_API_KEY`); they encode a dataset into the feature store once and then train from features
preloaded into RAM. Set `AGRITUNE_ENCODER_BASE_URL=` (empty) in `.env` to use the offline fake encoder instead.
With *Encoder input: Original image size* they encode images unchanged (masks at their own size) instead of
squaring them, and a store already built by `agritune features build` can be read in place by pointing the trainer
at it with `AGRITUNE_EXISTING_MANIFEST` and `AGRITUNE_EXISTING_STORE` (see the User Guide, *Train with AgriTune*).

The dataset catalog (`precisionai/magnidata/confi.yaml`) ships empty. From the portal's
"Bring Your Own Data" dialog, either prepare a self-contained demo (COCO128 or
AgriStress-500 — no external data required), build from a folder you've placed in
`./image_sets` (read in place — no upload and no size cap), or upload your own images. See
[`precisionai/magnidata/README.md`](precisionai/magnidata/README.md) for the full breakdown
of the app's layout, data mounts, and how to register a permanent catalog entry.
Docker accepts BYOD image/annotation uploads up to 16 GiB by default, plus a 100 MB
embeddings JSON. For larger uploads, raise both the API service's
`DATASET_BUILD_MAX_TOTAL_BYTES` and `dashboard/nginx.conf`'s `client_max_body_size` — or
use the `./image_sets` route, which bypasses the upload entirely.

### Hugging Face Space

[`deploy/huggingface/`](deploy/huggingface) builds MagniData as a single-container
[Hugging Face Space](https://huggingface.co/spaces/precisionaiinc/magnidata) with COCO128 and AgriStress-500
already loaded. Its Dockerfile clones this repository at a chosen branch or tag, builds the dashboard, builds
COCO128 while the image is built and installs AgriStress-500 from a downsized bundle; the API serves the dashboard
itself when `MAGNIDATA_STATIC_DIR` is set. The Space is a read-only demo (`MAGNIDATA_DEMO_MODE`), so anything that
changes data, training and the Experiments tab are turned off there (no MLflow or trainer). See
[`deploy/huggingface/MAINTAINING.md`](deploy/huggingface/MAINTAINING.md) to build it locally or update the Space.

---

## Use it from a script or an agent

MagniData has no SDK to learn: everything the dashboard shows comes from a plain HTTP API, and the public demo serves
that API without authentication.

- **Source:** `git clone https://github.com/Precision-AI-Inc/magnidata.git`
- **Python package:** not on PyPI yet (coming). Install the wheel from the latest GitHub Release instead:

  ```bash
  pip install https://github.com/Precision-AI-Inc/magnidata/releases/download/v0.1.1/magnidata-0.1.1-py3-none-any.whl
  python -m precisionai.magnidata.tools.features --help
  ```

  The package gives you the feature extractor and embeddings CLI (see
  [Feature extraction & embeddings CLI](#feature-extraction--embeddings-cli)). The API and dashboard run with
  `docker compose up` (see [Quickstart](#quickstart--run-the-full-stack)).
- **HTTP API base URLs:** `https://precisionaiinc-magnidata.hf.space/api` for the public demo, and
  `http://localhost:5051/api` for your own stack. Responses are JSON unless noted, and CORS is open.

The demo is read-only: every request that would change data, build a dataset or train answers
`403 {"error": "Disabled for the Demo"}`. It runs on small CPU hardware and sleeps after two days without visitors,
so the first request after a quiet spell can take a minute while it wakes. On your own stack nothing is disabled.

| Endpoint | Returns |
|---|---|
| `GET /api/health` | `{"status": "ok", ...}` |
| `GET /api/config` | `{"demo_mode": true}` on the demo |
| `GET /api/datasets` | Every dataset: `name`, `source` (the id the other endpoints take), `row_count`, `has_embeddings` |
| `GET /api/datasets/summary?source=` | Row and column counts, per-column min, max, mean and histogram, sample image paths |
| `GET /api/dataset/csv?source=` | The per-image feature table as CSV (one row per image, `image_path` first) |
| `GET /api/dataset/embeddings?source=&variant=` | `{"embeddings": {"<image name>": [floats, ...]}}`; `variant` (optional) picks a comparison model's file |
| `GET /api/dataset/embeddings/variants?source=` | `[{"name", "variant"}, ...]`: the comparison embedding files next to the dataset (`<stem>_<variant>.json`), each usable as `variant=` |
| `GET /api/embedding/projection?source=&method=&variant=` | `{"positions": [[x, y, z], ...]}` in row order; `method` is `pca`, `tsne` or `lle`; `variant` (optional) projects a comparison model's embeddings |
| `GET /api/embedding/clusters?source=&k=&variant=` | `{"labels": [...]}` in row order: k-means over the full embeddings, `k` from 1 to 50 (default 20); `variant` as above |
| `GET /api/embedding/compare?source=&variant=` | `{"nbr", "w", "origIdx"}`: for each item of the comparison model `variant` (required), its nearest neighbours as row indices into the base layout, with weights |
| `GET /api/image?path=&max_size=` | The image as JPEG, longest side capped at `max_size` (default 1600) |
| `GET /api/mask?path=&max_size=` | A mask image as PNG with black pixels made transparent |
| `GET /api/overlay?img=&mask=&max_size=` | The mask at `mask` composited over the image at `img`, as JPEG |
| `GET /api/annotation/overlay?path=&alpha=` | The image with its segmentation annotation drawn on top, as JPEG |

`path` and `img` take an `image_path` value from the CSV; `mask` and the `path` of `/api/mask` take a mask file's path
(a `mask_path` value, or the PNG under `masks/` next to `images/`). The examples below ran against the demo; outputs are trimmed.

```bash
API=https://precisionaiinc-magnidata.hf.space/api

# 1. Is it up?
curl -s "$API/health"
# {"local":{"datalake_accessible":false},"status":"ok"}

# 2. Which datasets are there?
curl -s "$API/datasets" | jq -c '.[] | {name, source, row_count, has_embeddings}'
# {"name":"COCO128","source":"data_user/coco128.csv","row_count":128,"has_embeddings":true}
# {"name":"AgriStress-500","source":"data_user/agristress500.csv","row_count":451,"has_embeddings":true}

# 3. What is in one?
curl -s "$API/datasets/summary?source=data_user/coco128.csv" \
  | jq -c '{row_count, column_count, first: .numeric_columns[0] | {name, min, max, mean}}'
# {"row_count":128,"column_count":41,"first":{"name":"width","min":333.0,"max":640.0,"mean":578.859375}}

# 4. Download the per-image feature table
curl -s "$API/dataset/csv?source=data_user/coco128.csv" -o coco128.csv
head -1 coco128.csv | cut -d, -f1-6
# image_path,width,height,cluster,cluster_l2,annotation_ratio

# 5. Project the embeddings to 3D and cluster them (both in CSV row order)
curl -s "$API/embedding/projection?source=data_user/coco128.csv&method=pca" | jq -c '.positions[0]'
# [-0.37185603380203247,-2.966247797012329,-1.1970314979553223]
curl -s "$API/embedding/clusters?source=data_user/coco128.csv&k=5" | jq -c '.labels[:10]'
# [3,1,2,1,4,1,3,1,4,0]

# 6. Fetch one image, 256 px on its longest side
curl -s "$API/image?path=data_user/coco128_images/images/000000000009.jpg&max_size=256" -o 000000000009.jpg
```

[`llms.txt`](llms.txt) lists the documents worth reading first. To cite MagniData, use [`CITATION.cff`](CITATION.cff)
(GitHub's "Cite this repository").

---

## Agent skills and Claude Code plugin

Two [Agent Skills](https://agentskills.io) teach an AI agent to install, run and use MagniData:
`magnidata` (run the stack or use the public demo, and explore datasets, images and embeddings over the HTTP API)
and `magnidata-build-dataset` (turn a folder of images into a dataset). In Claude Code, add this repository as a
plugin marketplace and install the plugin:

```text
/plugin marketplace add Precision-AI-Inc/magnidata
/plugin install magnidata@magnidata
```

From a shell, use `claude plugin marketplace add Precision-AI-Inc/magnidata`, then
`claude plugin install magnidata@magnidata`. Other agents that read Agent Skills can install them with
`npx skills add Precision-AI-Inc/magnidata`. Codex also finds them in [`.agents/skills/`](.agents/skills) when it
works inside a checkout. The skills live in [`plugins/magnidata/skills/`](plugins/magnidata/skills), and the
marketplace is [`.claude-plugin/marketplace.json`](.claude-plugin/marketplace.json).

---

## Installation (Python package only)

```bash
python -m venv .venv
source .venv/bin/activate

# API + feature-extraction toolkit
pip install -e .

# + offline embeddings generation and learned image-quality scoring (torch/timm/pyiqa)
pip install -e ".[ml]"

# Full development install (tests, pre-commit, type checking)
pip install -e ".[dev]"
pre-commit install
```

The dashboard is a separate Node/Vite app (Node 18+) — see
[`precisionai/magnidata/dashboard`](precisionai/magnidata/dashboard):

```bash
cd precisionai/magnidata/dashboard
npm install
npm run dev   # proxies /api to http://localhost:5051 by default — see vite.config.ts
```

---

## Running the API server

The primary path is Docker Compose (see Quickstart above). To run it directly instead —
note this must be run **from within `precisionai/magnidata/api/`**, not as an installed
package: its modules use flat, sibling-style imports (`import db`, `import local_files`)
matching how `Dockerfile.api` copies them into the container, so `python -m
precisionai.magnidata.api.app` will not work.

```bash
cd precisionai/magnidata/api
pip install -r requirements.txt
python app.py          # default: host 0.0.0.0, port 5050 (override with API_PORT)
```

`./start.sh` does the same, plus creating/activating a local `.venv` first.

## Feature extraction & embeddings CLI

```bash
python -m precisionai.magnidata.tools.features --input data/mydata.csv --output out.csv
python -m precisionai.magnidata.tools.embeddings --input data/mydata.csv --output data/mydata.json
```

An installed package also provides them as the `magnidata-features` and `magnidata-embeddings` commands.

See [`precisionai/magnidata/tools/README.md`](precisionai/magnidata/tools/README.md) for the
full column schema and flags, and
[`precisionai/magnidata/docs/data-contract.md`](precisionai/magnidata/docs/data-contract.md)
for the CSV/embeddings-JSON formats the dashboard consumes — including how to bring your
own data that conforms.
[`precisionai/magnidata/docs/recipes/from-images-to-dashboard.md`](precisionai/magnidata/docs/recipes/from-images-to-dashboard.md)
walks through turning a raw folder of images into a dashboard-ready dataset end to end.

---

## Testing

```bash
python -m pytest                                          # run all tests with coverage report
pytest precisionai/magnidata/tools/test_features.py          # run a specific module
pytest -k "coverage"                                        # run tests matching a keyword
```

Tests are colocated with the module they cover (`precisionai/magnidata/**/test_*.py`), not
under a top-level `tests/` — see [CONTRIBUTING.md](CONTRIBUTING.md#known-gaps). Coverage
is enforced at 60% (measured at 63.2% at migration time), short of the org's 90% target —
also tracked there.

---

## Pre-commit hooks

| Hook | What it checks |
|---|---|
| File hygiene | Large files (> 1800 KB), trailing whitespace, merge conflicts, private keys, debug statements, BOM removal |
| `ruff` | Linting and import sorting (auto-fix) on the files you touch |
| `ruff-format` | Code formatting (auto-fix) |
| `pyright` | Static type checking |
| `pytest` | Full test suite with a coverage floor |

Run all hooks manually without committing:

```bash
pre-commit run --all-files
```

---

## Project layout

```
precisionai/magnidata/
  api/          Flask API — images/masks/overlays, dataset listing, embedding
                projection/clustering, in-app dataset builds
  tools/        Feature extractor (images [+ COCO annotations] -> CSV) and
                embeddings generator (images -> JSON) — see tools/README.md
  scripts/      Standalone dataset-preparation scripts (e.g. the COCO128 demo)
  dashboard/    React + Vite + Three.js portal, served by nginx in Docker
  training/     Training feedback loop: MLflow contract, per-sample loss recorder, reference
                trainer and worker (torch only where noted)
  docs/         data-contract.md, training-contract.md and recipes/ — file formats and guides
  confi.yaml    Dataset registry (which CSVs the portal lists)
Dockerfile.api, Dockerfile.dashboard, Dockerfile.trainer, docker-compose.yml   at the repo root
plugins/magnidata/   Claude Code plugin holding the agent skills; .claude-plugin/ and .agents/skills/ point at it
docs/           Sphinx (HTML + LaTeX/PDF) for the Python package
```

See [CLAUDE.md](CLAUDE.md) for the full coding standard covering imports, docstrings, type hints, testing, and what to avoid — and [CONTRIBUTING.md](CONTRIBUTING.md#known-gaps) for where this repo currently deviates from it.

---

## Documentation

```bash
cd docs
make html       # HTML docs → docs/_build/html/index.html
make latexpdf   # PDF      → docs/_build/latex/documentation.pdf
make clean      # Remove build artefacts
```

Dependencies: `pip install -r docs/requirements.txt`

---


## Security

To report a security vulnerability, see [SECURITY.md](SECURITY.md). Do not open a public issue.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, branching, and PR guidelines, and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community expectations. [CLAUDE.md](CLAUDE.md) documents the code style and conventions this repo targets.

---

## License

[Apache 2.0](LICENSE.md) © Precision AI
