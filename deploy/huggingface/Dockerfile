# Hugging Face Space image: the MagniData API and dashboard in one container, with the COCO128 and
# AgriStress-500 demos already loaded. It is built for the CPU Basic hardware tier (2 vCPUs, no GPU).
#
# The Space repository holds only this file and README.md. The source is cloned from GitHub at
# MAGNIDATA_REF (a branch or a tag), so moving the Space to a release is a one-line change.
# COCO128 is built here, from its public download, by the app's own demo build. AgriStress-500 is
# too large to build in an image (12 GB of originals), so a compact copy exported with
# scripts/export_demo_bundle.py is fetched from the Hugging Face dataset repo DEMO_DATA_REPO. That
# repo exists only for this Space: it is a downsized derivative, not the AgriStress-500 dataset
# (precisionaiinc/AgriStress-500), and nothing else in MagniData reads it.
#
#   docker build -t magnidata-space -f deploy/huggingface/Dockerfile deploy/huggingface
#   docker run --rm -p 7860:7860 magnidata-space      # http://localhost:7860
#
# A private DEMO_DATA_REPO needs a read token. On a Space, create a secret named HF_TOKEN in the
# Space settings; locally, add `--secret id=HF_TOKEN,env=HF_TOKEN` to the build. Build with
# `--build-arg DEMO_DATA_REPO=` to skip the bundles and get COCO128 only.

ARG MAGNIDATA_REPO=https://github.com/Precision-AI-Inc/magnidata.git
ARG MAGNIDATA_REF=main
ARG DEMO_DATA_REPO=precisionaiinc/AgriStress-500-Demo

# ── Stage 1: source ───────────────────────────────────────────────────────────
FROM alpine:3.20 AS source

RUN apk add --no-cache git
ARG MAGNIDATA_REPO
ARG MAGNIDATA_REF
RUN git clone --depth 1 --branch "${MAGNIDATA_REF}" "${MAGNIDATA_REPO}" /src

# ── Stage 2: dashboard ────────────────────────────────────────────────────────
FROM node:22-alpine AS dashboard

WORKDIR /app
COPY --from=source /src/precisionai/magnidata/dashboard/package.json /src/precisionai/magnidata/dashboard/package-lock.json ./
RUN npm ci
COPY --from=source /src/precisionai/magnidata/dashboard ./
# Same permission fix as Dockerfile.dashboard: the build output can come out group-only.
RUN npm run build && chmod -R a+rX dist

# ── Stage 3: demo bundles ─────────────────────────────────────────────────────
FROM python:3.12-slim AS bundles

ARG DEMO_DATA_REPO
RUN pip install --no-cache-dir "huggingface_hub>=1.0,<2"
# A failed download fails the build: a Space that silently lost a dataset is worse than no Space.
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    mkdir /bundles \
    && if [ -n "${DEMO_DATA_REPO}" ]; then \
        DEMO_DATA_REPO="${DEMO_DATA_REPO}" HF_TOKEN="$(cat /run/secrets/HF_TOKEN 2>/dev/null || true)" python -c \
        "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ['DEMO_DATA_REPO'], repo_type='dataset', local_dir='/bundles', token=os.environ['HF_TOKEN'] or None)"; \
    fi

# ── Stage 4: API, dashboard and preloaded data ────────────────────────────────
FROM python:3.12-slim

# Spaces run the container as uid 1000.
RUN useradd --create-home --uid 1000 user

WORKDIR /app
COPY --from=source /src/precisionai/magnidata/api/requirements.txt ./api-requirements.txt
RUN pip install --no-cache-dir -r api-requirements.txt gunicorn==23.0.0

# The package layout Dockerfile.api uses: tools/, scripts/ and training/ are imported as
# precisionai.magnidata.*, while the API modules sit flat next to app.py.
COPY --from=source --chown=user:user /src/precisionai/__init__.py ./precisionai/__init__.py
COPY --from=source --chown=user:user /src/precisionai/magnidata/__init__.py ./precisionai/magnidata/__init__.py
COPY --from=source --chown=user:user /src/precisionai/magnidata/tools ./precisionai/magnidata/tools
COPY --from=source --chown=user:user /src/precisionai/magnidata/scripts ./precisionai/magnidata/scripts
COPY --from=source --chown=user:user /src/precisionai/magnidata/training ./precisionai/magnidata/training
# Every API module rather than Dockerfile.api's explicit list, so a module added there cannot
# be missed here.
COPY --from=source --chown=user:user /src/precisionai/magnidata/api/*.py ./
COPY --from=source --chown=user:user /src/precisionai/magnidata/confi.yaml ./confi.yaml
COPY --from=dashboard --chown=user:user /app/dist ./dashboard
RUN rm -f test_*.py conftest.py \
    && mkdir -p db data_user \
    && chown user:user db data_user

# One process serves /api and the dashboard (MAGNIDATA_STATIC_DIR) on the port a Space expects.
# MLflow and the trainer are not part of the Space, so a request to the tracking server must be
# refused at once, not retried: its default address is a dead localhost port. CPU Basic gives 2
# vCPUs but the container can see the host's cores, and without a cap numpy and scikit-learn size
# their thread pools to the host and thrash.
ENV DATA_ROOT=/app \
    DB_PATH=/app/db/notes.db \
    CONFIG_PATH=/app/confi.yaml \
    API_PORT=7860 \
    MAGNIDATA_STATIC_DIR=/app/dashboard \
    MLFLOW_HTTP_REQUEST_MAX_RETRIES=0 \
    MLFLOW_HTTP_REQUEST_TIMEOUT=3 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    PYTHONUNBUFFERED=1

USER user

# Load both datasets while the image is built, so the first visitor finds them ready. The bundles
# are mounted from their stage rather than copied, so the image holds one copy of the data. The
# build fails here, not on the Space, if a download or an install does.
RUN --mount=type=bind,from=bundles,source=/bundles,target=/bundles \
    args=""; \
    for dir in /bundles/*/; do \
        if [ -f "${dir}bundle.json" ]; then args="${args} --bundle ${dir%/}"; fi; \
    done; \
    python preload_demo.py ${args} coco128

# Demo mode makes the server refuse dataset builds and training, and the dashboard show those controls
# disabled. It starts only now: the step above drives the very endpoint it refuses.
ENV MAGNIDATA_DEMO_MODE=1

# gunicorn, not `python app.py` (Flask's development server, which is not meant to face visitors). One worker
# process, because the 3D view's analysis jobs and the compute caches live in the process's memory: a second worker
# would be asked about jobs it never started. Threads give the concurrency instead. /dev/shm keeps gunicorn's
# heartbeat file off the container's overlay filesystem.
EXPOSE 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "8", "--timeout", "120", \
     "--worker-tmp-dir", "/dev/shm", "--access-logfile", "-", "app:app"]
