# Recipe: images to dashboard dataset

This recipe starts with any local folder of images and ends with an agriviz dataset
that can be loaded in the dashboard with:

- a dashboard-ready multiparametric CSV
- DINOv2 embeddings for the 3D embeddings view
- optional ground-truth clusters derived from the source folder layout
- provenance sidecars and checksums for reproducibility

Run every command from the repository root.

## COCO128 one-command shortcut

For a reproducible public sample dataset, use the helper script. It downloads the
Ultralytics COCO128 zip, stages the images under `image_sets/coco128/`, converts
YOLO labels to the COCO-style polygon JSON files used by the feature extractor,
generates `data/coco128.csv`, generates DINOv2 embeddings at `data/coco128.json`,
and registers the dataset in `precisionai/agriviz/confi.yaml`.

```bash
python -m precisionai.agriviz.scripts.prepare_coco128_dashboard --device cpu
```

For a fast smoke test:

```bash
python -m precisionai.agriviz.scripts.prepare_coco128_dashboard \
  --dataset-stem coco128_smoke \
  --dataset-name "COCO128 Smoke" \
  --limit 10 \
  --device cpu
```

To prepare the full dataset and start the dashboard API/portal:

```bash
python -m precisionai.agriviz.scripts.prepare_coco128_dashboard \
  --device cpu \
  --start-dashboard \
  --validate-api
```

COCO128 has a flat image folder, so the script defaults `cluster` to the dominant
COCO class in each image. Use `--cluster-by single` for one cluster, or
`--cluster-by first-class` to use the first YOLO label instead.

## What the recipe creates

For a dataset stem such as `my_images`, the recipe creates:

```text
image_sets/my_images/
  images/                         # staged local image copies, ignored by git
  my_images_input.csv             # deterministic image manifest
  my_images_checksums.sha256      # source-image checksums

data/my_images.csv                # features consumed by the dashboard
data/my_images.json               # default embeddings consumed by the dashboard
data/my_images.csv.provenance.json
data/my_images.json.provenance.json
```

The dashboard discovers embeddings by filename. If the CSV is `data/my_images.csv`,
the default embeddings file must be `data/my_images.json`.

The `image_sets/` directory is mounted into the API container at `/app/image_sets`.
The manifest uses relative paths such as `image_sets/my_images/images/img_abc.jpg`,
so the same path works for host-side feature generation and container-side image
preview.

## 1. Prepare Python

Use Python 3.12 when possible, matching the Docker/API runtime.

```bash
python3.12 -m venv .venv-agriviz
source .venv-agriviz/bin/activate
python -m pip install --upgrade pip
```

For a CPU-only machine, install the pinned CPU PyTorch wheels first:

```bash
pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r precisionai/agriviz/requirements.txt
```

On a GPU machine, the default package index is fine:

```bash
pip install -r precisionai/agriviz/requirements.txt
```

## 2. Choose the source folder and dataset name

```bash
export IMAGE_ROOT="/absolute/path/to/your/images"
export DATASET_STEM="my_images"
export DEVICE="cpu"       # use "cuda" on a GPU machine
export STAGE_MODE="copy"  # copy is safest for Docker; hardlink is OK on one filesystem
```

`IMAGE_ROOT` can be flat:

```text
raw_images/
  a.jpg
  b.jpg
```

or grouped by folders:

```text
raw_images/
  volunteer_corn/
    a.jpg
    b.jpg
  kochia/
    c.jpg
  kochia/dense/
    d.jpg
```

Cluster rules:

- `cluster` is the image's relative parent folder, for example `kochia/dense`.
- `cluster_l2` is the first folder component, for example `kochia`.
- Images directly under `IMAGE_ROOT` get `cluster=all` and `cluster_l2=all`.
- If the folder is flat, the dashboard still gets one valid ground-truth cluster.

## 3. Build the reproducible manifest

This script stages images into `image_sets/<DATASET_STEM>/images/` with deterministic,
unique basenames. Unique basenames matter because the embeddings JSON is keyed by
image basename.

```bash
python - <<'PY'
from pathlib import Path
import csv
import hashlib
import os
import re
import shutil

root = Path(os.environ["IMAGE_ROOT"]).expanduser().resolve()
stem = os.environ.get("DATASET_STEM", "image_set")
stage_mode = os.environ.get("STAGE_MODE", "copy").lower()

stage_root = Path("image_sets") / stem
stage_img_dir = stage_root / "images"
manifest = stage_root / f"{stem}_input.csv"
checksums = stage_root / f"{stem}_checksums.sha256"
extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

def slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return value[:90] or "image"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def stage_file(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if stage_mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    if stage_mode == "symlink":
        # Only use symlink when the link target is also visible to the API container.
        dst.symlink_to(os.path.relpath(src, dst.parent))
        return
    shutil.copy2(src, dst)

if not root.is_dir():
    raise SystemExit(f"IMAGE_ROOT is not a directory: {root}")

stage_img_dir.mkdir(parents=True, exist_ok=True)

rows = []
checksum_lines = []
sources = sorted(
    p for p in root.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
)
if not sources:
    raise SystemExit(f"No images found under {root}")

for src in sources:
    rel = src.relative_to(root)
    rel_parent = rel.parent
    cluster = rel_parent.as_posix() if rel_parent.parts else "all"
    cluster_l2 = rel_parent.parts[0] if rel_parent.parts else "all"

    digest = hashlib.sha256(rel.as_posix().encode("utf-8")).hexdigest()[:12]
    staged_name = f"{slug(rel.with_suffix('').as_posix())}_{digest}{src.suffix.lower()}"
    staged = stage_img_dir / staged_name
    stage_file(src, staged)

    rows.append({
        "image_path": staged.as_posix(),
        "cluster": cluster,
        "cluster_l2": cluster_l2,
        "source_path": str(src),
    })
    checksum_lines.append(f"{sha256_file(src)}  {src}")

with manifest.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image_path", "cluster", "cluster_l2", "source_path"])
    writer.writeheader()
    writer.writerows(rows)

checksums.write_text("\n".join(checksum_lines) + "\n")

print(f"Wrote {len(rows)} rows to {manifest}")
print(f"Wrote checksums to {checksums}")
print("Clusters:")
for name in sorted({r["cluster"] for r in rows}):
    count = sum(1 for r in rows if r["cluster"] == name)
    print(f"  {name}: {count}")
PY
```

Sanity-check the manifest:

```bash
python - <<'PY'
from pathlib import Path
import csv
import os

stem = os.environ["DATASET_STEM"]
manifest = Path("image_sets") / stem / f"{stem}_input.csv"
with manifest.open(newline="") as f:
    rows = list(csv.DictReader(f))

basenames = [Path(r["image_path"]).name for r in rows]
duplicates = len(basenames) - len(set(basenames))
missing = [r["image_path"] for r in rows if not Path(r["image_path"]).is_file()]
clusters = sorted({r["cluster"] for r in rows})

print(f"rows={len(rows)}")
print(f"duplicate_basenames={duplicates}")
print(f"missing_staged_files={len(missing)}")
print(f"clusters={len(clusters)}")
if duplicates or missing:
    raise SystemExit("Manifest check failed")
PY
```

## 4. Generate dashboard features

Full feature extraction runs NIMA, NIQE, and BRISQUE. It is slower, but it produces
the full quality panel and the most comparable `complexity_score`.

```bash
export INPUT_CSV="image_sets/${DATASET_STEM}/${DATASET_STEM}_input.csv"
export FEATURE_CSV="data/${DATASET_STEM}.csv"

python -m precisionai.agriviz.tools.features \
  --input "$INPUT_CSV" \
  --output "$FEATURE_CSV" \
  --image-source fullres \
  --device "$DEVICE"
```

For a fast first pass:

```bash
python -m precisionai.agriviz.tools.features \
  --input "$INPUT_CSV" \
  --output "$FEATURE_CSV" \
  --image-source fullres \
  --device "$DEVICE" \
  --skip-nima
```

For arbitrary image folders, it is normal to see warnings such as
`coco:label_not_found` or missing metadata. RGB-derived features and embeddings still
work. If you provide COCO labels and metadata, the extractor will use them.

Optional COCO layout:

```text
image_sets/my_images/
  images/<staged_stem>.jpg
  labels/<staged_stem>.json
  masks/<staged_stem>.png
  metadata/images_metadata.csv
```

The COCO JSON must use polygon segmentation. Metadata joins by `id`, `original_name`,
or `uuid` matching the staged image stem.

## 5. Generate embeddings

The default dashboard embeddings must be named `data/<DATASET_STEM>.json`.

```bash
export EMBEDDINGS_JSON="data/${DATASET_STEM}.json"

python -m precisionai.agriviz.tools.embeddings \
  --input "$FEATURE_CSV" \
  --output "$EMBEDDINGS_JSON" \
  --model dinov2 \
  --image-source fullres \
  --device "$DEVICE"
```

This writes `data/<DATASET_STEM>.json.provenance.json` beside the embeddings file.

Optional comparison embeddings use the same CSV stem plus a variant suffix:

```bash
python -m precisionai.agriviz.tools.embeddings \
  --input "$FEATURE_CSV" \
  --output "data/${DATASET_STEM}_dinov2_retry.json" \
  --model dinov2 \
  --image-source fullres \
  --device "$DEVICE"
```

The dashboard will list `dinov2_retry` in the 3D view's Compare menu.

## 6. Register the dataset

Append the generated CSV to `precisionai/agriviz/confi.yaml`:

```bash
python - <<'PY'
from pathlib import Path
import json
import os

stem = os.environ["DATASET_STEM"]
name = os.environ.get("DATASET_NAME", stem.replace("_", " ").title())
description = os.environ.get("DATASET_DESCRIPTION", "Generated from a local image folder")
source = f"data/{stem}.csv"
path = Path("precisionai/agriviz/confi.yaml")
text = path.read_text()

if f"source: {source}" in text or f'source: "{source}"' in text:
    print(f"{source} is already registered")
else:
    if text and not text.endswith("\n"):
        text += "\n"
    text += (
        f"\n- name: {json.dumps(name)}\n"
        f"  description: {json.dumps(description)}\n"
        f'  source: {source}\n'
    )
    path.write_text(text)
    print(f"Registered {source}")
PY
```

The result should look like:

```yaml
datasets:
- name: "My Images"
  description: "Generated from a local image folder"
  source: data/my_images.csv
```

## 7. Start the dashboard

```bash
docker compose up --build
```

Open:

```text
http://localhost:5175
```

Load the new dataset card. In the Visual Explorer:

- `Ground Truth` clusters come from the `cluster` column generated from folders.
- `Embeddings` is enabled when `data/<DATASET_STEM>.json` exists.
- Image preview works because the CSV paths point into `image_sets/`, which is mounted
  read-only into the API container.

If the stack was already running before you added or changed `image_sets/`, restart it:

```bash
docker compose up --build -d
```

## 8. Reproducibility checklist

Keep these artifacts together:

```text
image_sets/<DATASET_STEM>/<DATASET_STEM>_input.csv
image_sets/<DATASET_STEM>/<DATASET_STEM>_checksums.sha256
data/<DATASET_STEM>.csv
data/<DATASET_STEM>.json
data/<DATASET_STEM>.csv.provenance.json
data/<DATASET_STEM>.json.provenance.json
```

Capture the code and dependency state:

```bash
git rev-parse HEAD
python -m pip freeze > "image_sets/${DATASET_STEM}/${DATASET_STEM}_pip_freeze.txt"
```

For strictest numeric reproducibility, use the pinned requirements and run on CPU.
GPU runs can have tiny floating-point differences across hardware, drivers, and CUDA
builds.

If you commit generated dashboard files, make sure Git LFS is installed first because
`data/*.csv` and `data/*.json` are LFS-tracked:

```bash
git lfs install
git add data/${DATASET_STEM}.csv data/${DATASET_STEM}.json
git add -f data/${DATASET_STEM}.csv.provenance.json data/${DATASET_STEM}.json.provenance.json
```
