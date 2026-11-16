# MagniData

MagniData is Precision AI's open-source explorer for image datasets. It turns a folder of images (optionally with
COCO segmentation annotations) into a table of per-image quality, complexity and coverage metrics plus embeddings,
and shows them in a dashboard: browse images and segmentation masks, filter by any metric, and view the embedding
space in 3D (PCA, t-SNE, clusters).

- **Live demo, no install:** <https://huggingface.co/spaces/precisionaiinc/magnidata> (COCO128 and AgriStress-500
  loaded, read-only)
- **Source, full stack and issues:** <https://github.com/Precision-AI-Inc/magnidata>
- **Technical report:**
  <https://github.com/Precision-AI-Inc/magnidata/blob/main/docs/paper/magnidata-technical-report.pdf>

## What this package contains

`pip install magnidata` installs the Python toolkit: the feature extractor, the embeddings generator and the
training feedback loop, importable as `precisionai.magnidata`.

It does not run the dashboard. The Flask API and the React dashboard run from a source checkout with Docker
Compose; see the
[quickstart](https://github.com/Precision-AI-Inc/magnidata#quickstart--run-the-full-stack).

## Install

```bash
pip install magnidata              # feature extraction (pixel, COCO and metadata features)
pip install "magnidata[ml]"        # + embeddings and learned image-quality (NIMA) scores; installs torch
pip install "magnidata[training]"  # + the reference segmentation trainer; installs torch
```

Python 3.10 to 3.13.

## Command-line tools

| Command | What it does | Needs |
|---|---|---|
| `magnidata-features` | CSV of image paths in, one row of features per image out, plus a provenance sidecar | base install; `[ml]` unless `--skip-nima` |
| `magnidata-embeddings` | CSV of image paths in, embeddings JSON out (DINOv2 by default) | `[ml]` |
| `magnidata-train` | Train a segmentation model and log the run, with per-image losses, to MLflow | `[training]` |
| `magnidata-attach` | Attach per-image losses from your own trainer to an MLflow run | base install |

Each command takes `--help`. The same tools run as modules, for example
`python -m precisionai.magnidata.tools.features`.

```bash
magnidata-features --input images.csv --output features.csv --skip-nima
magnidata-embeddings --input images.csv --output embeddings.json
```

The input CSV needs an `image_path` column. Images are expected at `<DATASET>/images/<STEM>.png`, with optional
COCO labels at `<DATASET>/labels/<STEM>.json`. The output CSV and embeddings JSON are the formats the dashboard
loads.

## Documentation

- [Feature extractor: inputs, flags and output columns](https://github.com/Precision-AI-Inc/magnidata/blob/main/precisionai/magnidata/tools/README.md)
- [Data contract: the CSV and embeddings JSON formats](https://github.com/Precision-AI-Inc/magnidata/blob/main/precisionai/magnidata/docs/data-contract.md)
- [Recipe: from a folder of images to a dashboard](https://github.com/Precision-AI-Inc/magnidata/blob/main/precisionai/magnidata/docs/recipes/from-images-to-dashboard.md)
- [Training contract: how runs and per-image losses are stored](https://github.com/Precision-AI-Inc/magnidata/blob/main/precisionai/magnidata/docs/training-contract.md)
- [Changelog](https://github.com/Precision-AI-Inc/magnidata/blob/main/CHANGELOG.md)

## License

Apache License 2.0. Copyright 2026 Precision AI.
