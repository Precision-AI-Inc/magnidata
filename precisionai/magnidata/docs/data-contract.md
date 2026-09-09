# Data contract

magnidata's dashboard consumes two kinds of files from `data/`: a **multiparametric
CSV** (one row per image, numeric/categorical columns for the parallel-coordinates
and 3D views) and, optionally, an **embeddings JSON** (one vector per image, for the
embeddings-space 3D view). Both can either be produced by the tools in this repo or
brought in from elsewhere, as long as they conform to the shapes below.

## Multiparametric CSV

**Source of truth:** `tools/schema.py`'s `OUTPUT_COLUMNS` — currently
`FEATURE_SCHEMA_VERSION = "dataviz-features/2.1.0"`, 41 columns grouped by concept
(identity, coverage, instances, classes, exposure, focus, quality panel, white
balance, learned IQA, composite difficulty, camera metadata, agronomic passthrough).
`dashboard/src/data/descriptions.json` + `colDescs.ts`'s `EXTRA` map document every
one of those columns in plain language for the Schema tab; `tools/test_schema_sync.py`
fails `pytest precisionai/magnidata/tools` if the two ever drift apart again.

### Generate it with the built-in tool

```bash
pip install -r precisionai/magnidata/requirements.txt
python -m precisionai.magnidata.tools.features --input images.csv --output features.csv
```

See `tools/README.md` for the full flag reference (`--skip-nima`, `--image-source`,
`--device`, `--limit`) and the on-disk layout convention it expects
(`<DATASET>/images/<STEM>.png` + sibling `labels/`/`metadata/`). A
`<output>.provenance.json` sidecar is written alongside, recording exactly what
produced the file (code version, library versions, input hash) for reproducibility.

### Bring your own

Your CSV must contain the columns listed in `tools/schema.py`'s `OUTPUT_COLUMNS` (a
subset also works — the dashboard type-sniffs each column at load time rather than
requiring every one). Add an entry to `confi.yaml`:

```yaml
datasets:
  - name: "My Dataset"
    description: "..."
    source: data/my-dataset.csv
```

## Embeddings JSON

**Format**, exactly what `api/app.py` reads (`_load_emb_by_name`):

```json
{"embeddings": {"<image_basename>": [0.0123, -0.045, "... one float per dimension"]}}
```

Keys are `os.path.basename(image_path)` — not full paths — matched against the CSV's
image-path column by the API. All vectors in one file must share the same dimension;
different embedding files (e.g. different models) can use different dimensions.

**Naming convention** (by filename, not registered in `confi.yaml`): for a dataset
whose CSV is `data/<stem>.csv`, its default embeddings live at `data/<stem>.json` —
this is what the dashboard loads by default and what enables the "Embeddings" toggle
in the 3D view. A second embeddings file for the same dataset, from a different
model, goes to `data/<stem>_<variant>.json` (e.g. `data/datalake_4k_dinov2.json`) and
appears as a "Compare" option that overlays as a second point cloud, linkable back to
the original layout.

### Generate it with the built-in tool

```bash
python -m precisionai.magnidata.tools.embeddings \
    --input images.csv --output data/datalake_4k_dinov2.json --model dinov2
```

Default backbone is DINOv2 (`vit_small_patch14_dinov2.lvd142m` via `timm`, 384-dim),
already covered by `precisionai/magnidata/requirements.txt` — no extra install. See
`tools/embedding_models.py` for the model registry; adding a different backbone
(including a future agriculture-trained checkpoint) means adding one entry there.
A `<output>.provenance.json` sidecar is written alongside, same as the feature
extractor.

### Bring your own

Any precomputed vectors — from any model, generated anywhere — work as long as they
conform to the JSON shape above and the naming convention. Drop the file into `data/`;
no `confi.yaml` entry is needed (embeddings are discovered by filename against the
dataset's CSV stem, not registered separately).
