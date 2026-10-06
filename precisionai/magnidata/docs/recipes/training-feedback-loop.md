# Recipe: the training feedback loop

This recipe trains a segmentation model from the dashboard, then uses what the model
struggles with to improve the dataset and train again.

```
dataset ──► Train ──► MLflow run ──► per-sample losses ──► hard samples
   ▲                                                            │
   └──────────── curated child dataset ◄───── review / relabel ─┘
```

Everything runs locally from one command. MLflow is the system of record: runs,
params, metric curves, per-sample loss files, predictions and checkpoints all live there.

## 1. Start the stack

```bash
docker compose up --build
```

| Service | URL | What it does |
|---|---|---|
| portal | http://localhost:5175 | the dashboard |
| api | http://localhost:5051 | Flask API; proxies MLflow, never imports torch |
| mlflow | http://localhost:5052 | the MLflow UI and tracking server |
| trainer | (no port) | claims scheduled runs and trains them; CPU by default, GPU with the override below |

Training needs a dataset **with masks** (`<dataset>/images/<stem>.png` next to
`<dataset>/masks/<stem>.png`, colour-coded per class, black = background). The
AgriStress-500 demo has them: open **BYOD → Load Demo → AgriStress-500**, then load it from
**MagniData**. Training on the CPU is slow for 500 high-resolution images; use a small
`image_size` (128) and few epochs for a first try, or switch to the GPU as described next.

### Training on an NVIDIA GPU

The base `docker-compose.yml` is CPU-only so it runs anywhere. On a GPU host (with the
NVIDIA container toolkit) add the override once, in the git-ignored `.env` at the repo root,
and keep using plain `docker compose up --build`:

```
COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml
```

The override builds the trainer with CUDA torch (torch 2.14.0 is published on the **cu130**
wheel index; cu128 stops at 2.11 — RTX 50xx / Blackwell needs cu128 or newer, and the host
driver must support CUDA 13) and reserves the GPU. It also sets `MAGNIDATA_TRAIN_DEVICE=cuda`, so a
trainer that cannot see the GPU fails the run (`CUDA requested but not available`) instead
of silently training on the CPU. Verify it:

```
docker compose exec trainer python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The first line of a run's `logs/trainer.log` (uploaded every 30 s, so it is near-live in run
detail) reads `device=cuda:0 (…)`. Run detail and the Experiments table show a **GPU/CPU**
badge, and `ETA` while the run is training.

### What makes training fast

- **Resized-image cache.** Every image and mask is decoded once, resized to `image_size` and
  stored under `data_user/.train_cache/<size>/`; later epochs, validation and scoring read the
  cache. The cache is bounded by `TRAIN_CACHE_MAX_MB` (default 4096, least recently used
  entries go first) and keyed by file path, mtime and size, so edited images are decoded again.
- **Loader processes** (`num_workers`, default one per spare core up to 8) and **mixed
  precision** (`amp`, on by default, GPU only; bfloat16 where supported). Per-sample losses are
  always computed in float32. Both are in the Train dialog.
- Progress goes to the log (`epoch 3/20 train batch 40/45 …`, `done … ETA …`) and to the
  metrics `perf/epoch_seconds` and `perf/samples_per_sec`.

## 2. Train

With a dataset open, press **Train** in the tab bar. Pick a preset, adjust the allowed
parameters (the dialog shows each bound and the server re-checks them), and start.

- The Header shows `Training · queued`, then `Training · epoch 3/20`; a toast reports how it ended.
- The button is disabled, with the reason as a tooltip, when MLflow or the trainer is
  unavailable, no dataset is open, or another run is active (one UI run at a time).
- **Cancel run** in the run detail stops a scheduled or running run.

Each epoch the trainer logs aggregate metrics, then re-scores **every** image in eval mode
without augmentation and logs one file of per-image losses. Losses seen while training are
noisy (augmentation, weights changing mid-epoch); the scoring pass is what the dataset
axes come from.

## 3. Watch it in Experiments

Open the **Experiments** tab (also in the header, no dataset needed). Pick experiments on
the left; filter by status, dataset or a run name; sort by any metric or param column and
choose the columns. Click a run for its detail: loss curves with train and val overlaid per
term, IoU and learning-rate charts (smoothing, log scale, relative time), params and tags,
and an artifact tree with image previews. Select 2–10 runs and press **Compare** for
overlaid curves, a final/best table and the params that differ. The comparison is in the
URL, so reloading keeps it.

## 4. Overlay the losses on the dataset

Back in a dataset view, the **Run** dropdown lists every run linked to the dataset, newest
first, and refreshes by itself every 10 s while one is queued or training. A run that cannot
be picked yet is greyed out with the reason: `queued`, `training · first losses after epoch 1
scoring`, `failed — no per-sample data`. As soon as the trainer has logged its train/val
split the run is pickable (`split only`), and the run you just launched is **overlaid
automatically** (a toast says so); the loss columns join as soon as epoch 1 is scored, whatever
`score_every` is. Pick another run, or **None**, at any time.

Its per-sample columns behave like any other: parallel-coordinates axes, 3D colour-by,
histograms, table columns and filters, grouped under **Training**. Parallel coordinates adds
`loss_total`, `loss_total_mean` and `hardness` next to your photometric axes, and **High-loss
samples** in the Fields panel's Training group brushes the persistently hard samples
(`hardness ≥ 0.6`, or the top fifth of `loss_total_mean`), so the table, histograms and 3D
view show what those images look like. The **epoch scrubber** next to it moves through
the scored epochs (or follows `latest` while the run trains); **play** steps through them.

| Column | Reading |
|---|---|
| `loss_<term>` | the term's loss for this image at the selected epoch |
| `loss_<term>_mean` | average over the scored epochs so far: how hard the image is across training |
| `loss_<term>_delta` | first value minus current. **Positive = improved** |
| `loss_<term>_slope` | trend over the last 5 scored epochs. **Negative = getting easier** |
| `loss_<term>_rank` | percentile rank within the image's split (1 = hardest) |
| `hardness` | share of scored epochs in which the image was in the worst 20% of its split by total loss |
| `iou`, `confidence` | mean IoU of the predicted mask; mean top-class probability |
| `split` | train / val / test of the run (ranks are computed within a split) |

### The train/val partition

The overlay brings `split` along: it becomes the first categorical axis, colours the points
train/val (fixed colours in every view, unless you chose a colour yourself), and the toolbar
shows the counts (`train 355 · val 96`). Histograms can **stack by split** to compare the
photometric distributions of the two sets, the 3D view can group by `split` (or any column
with at most 12 values), and the table pins `split` next to the image with filter chips.

Combinations worth looking for:

- **High `hardness`, high `loss_total_mean`, `slope` near 0**: persistently hard. Look at
  these images first; many are label noise or a genuinely rare case.
- **High `loss_dice_rank` but low `loss_cce_rank`**: pixels are mostly classified right but the
  region's *shape* is off: suspect a coarse or shifted mask rather than class confusion.
- **High `loss_cce_rank` and low `confidence`**: the model is unsure *what* it is looking at: ambiguous
  classes or too few examples.
- **Large positive `delta`, negative `slope`**: still learning; wait for more epochs before judging.
- **High on `val` but not on `train`**: overfitting or a split that differs from training.

Open an image to see its loss curve per term with the viewed epoch marked, and a
**Prediction** view of the model's mask. Only the hardest samples of each epoch get a saved
prediction; the view falls back to the nearest earlier epoch and says so.

## 5. Curate and retrain

1. Select images (brush `hardness` on its axis, or add images to the basket from the
   preview with **A**) and **Create dataset**. The original dataset is never changed.
2. Press **Train on it** in the banner. The dialog opens on the new dataset with the
   overlaid run's settings, and the new run records the old one as its **parent**.
3. **Export relabel queue** downloads the selected images with losses, hardness and your
   notes, hardest first: a worklist for whoever fixes the labels.

## 6. Did it help?

In Experiments, tick the parent and the new run and press **Compare**: the curves tell you
about the model. For the data, press **Compare on dataset**: each image gets
`cmp_hardness_change` and `cmp_loss_total_change` (new minus old, positive = got harder),
coloured on a diverging scale centred on zero. Images only one run saw are left empty.
Run detail shows the lineage ("Trained on <dataset>, from run <parent>") and the table can
show a **Parent run** column.

## Runs from other trainers

Any run in the local MLflow appears in Experiments, including pai-seed runs; see
[Logging host-side runs to the local MLflow](../training-contract.md#logging-host-side-runs-to-the-local-mlflow).
To give such a run per-sample axes, follow
[Emitting per-sample losses from your own trainer](../training-contract.md#emitting-per-sample-losses-from-your-own-trainer),
then open its detail and use **Link to current dataset**, which first reports how many
images match.
