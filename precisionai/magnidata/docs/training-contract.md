# Training contract

Training runs are stored in MLflow. Any run that follows this contract can be browsed in
MagniData, and runs that also log per-sample artifacts get dataset axes. The contract
lives in `precisionai/magnidata/training/contract.py`, which imports neither torch nor
mlflow, so external trainers (for example pai-seed) can depend on it directly.

Schema version: `magnidata-training/1.0.0`.

## Tags

| Tag | Meaning |
|---|---|
| `magnidata.dataset_source` | Dataset CSV path relative to `DATA_ROOT`; the join key to a MagniData dataset |
| `magnidata.dataset_name` | Display name |
| `magnidata.origin` | `ui` (launched from MagniData) or `external` |
| `magnidata.preset` | Preset key |
| `magnidata.engine` | `reference` (the bundled trainer) or `agritune`; absent means `reference`. Selects the trainer the worker starts |
| `magnidata.loss_terms` | Comma-separated term names |
| `magnidata.per_sample_schema` | Schema version; presence means per-sample artifacts exist |
| `magnidata.scored_epochs` | Comma-separated epochs whose per-sample file is logged |
| `magnidata.cancel_requested` | `true` when the UI asks to cancel |
| `magnidata.parent_run_id` | Lineage for retraining |
| `magnidata.worker` | Id of the worker that claimed the run |
| `magnidata.has_split` | `true` once `magnidata/split.csv` is logged (before any epoch is scored) |
| `magnidata.device` | Device the trainer used, e.g. `cuda:0 (NVIDIA GeForce RTX 5070 Laptop GPU)` or `cpu` |
| `magnidata.eta_seconds` | Trainer's estimate of the remaining time, updated each epoch |

Run statuses: `SCHEDULED`, `RUNNING`, `FINISHED`, `FAILED`, `KILLED`.

## Metric keys

These match pai-seed's `Trainer`. Step is the epoch, except `Learning_rate/lr` which uses
the global step. Runs trained through the AgriTune engine log **every** metric once per epoch, including
the learning rate (its last value of the epoch) and `train_loss/loss` (the mean over the epoch's steps), so a
run costs a few dozen MLflow points instead of thousands. The loss components (`train_loss/<term>`,
`val_loss/<term>`) are added on the epochs that are scored.

- `train_loss/loss`, `train_loss/<term>`
- `val_loss/loss`, `val_loss/<term>`
- `Learning_rate/lr`
- `val_metric/miou`, `val_metric/iou_<class>`
- `perf/epoch_seconds`, `perf/samples_per_sec` (reference trainer only)

## Artifact layout

```
magnidata/split.csv                          image_name,split   (train|val|test)
magnidata/palette.json                       class index -> RGB + name
magnidata/per_sample/epoch_0001.csv          one row per scored sample
magnidata/predictions/epoch_0001/<stem>.png  optional, hardest samples only
checkpoints/last.pt                          optional, trainer-owned
```

## Per-sample columns

Required: `image_name`, `split`, `loss_total`. One `loss_<term>` column per term; term
names match `^[a-z][a-z0-9_]{0,31}$`. Optional: `confidence`, `iou`. `image_name` is the
basename of the image path, the same key used to join embeddings to CSV rows.

```csv
image_name,split,loss_total,loss_dice,loss_cce,confidence,iou
img_0001.png,train,0.412,0.301,0.111,0.93,0.81
img_0002.png,val,0.870,0.640,0.230,0.71,0.44
```

## Writing the files

Use `LossRecorder` (`training/recorder.py`) so every trainer writes identical files. It
needs only numpy; values may be numpy arrays, lists or detached torch tensors.

```python
sink = MlflowSink(run_id)                      # training/mlflow_sink.py
rec = LossRecorder(["dice", "cce"], sink, scratch_dir)
rec.write_split({"a.png": "train", "b.png": "val"})
rec.begin_epoch(1)
rec.add_batch(names, "train", {"total": total, "dice": dice, "cce": cce})
rec.end_epoch()                                # logs magnidata/per_sample/epoch_0001.csv
```

For a run that already exists, attach files after the fact; a long-format file with an
`epoch` column is split into one artifact per epoch, and nothing is logged unless the whole
file validates:

```bash
python -m precisionai.magnidata.training.attach --run-id <id> --epoch 12 --file losses.csv --split split.csv
```

## Logging host-side runs to the local MLflow

The compose stack publishes MLflow on port 5052. A trainer running on the host (for example
pai-seed through `PAIExperiment`, which uses MLflow by default) logs to it with:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5052
```

Those runs show up in the **Experiments** tab with their params, metric curves and
artifacts. Only runs that carry the tags above are tied to a dataset; for others use
**Link to current dataset** in the run detail.

Where the data lives: the `mlflow` service keeps its database and artifacts in the named
volume `magnidata_magnidata-mlflow` (mounted at `/mlflow`). It survives
`docker compose down` and `up`.

```bash
# back up
docker run --rm -v magnidata_magnidata-mlflow:/mlflow -v "$PWD":/backup alpine \
    tar czf /backup/mlflow-backup.tgz -C /mlflow .

# reset (deletes every run; keep the compose project stopped first)
docker compose down
docker volume rm magnidata_magnidata-mlflow
```

Downloaded per-sample files are cached in `data_user/.mlflow_cache` (bounded by
`MLFLOW_CACHE_MAX_MB`, default 2048) and can be deleted at any time.

## Emitting per-sample losses from your own trainer

Give each loss term per image (`reduction="none"`, then average over pixels), run a
separate scoring pass in eval mode without augmentation, and hand the numbers to the
recorder. It needs only numpy and works with torch tensors.

```python
import torch
import torch.nn.functional as F

from precisionai.magnidata.training.mlflow_sink import MlflowSink
from precisionai.magnidata.training.recorder import LossRecorder

sink = MlflowSink(run_id)                       # uses MLFLOW_TRACKING_URI
recorder = LossRecorder(["cce", "dice"], sink, scratch_dir)
recorder.write_split({"a.png": "train", "b.png": "val"})

for epoch in range(1, epochs + 1):
    train_one_epoch(model)
    model.eval()
    recorder.begin_epoch(epoch)
    with torch.no_grad():
        for names, images, masks in score_loader:        # no augmentation
            logits = model(images)
            cce = F.cross_entropy(logits, masks, reduction="none").mean((1, 2))
            dice = per_image_dice_loss(logits, masks)    # shape [B]
            recorder.add_batch(
                names, "train",                           # one split label, or one per image
                {"total": cce + dice, "cce": cce, "dice": dice},
            )
    recorder.end_epoch()                                   # logs magnidata/per_sample/epoch_NNNN.csv
    model.train()
```

Image names are the basenames of the dataset's image paths. For runs that already
finished, attach a file of per-sample losses afterwards (wide format with `--epoch`, or long
format with an `epoch` column):

```bash
python -m precisionai.magnidata.training.attach --run-id <id> --file losses.csv --split split.csv
```

Finally, open the run in the **Experiments** tab and press **Link to current dataset** (with
the dataset open). The report shows how many images match before anything is written.
