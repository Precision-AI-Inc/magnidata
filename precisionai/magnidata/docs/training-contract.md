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
| `magnidata.loss_terms` | Comma-separated term names |
| `magnidata.per_sample_schema` | Schema version; presence means per-sample artifacts exist |
| `magnidata.scored_epochs` | Comma-separated epochs whose per-sample file is logged |
| `magnidata.cancel_requested` | `true` when the UI asks to cancel |
| `magnidata.parent_run_id` | Lineage for retraining |
| `magnidata.worker` | Id of the worker that claimed the run |

Run statuses: `SCHEDULED`, `RUNNING`, `FINISHED`, `FAILED`, `KILLED`.

## Metric keys

These match pai-seed's `Trainer`. Step is the epoch, except `Learning_rate/lr` which uses
the global step.

- `train_loss/loss`, `train_loss/<term>`
- `val_loss/loss`, `val_loss/<term>`
- `Learning_rate/lr`
- `val_metric/miou`, `val_metric/iou_<class>`

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
