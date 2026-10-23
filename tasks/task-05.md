# task-05 — Reference trainer logging to MLflow

**Depends on:** task-02, task-03, task-04
**Commit message:** `Add reference segmentation trainer that logs to MLflow`

## Description

Implement a runnable trainer that takes an MLflow run (created by the API in task-08, or
created on the fly for CLI use), trains a small segmentation model, and logs everything to
that run: params, per-epoch aggregate metrics (losses per term, mIoU, per-class IoU, LR),
per-sample loss files from an **eval-mode, no-augmentation scoring pass** over train and
val, predictions for the hardest samples, and a checkpoint. In-loop training losses are
noisy (dropout, augmentation, weights changing mid-epoch), so the scoring pass is the
source of truth for the dataset axes.

## Proposed solution

`precisionai/magnidata/training/train.py`, runnable as
`python -m precisionai.magnidata.training.train --run-id <mlflow_run_id> --data-root <path> [--device auto]`
or, without `--run-id`, `--dataset <source> --preset tiny_unet --experiment <name> [--param k=v …]`
(creates the run itself, `origin=external`).

1. Read params and tags from the MLflow run (`magnidata.dataset_source`,
   `magnidata.preset`, `magnidata.loss_terms`, params such as `epochs`, `lr`, …).
   Set status `RUNNING` (`MlflowClient.update_run`).
2. Resolve samples, build palette, make split (task-04); `recorder.write_split`,
   `recorder.write_palette`.
3. Build the model from the preset:
   - `tiny_unet` — small in-repo UNet (≈ 0.5M params, no downloads) — tests and CPU.
   - `lraspp_mobilenet` — `torchvision.models.segmentation.lraspp_mobilenet_v3_large`
     with `num_classes` from the palette; ImageNet backbone weights unless
     `params.pretrained=false`.
4. Train with AdamW + cosine LR and `SegmentationCriterion` (task-03). Log to MLflow with
   the contract's metric keys:
   - `Learning_rate/lr` every N steps (step = global step);
   - per epoch: `train_loss/loss`, `train_loss/<term>`, `val_loss/loss`, `val_loss/<term>`,
     `val_metric/miou`, `val_metric/iou_<class>` (step = epoch, from `confusion_counts`);
   - `best_epoch` / `best_val_loss` when improved.
   Use `log_batch` to keep request count low.
5. Every `score_every` epochs (and always on the last epoch): scoring pass with
   `model.eval()` and `torch.no_grad()` over train+val →
   `recorder.add_batch(names, split, per_sample, extras={"iou": …, "confidence": …})`
   (`confidence` = mean max-softmax over pixels) → `recorder.end_epoch()`.
6. Predictions for the **top-K (param `top_k_predictions`, default 50) hardest by
   `loss_total`** in that scored epoch: coloured masks via the palette, logged under
   `magnidata/predictions/epoch_XXXX/`.
7. Log `checkpoints/last.pt` at the end (and every scored epoch if `params.checkpoint_every_score`).
8. Between batches, `recorder.cancel_requested()` → status `KILLED`, clean exit 0.
   Any exception → status `FAILED`, tag `magnidata.error` = first line of the message,
   traceback to stdout, exit ≠ 0. Success → `FINISHED`.
9. Determinism: seed torch/numpy/python from `params.seed`.

Keep functions small (ruff C90 ≤ 10): `build_model`, `train_epoch`, `evaluate`,
`score_epoch`, `log_predictions`, `run`.

Add `[project.optional-dependencies] training = ["torch==…", "torchvision==…", "mlflow-skinny==3.11.1"]`
in `pyproject.toml` (versions matching `precisionai/magnidata/requirements.txt`). Not in
`api/requirements.txt`.

## Testing

`precisionai/magnidata/training/test_train.py` (real sqlite MLflow fixture, synthetic
dataset fixture, CPU, `tiny_unet`, image_size 32):
- 2 epochs, `score_every=1` → run `FINISHED`; metric history for `train_loss/loss` has 2
  points; `val_metric/miou` present; artifacts `magnidata/per_sample/epoch_0001.csv` and
  `epoch_0002.csv` with one row per sample; `split.csv`, `palette.json` present;
  ≤ K prediction PNGs; tag `magnidata.scored_epochs == "1,2"`.
- learnable fixture (solid-colour masks): epoch-2 mean `loss_total` < epoch-1.
- cancel tag set before start → status `KILLED`, exit 0.
- missing dataset → `FAILED` with `magnidata.error` tag, exit ≠ 0.
- CLI mode without `--run-id` creates a run tagged `origin=external`.
- Keep the e2e test under 30 s on CPU so it runs by default (else mark `integration`).
