# task-03 — Per-sample segmentation loss terms

**Depends on:** task-00
**Commit message:** `Add per-sample segmentation loss terms`

## Description

Typical trainers (including pai-seed's `Trainer`, which logs `outputs.loss_dict`) reduce
each loss over the batch, so per-image values never exist. The feature needs every loss
term computed **per sample** (shape `[B]`), while still producing the usual scalar for
backprop. This task implements the segmentation terms in that form.

## Proposed solution

`precisionai/magnidata/training/losses.py` (torch; imported only by trainer code, never by
`training/__init__.py`; optional-import flag with an install hint):

- `cce_per_sample(logits, target, ignore_index=255) -> Tensor[B]` — pixel-wise
  cross-entropy with `reduction="none"`, averaged over the valid pixels of each image
  (zero valid pixels → 0, not NaN).
- `dice_per_sample(logits, target, num_classes, ignore_index=255, eps=1e-6) -> Tensor[B]` —
  soft multi-class Dice loss (`1 - mean_c dice_c`) per image, softmax vs one-hot target,
  ignoring `ignore_index` pixels and excluding classes absent from both prediction and
  target in that image (document the choice in the docstring).
- `bce_fg_per_sample(logits, target, background_index=0) -> Tensor[B]` — foreground vs
  background BCE from `1 - p(background)`; isolates crop/soil separation.
- `iou_per_sample(logits, target, num_classes) -> Tensor[B]` — hard mean IoU, no grad;
  emitted as the `iou` extra column. Also `confusion_counts(logits, target, num_classes)`
  so the trainer can accumulate dataset-level mIoU / per-class IoU for MLflow metrics.
- `class SegmentationCriterion(terms, weights)` → `LossOutput(total: Tensor scalar,
  per_sample: dict[str, Tensor[B]])` with `per_sample["total"] = Σ w_t · term_t` and
  `total = per_sample["total"].mean()`.
- `LOSS_REGISTRY` mapping names → functions so presets (task-08) can validate term names
  without importing torch (put the **names** in `contract.py`, the functions here).

## Testing

`precisionai/magnidata/training/test_losses.py` (CPU, tiny tensors, seeded;
`pytest.importorskip("torch")`):
- every function returns `[B]` for `B=3`.
- perfect prediction → CCE ≈ 0, Dice ≈ 0, IoU = 1; inverted prediction → Dice ≈ 1, IoU = 0.
- `cce_per_sample(...).mean()` equals `F.cross_entropy(..., reduction="mean")` within 1e-6
  when no pixels are ignored and images are equal-sized.
- per-sample independence: changing image 2's logits leaves images 0 and 1 unchanged.
- an all-ignored image yields 0, not NaN.
- `confusion_counts` summed over two batches equals the counts of the concatenated batch.
- criterion total equals the weighted sum; `backward()` yields finite grads; unknown term
  → `pytest.raises(ValueError, match="unknown loss term")`.
