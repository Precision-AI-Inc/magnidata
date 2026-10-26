# task-15 — Epoch scrubber, per-sample loss history, prediction overlay

**Depends on:** task-13, task-14
**Commit message:** `Add epoch scrubber and per-sample loss history to image preview`

## Description

Answer "is this sample getting easier, and what does the model actually predict?" at two
levels: dataset-wide, by scrubbing the overlaid run through its scored epochs and watching
the axes / 3D colours change; and per image, in the existing preview modal, with the
sample's loss trajectory per term and the model's predicted mask.

## Proposed solution

Backend (`experiments` blueprint, using the artifact cache from task-07):
- `GET /api/runs/<run_id>/samples/<image_name>` →
  `{split, epochs: [...], terms: {total: [...], dice: [...], …}, iou: [...], prediction_epochs: [...]}`
  (prediction epochs = those with `magnidata/predictions/epoch_XXXX/<stem>.png`; list once
  per epoch and cache).
- `GET /api/runs/<run_id>/prediction?image=<name>&epoch=<n|latest>&alpha=0.5` → JPEG of
  the dataset image with the predicted mask composited. Extract the compositing logic
  behind `/api/annotation/overlay` (`_img_from_bytes`, `_mask_as_rgba`, alpha) into a
  shared helper and reuse it — do not copy it. 404 when that sample has no prediction for
  that epoch (only top-K are logged).

Frontend:
- `components/EpochScrubber.tsx` in the Visual Explorer toolbar when a run is overlaid:
  slider over `scored_epochs`, "latest" toggle, play button (advances every 1.5 s).
  Debounce fetches; keep previous columns until new ones arrive (no flicker).
- `ImageModal.tsx`: when a run is overlaid, a "Training" section with a small line chart
  per term across epochs (selected epoch marked), current `hardness` and rank values, and
  a **Prediction** view mode next to image/mask/overlay using the existing opacity slider;
  disabled with a tooltip when no prediction exists for that sample/epoch (falls back to
  the nearest earlier epoch with a prediction, labelled as such).

## Testing

- `api/test_experiments.py` additions (real sqlite MLflow fixture): sample history arrays
  aligned with epochs; unknown sample → 404; prediction endpoint returns `image/jpeg` for a
  logged prediction, 404 otherwise; traversal in `image` → 400.
- `api/test_app.py`: the annotation overlay endpoint still behaves identically after the
  helper extraction.
- vitest for pure helpers (nearest scored epoch for a slider value; nearest earlier
  prediction epoch).
- `npm run build`, `npm test`, full `pytest` pass.
- Browser check (`chrome-devtools-axi`) on a run with ≥ 3 scored epochs: sliding changes
  3D colours and the `loss_total` axis distribution; play/stop works; open a high-hardness
  image → loss chart + Prediction view render; open an easy image → Prediction disabled
  with tooltip. Screenshots reviewed before committing.
