# task-31 — Audit: predictions must never show ground-truth masks

**Status:** completed

**Commit message:** `Keep prediction and ground-truth previews separate`

## Findings

- API: `/api/annotation/{mask,overlay}` read the dataset's annotation (COCO labels, else `masks/<stem>.png`);
  `/api/runs/<id>/prediction` reads only the run's MLflow artifact `magnidata/predictions/epoch_NNNN/<stem>.png`.
  Predictions are written to MLflow only (never into the dataset), so the two never share a file. A cross-endpoint
  test proves different pixels and that an image without a prediction returns 404.
- Bug (fixed): in `ImageModal`, the `imgSrc` ternary ended in the annotation overlay URL, so the Prediction tab with no
  saved prediction displayed the ground truth, and the "No saved inference" screen was unreachable.
- Colours: prediction PNGs use the palette derived from the ground-truth mask colours, so classes share colours.

## Open risks (not changed)

- The Mask/Overlay tabs prefer COCO label JSON over raster masks, while the trainer's target is the raster mask
  (`dataset_check.find_mask`). If an image has both and they differ, the previewed ground truth is not the training target.
- Predictions are keyed by image basename; same-named images in different folders (e.g. after merging datasets) would
  share one prediction.
