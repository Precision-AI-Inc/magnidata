# task-16 — Close the loop: train on child dataset, relabel export, per-sample run compare

**Depends on:** task-12, task-15
**Commit message:** `Close the training loop with curated retraining and per-sample run comparison`

## Description

Turn findings into the next training run. The existing "save selection as dataset" flow
already creates child datasets; this task connects it to training with lineage recorded
in MLflow, adds a relabel-queue export for suspected label noise, and compares two runs
**per sample** to show which images got easier or harder after the dataset changed.

## Proposed solution

1. **Train on this dataset** — after a child dataset is created (existing
   `POST /api/datasets` flow in `App.tsx`), offer "Train on it": opens `TrainDialog`
   preloaded with that dataset and the params of the currently overlaid run, passing
   `parent_run_id` → tag `magnidata.parent_run_id`. Run detail shows lineage
   ("trained on <child dataset>, from run X" with a link); the runs table gets an optional
   "Parent run" column.
2. **Relabel queue export** — from the current selection, "Export relabel queue" downloads
   a CSV: `image_name, image_path, split, loss_total, loss_<terms>…, hardness, iou, note`
   (note from the existing notes DB when present). Client-side from the merged rows.
3. **Per-sample run comparison** —
   - API `GET /api/runs/compare-samples?a=<run_id>&b=<run_id>&epoch=latest` → CSV
     `image_name, hardness_a, hardness_b, hardness_change, loss_total_change, in_a, in_b`
     (outer join on image_name). Both runs need per-sample artifacts (409 otherwise) and
     ≥ 1 shared image (400 otherwise).
   - UI: in the compare view (task-12), when exactly two runs with per-sample data are
     selected, "Compare on dataset" overlays these columns (via `mergeRunColumns` with a
     `cmp_` prefix) so the 3D view can be coloured by `cmp_hardness_change` with a
     **diverging** palette centred on 0 (dataviz skill).

## Testing

- `api/test_experiments.py` (real sqlite MLflow fixture): compare-samples with partial
  overlap → correct changes and `in_a`/`in_b`; same run twice → all changes 0; no overlap
  → 400; run without per-sample data → 409; invalid ids → 400.
- training launch test: `parent_run_id` becomes the `magnidata.parent_run_id` tag; an
  unknown parent id → 400.
- vitest: relabel CSV builder (column order, escaping of commas/quotes/newlines in notes,
  only selected rows); lineage label helper.
- `npm run build`, `npm test`, `pytest` pass.
- Browser check: brush `hardness > 0.6` → save as dataset → Train on it (`tiny_unet`,
  2 epochs) → when finished, lineage shown; compare with the parent run → "Compare on
  dataset" → colour by `cmp_hardness_change` with a diverging legend; export the relabel
  CSV and open it. Screenshots reviewed before committing.
