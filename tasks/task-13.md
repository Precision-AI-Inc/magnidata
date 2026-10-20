# task-13 — Dashboard: run overlay — per-sample loss columns as axes

**Depends on:** task-07, task-10
**Commit message:** `Show per-sample run losses as dashboard axes`

## Description

This is where training meets the dataset: choose a run with per-sample artifacts for the
current dataset, and its derived columns (`loss_dice`, `loss_cce_rank`, `hardness`,
`split`, …) appear as regular columns in the parallel coordinates, histograms, 3D
colour-by, table, and Schema dictionary — without re-downloading the dataset CSV.

## Proposed solution

- `api.ts`: `getRunColumns(runId, epoch)` (CSV text, parsed with the parser `useCSV`
  already uses), `getRunMatch(runId, datasetSource)`, `linkRun(runId, datasetSource)`.
- `lib/mergeRunColumns.ts` — **pure**
  `mergeRunColumns(data, runRows, imageKeyCol, prefix = "") -> {data, added, matched, unmatched}`:
  joins on image basename (same rule as the API), adds columns, types them (`split`
  categorical, the rest numeric), and **replaces** previously merged run columns when the
  run/epoch changes (tracked in `data.runColumns`). Reuse/extract the add-column path in
  `useCSV.ts` (the k-means `clusterCol` code, ~line 272) instead of a parallel mechanism.
- `hooks/useRunOverlay.ts` — `{runId, epoch}`, `select(runId)`, `clear()`, refetch on change.
- Visual Explorer toolbar: **Run** dropdown listing runs for the current dataset
  (`/api/runs?dataset=…`, only `has_per_sample`), with status badge and scored-epoch count;
  "None" removes the columns. Runs Table rows (task-10) with per-sample data get an
  "Overlay on dataset" action that loads the dataset if needed and selects the run.
- For an external run without `magnidata.dataset_source`, the run detail (task-11) offers
  "Link to current dataset" → `POST /api/runs/<id>/link`, showing the match report first.
- Axis pickers / ParallelCoords / Histograms / Graph3D colour-by: run columns grouped
  under a **"Training"** heading (prefix `loss_`, plus `hardness`, `iou`, `confidence`,
  `split`), sequential palette for loss columns.
- `data/colDescs.ts`: descriptions for the derived column patterns (`loss_<t>`, `_mean`,
  `_delta` (> 0 = improved), `_slope` (< 0 = getting easier), `_rank`, `hardness`).
  Training columns are not in `OUTPUT_COLUMNS`; touch `tools/test_schema_sync.py` only if
  it would otherwise fail.
- Non-blocking notice when the match rate is < 100% ("312 / 451 samples have training data").

## Testing

- vitest `lib/mergeRunColumns.test.ts`: basename join (full dataset paths vs basenames);
  unmatched rows get `null` and are counted; re-merge with another epoch replaces values
  without duplicating columns; clear restores original column lists; `split` categorical.
- `npm run build` and `npm test` pass.
- Browser check (`chrome-devtools-axi`): load AgriStress-500 (or the synthetic fixture
  dataset), pick a finished run in the Run dropdown → `loss_total`, `loss_dice_rank`,
  `hardness` selectable as parallel-coordinates axes and 3D colour-by; brushing
  `hardness > 0.6` filters the table; console clean. Screenshots reviewed before committing.
