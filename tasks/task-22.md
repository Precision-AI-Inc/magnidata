# task-22 — Show the train/val partition in the dashboard

**Depends on:** task-21
**Commit message:** `Show the train/val split across dashboard views`

## Description

The backend produces and serves `split` correctly (per-sample CSV and `split.csv` both
carry it; `/columns` returns `image_name,split,…`; `mergeRunColumns.ts:16` types it
categorical). The dashboard drops it:
- Parallel coordinates: the overlay effect (`ParallelCoords.tsx:161-175`) only auto-adds
  numeric headline columns; `DEFAULT_CATEGORICAL_COLS` (`:51`) lacks `split`; the Training
  group heading only covers numeric fields (`:515-516`), so `split` is buried in the generic
  "Categorical" list (`:707`).
- Colour-by: `split` is in the Training group (`App.tsx:523-526`) but never chosen.
- 3D partition mode only uses `data.clusterCol` (`Graph3D.tsx:827`, `useCSV.ts:89-93`).
- Run comparison drops it on purpose (`experiments.py:580`, `include_split=False`) — fine.

## Proposed solution

- `ParallelCoords.tsx`: when `split` is in `data.runColumns`, add it to the selected
  categorical axes (first position) on the first overlay load; list training categorical
  columns under the **Training** heading in the field picker. Reuse the `lib/runAxes.ts`
  helper from task-21 (`defaultTrainingAxes` returns categorical additions too).
- Colour-by: on first overlay load, if colour-by is still the default, switch to `split`
  with a fixed two-colour categorical palette (`train`, `val`, plus `test` if present),
  consistent across Graph3D, ParallelCoords lines and histograms. The user's later choice wins.
- Histograms: option "stack by split" for numeric columns when `split` exists, so train vs
  val photometric distributions can be compared.
- Graph3D: allow partition mode by any low-cardinality categorical column (≤ 12 values),
  with `split` offered first when present; `clusterCol` stays the default otherwise.
- Data table: `split` column pinned next to the image column when present; filter chip.
- Summary chip in the toolbar: `train 361 · val 90` for the overlaid run.

## Testing

- vitest: `defaultTrainingAxes` adds `split` to categorical axes once; split palette helper
  maps `train`/`val`/`test` to stable colours; partition-column eligibility (≤ 12 distinct).
- `npm run build && npm test`.
- Browser check on AgriStress-500 with a split-only or scored overlay: `split` is an axis
  in Parallel Coordinates, points coloured train/val in 3D, histogram stacking works,
  toolbar chip counts match `split.csv`. Console clean; screenshots reviewed.
