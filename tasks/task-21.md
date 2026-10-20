# task-21 — Run picker lists live runs and loss axes appear automatically

**Depends on:** task-20
**Commit message:** `List live runs in the Run picker and auto-overlay their loss axes`

## Description

Two user-reported bugs with one cause:
1. "The Run dropdown does not show the actual run to select." The server returns it
   (verified: `GET /api/runs?filter=tags.\`magnidata.dataset_source\` = 'data_user/agristress500.csv'`
   returns the RUNNING run), but `hooks/useRunOverlay.ts:33` keeps only
   `r.has_per_sample`, and the list is fetched once per dataset change — no polling, no
   refresh after a launch, refresh only when the *already overlaid* run gains an epoch
   (`App.tsx:306-313`). Status labels in `RunOverlayPicker.tsx:30` go stale.
2. "Losses are not shown as new axes in the parallel coordinates." Merge + axis wiring works
   for finished runs (`ParallelCoords.tsx:160-175` auto-adds `loss_total` and `hardness`),
   but the user never gets there: the run is hidden, and nothing auto-selects it when its
   first scored epoch lands.

Goal of the feature: see samples whose loss **stays** high next to their photometric
characteristics, without hunting through menus.

## Proposed solution

- `useRunOverlay.ts`: keep every run linked to the dataset; expose
  `runs: {run, selectable, reason}[]` sorted newest first. Selectable = `has_per_sample || has_split`.
  Poll `refreshRuns` every 10 s while any listed run is SCHEDULED/RUNNING (stop when none
  are, and when the tab is hidden); refresh immediately after the Train dialog's
  `onLaunched` and when the picker opens.
- `RunOverlayPicker.tsx`: status badge + device badge (task-18) + `epoch N/M` + scored
  count. Unselectable runs render disabled with reason ("queued", "training · first losses
  after epoch 1 scoring", "failed — no per-sample data"). Runs with only a split are
  selectable and labelled "split only".
- Auto-overlay (`App.tsx`): when a run launched from this dataset (or the newest RUNNING
  run for it) becomes selectable and no overlay is active, select it with epoch `latest`
  and show a toast "Overlaying run <name>". When the overlaid run gets a new scored epoch
  and the user is on `latest`, re-fetch columns (existing behaviour) — keep brushes.
- Parallel coordinates default when an overlay first loads: keep the user's photometric
  axes and append `loss_total`, `loss_total_mean`, `hardness` (only if present), placed
  next to each other at the right end, coloured with the sequential loss palette. Add a
  one-click **"High-loss samples"** action in the Training group: brushes
  `hardness ≥ 0.6` (or top 20% `loss_total_mean` when hardness is absent) so the table,
  histograms and 3D view show those samples' photometric values.
- Extract the selection/ordering logic to a pure helper `lib/runAxes.ts`
  (`defaultTrainingAxes(columns, current) -> string[]`) so it is unit-testable.

## Testing

- vitest `lib/runAxes.test.ts`: appends loss axes once, no duplicates on re-merge, skips
  missing columns, preserves existing photometric axes.
- vitest for the picker model (pure function building `{selectable, reason}` from run
  summaries): RUNNING without data → disabled with reason; split-only → selectable;
  FINISHED with per-sample → selectable; ordering newest first.
- `npm run build && npm test`.
- Browser check (`chrome-devtools-axi`) on AgriStress-500: launch a short run (2 epochs,
  128 px) from the Train dialog → the run appears in the picker within 10 s as disabled
  "training"; becomes "split only" then auto-overlays after epoch 1 scoring;
  `loss_total` / `hardness` axes appear in Parallel Coordinates beside the photometric
  axes; "High-loss samples" filters the table. Console clean; screenshots reviewed.
