# task-10 — Dashboard: Experiments tab (experiments + runs table)

**Depends on:** task-06
**Commit message:** `Add Experiments tab to browse MLflow runs`

## Description

The first view of MLflow inside MagniData: pick experiments, see their runs in a sortable
table with the metrics and params that matter, filter by status/dataset/text, and jump to
a run. It must work without any dataset loaded (experiments are global) and degrade
gracefully when MLflow is unreachable.

## Proposed solution

Testing infra first (used by all later dashboard tasks): add **vitest** as a dev
dependency, `"test": "vitest run"` in `package.json`, config in `vite.config.ts` (node
environment; no DOM library needed for pure functions).

Frontend (`precisionai/magnidata/dashboard/src/`):
- `types.ts`: add `'experiments'` to `TabId`; `App.tsx`: add the tab (lucide
  `FlaskConical`). Unlike other tabs it renders even when no dataset is loaded (adjust the
  `data && tab === …` gating and the landing-page condition accordingly).
- `api.ts`: types `Experiment`, `RunSummary`, `RunStatus`, `MlflowStatus`; functions
  `getMlflowStatus`, `listExperiments`, `searchRuns(params)`.
- `components/experiments/ExperimentsTab.tsx`:
  - left: experiment list with search and multi-select (default: experiments with
    prefix `magnidata/` plus the most recently updated ones).
  - toolbar: status filter chips (Scheduled/Running/Finished/Failed/Killed), "Only runs
    for the current dataset" toggle (when a dataset is loaded), free-text MLflow filter
    input (server validates), refresh.
  - runs table: name, status badge, dataset, preset/origin, start, duration, user, and
    **configurable metric/param columns** (column picker listing keys seen in the result;
    default: `val_loss/loss`, `val_metric/miou`, `train_loss/loss`, `epochs`, `lr`).
    Sorting by any column maps to MLflow `order_by`; "Load more" uses `page_token`.
    Reuse styling from `DataTable.tsx`.
  - row click selects the run (detail in task-11); each row also has an
    "Open in MLflow" link (`mlflow_ui_url`).
  - banner when `getMlflowStatus().reachable` is false, with the configured URI.
- `hooks/useExperiments.ts`: experiment/run fetching, selection, column-picker state
  persisted in `localStorage` (wrapped in try/catch).
- Pure helpers in `lib/experiments.ts`: `metricColumns(runs)`, `formatDuration`,
  `statusTone(status)`, `buildFilter(ui state) -> string`.

## Testing

- vitest `lib/experiments.test.ts`: column discovery across runs with different keys;
  duration formatting (s/min/h, running runs); `buildFilter` combinations (status +
  dataset + free text) produce the expected MLflow filter string.
- `npm run build` and `npm test` pass.
- Browser check with `chrome-devtools-axi` (stack with the `mlflow` service): seed runs
  (e.g. run `python -m precisionai.magnidata.training.train --dataset … --preset tiny_unet`
  twice, or log a few runs with a short script), open Experiments: experiments listed,
  runs table sorted by `val_loss/loss`, column picker adds `val_metric/iou_1`, status
  filter works, "Open in MLflow" opens the right run on :5052. Stop the mlflow container
  → banner appears, no uncaught errors in the console. Screenshots reviewed before committing.
