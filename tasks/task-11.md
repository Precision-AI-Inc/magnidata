# task-11 — Dashboard: run detail (metric charts, params, artifacts)

**Depends on:** task-10
**Commit message:** `Add run detail view with metric charts, params, and artifacts`

## Description

Selecting a run must show what people open MLflow for: every metric as a curve, the
params and tags, and the artifacts — for any run in MLflow, not only MagniData-launched
ones.

## Proposed solution

`components/experiments/RunDetail.tsx` (panel/drawer next to the runs table):
- **Header**: name, status badge, experiment, dataset (link "Load dataset" when
  `magnidata.dataset_source` is set and the dataset exists), duration, user, run id
  (copy button), "Open in MLflow", `magnidata.error` shown prominently for failed runs.
- **Metrics**: metric keys grouped by prefix (`train_loss/`, `val_loss/`, `val_metric/`,
  `Learning_rate/`, other). Default layout: one small-multiple per loss term with **train
  and val overlaid** (pairing `train_loss/<t>` ↔ `val_loss/<t>`), plus mIoU and LR charts.
  Controls: x-axis step / relative time, smoothing (EMA slider 0–0.99), log-scale y,
  show/hide groups. Fetch via `GET /api/runs/<id>/metrics` (keys requested lazily per
  visible group). Follow the `dataviz` skill for chart styling, reuse `vizColors.ts` and
  the charting approach already used in `Histograms.tsx` (no new chart library unless one
  is already a dependency).
- **Params & tags**: searchable two-column table; `magnidata.*` tags shown in their own section.
- **Artifacts**: lazy tree via `/api/runs/<id>/artifacts?path=`; images preview inline,
  other files download via `/api/runs/<id>/artifact?path=`.
- **Live**: if status is `SCHEDULED`/`RUNNING`, poll metrics and status every 5 s (stop on
  terminal status or when the panel closes).
- Pure helpers in `lib/metrics.ts`: `groupMetricKeys`, `pairTrainVal`, `ema(series, alpha)`,
  `toRelativeTime`.

## Testing

- vitest `lib/metrics.test.ts`: grouping by prefix; train/val pairing incl. unpaired keys;
  EMA (alpha 0 = identity, known 3-point result); relative-time conversion.
- `npm run build` and `npm test` pass.
- Browser check (`chrome-devtools-axi`): open a finished `tiny_unet` run → loss
  small-multiples show train and val lines, mIoU and LR charts render, smoothing and log
  scale change the plot, params table searchable, artifact tree lists
  `magnidata/per_sample/…` and previews a prediction PNG. Open a pai-seed-style run
  without `magnidata.*` tags (log one with a script) → metrics/params still render, no
  per-sample section errors. Screenshots reviewed before committing.
