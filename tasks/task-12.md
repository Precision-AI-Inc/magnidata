# task-12 — Dashboard: compare runs (overlaid curves, params diff)

**Depends on:** task-11
**Commit message:** `Add run comparison with overlaid metric curves and params diff`

## Description

Model-level comparison, as in MLflow's compare view: select 2–10 runs and see their
metric curves overlaid, a summary table of final/best metrics, and which params differ.
(Per-sample comparison — which images got easier — is task-16.)

## Proposed solution

- Runs table (task-10): checkbox column + "Compare (n)" button (enabled for 2–10 runs).
- `components/experiments/RunCompare.tsx`:
  - **Curves**: metric picker (keys present in any selected run, grouped as in task-11);
    one chart per chosen metric with one line per run, consistent run colours across all
    charts and the legend (categorical palette from `vizColors.ts` / dataviz skill),
    smoothing + log-scale controls shared with task-11. Data via `POST /api/runs/metrics`.
  - **Summary table**: rows = runs, columns = chosen metrics showing final value and best
    value (min for `*loss*`, max for `*miou*`/`*iou*`/`*acc*`, configurable per column),
    best cell per column highlighted.
  - **Params diff**: rows = param keys, columns = runs; "only differing params" toggle
    (default on); differing cells highlighted.
  - URL hash keeps the selected run ids so a comparison can be reloaded.
- Pure helpers in `lib/compare.ts`: `diffParams(runs)`, `bestValue(series, direction)`,
  `defaultDirection(key)`, `assignRunColors(runIds)`.

## Testing

- vitest `lib/compare.test.ts`: `diffParams` (missing keys count as differing, equal
  values hidden); `bestValue` min/max; `defaultDirection` for loss/iou/unknown keys;
  stable colour assignment for the same id order.
- `npm run build` and `npm test` pass.
- Browser check (`chrome-devtools-axi`): compare two `tiny_unet` runs with different `lr`
  → `lr` is the only differing param shown, `val_loss/loss` curves overlaid with legend,
  summary highlights the better run; reload keeps the selection. Screenshots reviewed.
