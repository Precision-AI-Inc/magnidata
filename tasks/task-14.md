# task-14 — Dashboard: Train dialog and live run status

**Depends on:** task-08, task-09, task-11
**Commit message:** `Add Train dialog to launch runs from the dashboard`

## Description

The "train from the UI" entry point: launch a run on the currently loaded dataset
(including child datasets — that is what makes the loop work), follow it live while it
trains, and cancel it.

## Proposed solution

- `components/experiments/TrainDialog.tsx` (pattern: `ByodDialog.tsx`): name; dataset =
  currently loaded dataset (read-only); preset select; the preset's tunable params with
  bounds from `GET /api/training/presets` (number inputs with min/max/step); loss-term
  checkboxes + weights; split mode. Client validation mirrors the bounds; server errors
  shown inline per field. Submit → `POST /api/training/runs` → close, switch to the
  Experiments tab with the new run selected.
- Entry points: a "Train" button in the Visual Explorer toolbar and in the Experiments tab
  (disabled with the reason as tooltip when no dataset is loaded, MLflow is unreachable,
  the worker is unavailable, or another UI run is active).
- `hooks/useActiveRun.ts` (pattern: `useBuildJob.ts`): tracks the active UI-origin run at
  App level; polls `/api/runs/<id>` every 5 s while `SCHEDULED`/`RUNNING`; Header badge
  "Training · epoch 3/20" like the build badge; toast on finish/fail/kill.
- Run detail (task-11) shows **Cancel** for scheduled/running UI-origin runs
  (`POST /api/runs/<id>/cancel`, confirm first).
- **Live axes**: when the overlaid run (task-13) is the active run and the overlay is on
  "latest", refresh its columns when `scored_epochs` grows.
- Banner in the Experiments tab when the worker is unavailable, with the
  hint to check `docker compose ps` / `docker compose logs trainer`.
- Pure helpers in `lib/train.ts`: `validateParams(params, bounds)`, `isActive(status)`,
  `trainDisabledReason(state)`.

## Testing

- vitest `lib/train.test.ts`: bound validation (min/max, multiple-of-32 image size,
  score_every ≤ epochs, ≥ 1 loss term); `isActive`; disabled-reason precedence.
- `npm run build` and `npm test` pass.
- Browser check (`chrome-devtools-axi`), stack started with
  `docker compose up --build`: load AgriStress-500, Train → `tiny_unet`,
  2 epochs, image_size 128 → Header badge shows scheduled → running → finished; the run
  appears in MLflow (:5052) with params, metrics, and per-sample artifacts; overlay
  refreshes after each scored epoch; a second run can be cancelled and ends `KILLED`.
  Stop the trainer → Train disabled with the worker reason. Screenshots reviewed.
