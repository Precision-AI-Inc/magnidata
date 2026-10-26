# task-17 — Documentation and end-to-end smoke test

**Depends on:** task-00 … task-16
**Commit message:** `Document the training feedback loop and add an end-to-end smoke test`

## Description

Make the feature usable by someone who wasn't here, document how existing trainers
(pai-seed via `PAIExperiment`) show up and how they add per-sample axes, and lock the
whole path in with one automated smoke test: launch → worker trains → MLflow → derived
axes → API columns.

## Proposed solution

- `precisionai/magnidata/docs/recipes/training-feedback-loop.md`: end-to-end walkthrough —
  `docker compose up --build` (portal :5175, API :5051, MLflow UI :5052), build the
  AgriStress-500 demo, Train, watch metrics in the Experiments tab, compare runs, overlay
  per-sample losses, scrub epochs, find persistent hard samples, save as dataset, retrain,
  compare per sample. Include a table of every derived column and how to read combinations
  (e.g. high `loss_dice_rank` + low `loss_cce_rank` → mask shape problem, not class confusion).
- `precisionai/magnidata/docs/training-contract.md`: add **"Logging host-side runs to the local MLflow"**
  (`MLFLOW_TRACKING_URI=http://localhost:5052` for pai-seed / `PAIExperiment` runs on the
  host; where the MLflow data volume lives; how to back it up or reset it) and **"Emitting
  per-sample losses from your own trainer"** (pai-seed-style snippet: per-term
  `reduction="none"`, eval-mode scoring pass, `LossRecorder` + `MlflowSink`, or the
  `training.attach` CLI for past runs; then "Link to dataset" in the UI).
- `HelpPage.tsx`: short "Experiments & training" section pointing to the recipe.
- `README.md` (repo root) feature list and services/ports table (mlflow :5052, trainer
  service); `precisionai/magnidata/README.md` likewise; `CONTRIBUTING.md` known gaps if any
  were introduced.
- `CHANGELOG.md` `[Unreleased]`: consolidate the entries from tasks 00–16 into one
  coherent "Added" block.
- Sphinx `docs/modules.rst`: autodoc entries for the new public modules
  (`training.contract`, `training.recorder`, `training.losses`, `training.data`,
  `training.train`, `training.worker`, `api.mlflow_store`, `api.run_axes`) if modules are
  listed explicitly.

Smoke test `precisionai/magnidata/training/test_e2e_smoke.py`:
- real sqlite MLflow fixture + synthetic dataset fixture registered as a dataset;
  `POST /api/training/runs` via the Flask test client (`tiny_unet`, 2 epochs, image_size 32);
  run `worker.loop(max_iterations=…)` in-process until the run is terminal; then assert
  `GET /api/runs/<id>` (FINISHED, 2 scored epochs), `GET /api/runs/<id>/metrics`
  (`train_loss/loss` has 2 points), `GET /api/runs/<id>/columns` (every fixture image,
  `hardness` in [0, 1]).
- `@pytest.mark.integration` only if > 30 s on CPU.

## Testing

- `pytest -q` (and `pytest -m integration` if the smoke test is marked) pass.
- `pre-commit run --all-files` passes.
- `cd docs && make html` builds without new warnings for the added modules.
- `npm run build` and `npm test` pass.
- Follow the recipe literally in a fresh browser session (`chrome-devtools-axi`) and fix
  any step that doesn't match the UI.
