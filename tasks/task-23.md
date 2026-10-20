# task-23 — End-to-end verification on AgriStress-500 and docs update

**Depends on:** task-18 … task-22
**Commit message:** `Document GPU training and verify the hard-sample workflow end to end`

## Description

Prove the user's workflow works on the real dataset with the GPU stack: launch a run from
the dashboard, see it immediately in the Run picker, get split then loss axes automatically,
and find samples whose loss stays high together with their photometric characteristics.

## Proposed solution

- README / HelpPage / Sphinx docs: GPU setup (`docker-compose.gpu.yml`, `.env`
  `COMPOSE_FILE`, cu128 for RTX 50xx, how to verify `cuda.is_available()`), new presets
  (`num_workers`, `amp`), the training cache location and size bound, the Run picker states,
  auto-overlay, "High-loss samples", and split views.
- Extend `api/test_e2e_smoke.py`: assert `split` columns are served before the first scored
  epoch and `has_per_sample` flips after epoch 1 even with `score_every > 1`.
- CHANGELOG `[Unreleased]` entries consolidated for tasks 18–22.

## Testing

- Full gates: `pre-commit run --all-files`, `python -m pytest -q`, dashboard
  `npm run build && npm test`.
- Manual end-to-end with `docker compose up --build` (GPU override active):
  1. Train AgriStress-500 @256, 10 epochs, from the Train dialog; record time per epoch,
     `perf/samples_per_sec`, and `nvidia-smi` utilisation.
  2. Run picker shows it within 10 s; split axis/colours appear once `split.csv` is logged;
     loss axes auto-appear after epoch 1.
  3. "High-loss samples" brush → screenshot of Parallel Coordinates with photometric axes +
     `loss_total_mean` + `hardness`, and the 3D view coloured by split.
  4. Report the numbers and screenshots in the final summary to the user.
