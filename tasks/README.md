# Training feedback loop — task list

Goal: train segmentation models from the MagniData UI, store every experiment in
**MLflow**, browse those experiments (runs, params, metric curves, comparisons) from the
MagniData UI, and turn **per-sample** loss terms (`loss_total`, `loss_dice`, `loss_cce`,
`loss_bce`, …) into live dataset axes so persistently hard samples can be found,
inspected, and turned into the next dataset.

Background: `MagniData_Training_Loop_Idea (1).md` at the repo root.

## Architecture in one picture

```
Dashboard ──► MagniData API (Flask, no torch) ──────────► MLflow tracking server
                 │  browse experiments/runs/metrics          ▲   (system of record:
                 │  create SCHEDULED run / set cancel tag    │    params, metrics, tags,
                 │  download per-sample artifacts → axes     │    artifacts, run status)
                 ▼                                           │
           Trainer worker (torch, GPU optional) ── polls SCHEDULED runs, trains, logs ──┘
```

- **MLflow is the only state store for experiments.** Run status, params, metrics,
  per-sample loss files, predictions, and checkpoints all live in MLflow. MagniData keeps
  no run table of its own.
- The **job queue is MLflow itself**: the API creates a run with status `SCHEDULED`; the
  worker picks up `SCHEDULED` runs tagged `magnidata.origin=ui`. Cancellation is a tag.
- Any run in the local MLflow is browsable — including pai-seed runs logged through
  `PAIExperiment` (MLflow by default) when pointed at it with
  `MLFLOW_TRACKING_URI=http://localhost:5052`. Runs that also log per-sample artifacts
  following the contract (task-00) additionally get dataset axes.

## How to run these tasks (instructions for the implementing agent)

1. Implement the tasks **in numeric order**, `task-00.md` → `task-23.md`. Each task lists
   its dependencies; never start a task before its dependencies are committed.
2. **One task = one commit** on the current local branch. Commit message: the task's
   `Commit message` line (imperative mood, no "task-NN:" prefix).
3. **Do not push**, open PRs, create tags, or touch GitHub in any way.
4. **Never stage** `tasks/` or files unrelated to the task (e.g. the pre-existing
   uncommitted change to `.github/workflows/release.yml`). Stage files explicitly by
   path — no `git add -A` / `git add .`.
5. **No attribution trailers** — no `Co-Authored-By`, no "Generated with Claude Code".
6. Before each commit, the task's **Testing** section must pass, plus the repo gates:
   - `pre-commit run --files <changed files>` (ruff, ruff-format, pyright, yaml/json checks)
   - `python -m pytest -q` (coverage gate is 60% — do not lower it)
   - for dashboard changes: `cd precisionai/magnidata/dashboard && npm run build && npm test`
     (`npm test` exists from task-10 on)
7. Follow `CLAUDE.md` exactly: SPDX header on every `.py` file, NumPy docstrings with
   imperative summaries, full type hints, imports at module top (optional deps via
   module-level `try/except ImportError` + flag), `raise … from err`,
   `pytest.raises(…, match=…)`, `@pytest.fixture` without parens, no `# noqa`.
8. Tests are **colocated** (`test_<module>.py` next to the module), matching the existing
   repo convention (a tracked known gap in CONTRIBUTING.md).
9. **No mocks for MLflow.** Tests use a real MLflow tracking store in `tmp_path`
   (`sqlite:///<tmp>/mlflow.db` with artifact root `<tmp>/artifacts`) via the shared
   fixture added in task-01. Network-dependent tests are `@pytest.mark.integration`.
10. Add a line under `## [Unreleased]` in `CHANGELOG.md` in every task that changes
    user-visible behaviour.
11. If a task turns out to be wrong or blocked, stop and report — do not silently
    redesign the contract from task-00; later tasks depend on it.

## Decisions baked into these tasks

| Question | Assumed answer |
|---|---|
| Experiment store | **Local MLflow 3.11.1** (same version pai-seed uses), run as the `mlflow` service in this repo's docker-compose, no auth. No remote/org MLflow server, no tokens. |
| MLflow client in the API | `mlflow-skinny==3.11.1` (no server deps, no torch) wrapped in one adapter module. Artifacts go through the local server's artifact proxy (`--serve-artifacts`), so no container needs direct access to the artifact volume except `mlflow`. |
| Which trainer? | A small **reference trainer** inside this repo (`precisionai/magnidata/training/`). pai-seed integrates later by logging the per-sample artifacts with the torch-free recorder (task-02); no pai-seed commits here. |
| Task type | **Semantic segmentation only** for v1 (Dice / CCE / BCE). |
| Where training runs | A `trainer` docker-compose service, started by plain `docker compose up` like every other service. CPU-safe base compose file; GPU hosts add `docker-compose.gpu.yml` (cu128 torch + nvidia device) via `COMPOSE_FILE` in the git-ignored `.env` (task-18). The API never imports torch and never runs commands from the UI. Nomad is out of scope. |
| Access | Everything runs and is reachable through `docker compose up --build`: portal :5175, API :5051, MLflow UI :5052, trainer (no port). Nothing outside the compose stack is required. |
| Metric naming | Matches pai-seed's `Trainer`: `train_loss/loss`, `train_loss/<term>`, `val_loss/loss`, `val_loss/<term>`, `Learning_rate/lr`, plus `val_metric/miou`, `val_metric/iou_<class>`. |
| Per-sample format | CSV artifacts (no pyarrow dependency). Parquet is a future optimisation. |

## Task status

The **Status** and **Commit** columns of the task index below are the single source of
truth for progress. Do not track status anywhere else (not in the task files).

| Status | Meaning |
|---|---|
| `open` | not started |
| `in-progress` | being implemented now — at most one task at a time |
| `blocked` | cannot proceed; add a one-line reason under **Blocked notes** below |
| `completed` | committed; the **Commit** column holds the short SHA |

Rules for the implementing agent:
1. Before starting a task, set it to `in-progress`.
2. After its commit succeeds, set it to `completed` and fill **Commit** with
   `git rev-parse --short HEAD`.
3. If it fails or is blocked, set `blocked`, add a note, and stop — do not skip ahead.
4. On resume, continue from the first task that is not `completed` (an `in-progress` task
   means the previous session stopped mid-task: check `git status`/`git log` before redoing it).
5. `tasks/` is git-excluded, so these edits never end up in a commit.

### Blocked notes

## Bug-fix round (tasks 18–23)

Reported after tasks 00–17 were done, diagnosed against the running stack on 2026-10-06:

| Symptom | Root cause | Task |
|---|---|---|
| Training takes forever | Trainer container has CPU-only torch (`2.14.0+cpu`) and no GPU device although the host has an RTX 5070; full 12 MP PNG decode every pass, `num_workers=0`, no AMP; no progress output | 18, 19 |
| Losses not shown as axes | No per-sample data until the first full scoring pass (15–20 min on CPU); the run is hidden until then and never auto-overlaid | 20, 21 |
| Run dropdown missing the run | `useRunOverlay.ts` filters `has_per_sample`; list never refreshes; 5 s `/api/runs` cache not cleared on launch | 20, 21 |
| Train/val partition not displayed | `split` served correctly but not auto-added to parallel coords / colour-by; 3D partition only uses k-means `clusterCol`; `/columns` ignores `split.csv` before first scored epoch | 20, 22 |

## Task index

| Task | Title | Depends on | Status | Commit |
|---|---|---|---|---|
| task-00 | Training contract: MLflow tags, metric keys, artifact layout, per-sample columns | — | completed | 0ee5f49 |
| task-01 | MLflow service, configuration, and API adapter | 00 | completed | c26682a |
| task-02 | Torch-free per-sample loss recorder with MLflow sink | 00, 01 | completed | 4251dd4 |
| task-03 | Per-sample segmentation loss terms | 00 | completed | e4f6784 |
| task-04 | Segmentation dataset, mask decoding, and splits | 00 | completed | b3b4da1 |
| task-05 | Reference trainer logging to MLflow | 02, 03, 04 | completed | a6e4aa3 |
| task-06 | API: browse experiments, runs, params, metric history, artifacts | 01 | completed | c9c9fe3 |
| task-07 | API: derived per-sample axes from MLflow artifacts | 06 | completed | 5a4f45f |
| task-08 | API: training presets, launch (SCHEDULED run) and cancel | 06 | completed | 7ba9cfb |
| task-09 | Trainer worker service and Docker wiring | 05, 08 | completed | b760c44 |
| task-10 | Dashboard: Experiments tab (experiments + runs table) | 06 | completed | 45dc8c2 |
| task-11 | Dashboard: run detail (metric charts, params, artifacts) | 10 | completed | 48e7bc7 |
| task-12 | Dashboard: compare runs (overlaid curves, params diff) | 11 | completed | bd1c78a |
| task-13 | Dashboard: run overlay — per-sample loss columns as axes | 07, 10 | completed | 1bd0c34 |
| task-14 | Dashboard: Train dialog and live run status | 08, 09, 11 | completed | 2e02aa5 |
| task-15 | Epoch scrubber, per-sample loss history, prediction overlay | 13, 14 | completed | 6260f81 |
| task-16 | Close the loop: train on child dataset, relabel export, per-sample run compare | 12, 15 | completed | 9565b8e |
| task-17 | Documentation and end-to-end smoke test | all | completed | b712ba1 |
| task-18 | GPU trainer image, device visibility, and progress logging | 09 | completed | b529d4e |
| task-19 | Fast data pipeline: resize-once cache, parallel loading, mixed precision | 18 | completed | f7ae0df |
| task-20 | Per-sample data from the start of a run: early scoring and split before scoring | 19 | completed | 3962027 |
| task-21 | Run picker lists live runs and loss axes appear automatically | 20 | completed | c8c929b |
| task-22 | Show the train/val partition in the dashboard | 21 | completed | 63083a9 |
| task-23 | End-to-end verification on AgriStress-500 and docs update | 18–22 | completed | 565154a |
| task-24 | Calm Material-style dashboard shell | 23 | completed | f16e83b |
| task-25 | Restore the last dataset and keep large visualizations responsive | 24 | completed | a94a238 |
| task-26 | Cloud dataset graph and complement curation | 25 | completed | f9bde14 |
| task-27 | Interactive dataset lineage viewport | 26 | completed | 9145be5 |
| task-28 | Unified workspace: lineage graph with dataset detail panel | 27 | completed | |
| task-29 | Create datasets from explorer selections and by merging datasets | 28 | completed | |
| task-30 | Experiment badge on lineage dataset cards | 29 | completed | |
| task-31 | Audit: predictions must never show ground-truth masks | 30 | completed | |
| task-32 | Self-contained portal guide: AgriStress-500 to a curated, retrained dataset | 31 | completed | |
| task-33 | AgriTune training engine integration and responsive portal controls | 32 | completed | |
| task-34 | AgriTune: hosted encoder by default and preloaded cached features | 33 | completed | 5e28d3b |
| task-35 | AgriTune: full-resolution runs, existing feature stores, report training settings | 34 | completed | 358392b |
| task-36 | Reference trainer: native-resolution training | 35 | completed | e6639e0 |
| task-37 | Native resolution with mixed image sizes | 36 | completed | 3aa2978 |
| task-38 | Run detail and compare plot every metric | 37 | completed | 4a12983 |
| task-39 | Advanced AgriTune training settings in the Train dialog | 38 | completed | 11468ca |
| task-40 | Every AgriTune decoder as a preset | 39 | completed | 416cd33 |
| task-41 | Trainer shared memory for full-size training | 40 | completed | 694fa3a |
| task-42 | Demo datasets: build again after a stale record, and train on COCO128 | 41 | completed | 68a0209 |
| task-43 | Hugging Face Space with both demo datasets, sized for CPU Basic | 42 | completed | |
