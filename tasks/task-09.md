# task-09 — Trainer worker service and Docker wiring

**Depends on:** task-05, task-08
**Commit message:** `Add trainer worker service that runs scheduled MLflow runs`

## Description

Connect the MLflow "queue" (task-08) to the trainer (task-05): a long-running worker
process, in its own container with torch (and a GPU when available), that claims
`SCHEDULED` UI-origin runs one at a time, runs them, and survives restarts without
leaving runs stuck in `RUNNING`.

## Proposed solution

`precisionai/magnidata/training/worker.py`,
`python -m precisionai.magnidata.training.worker --data-root /app`:

- Loop every 5 s: touch `DATA_ROOT/data_user/.trainer_heartbeat`; `search_runs` across
  experiments with prefix `magnidata/` for `attributes.status = 'SCHEDULED' and
  tags."magnidata.origin" = 'ui'`, oldest first.
- **Claim**: MLflow has no compare-and-set, so v1 assumes **one worker** (documented).
  Still guard against double-starts: set tag `magnidata.worker=<host:pid>`, re-read the run,
  proceed only if the tag is ours and the status is still `SCHEDULED`; then the trainer
  sets `RUNNING`.
- Run the trainer as a **subprocess**
  (`python -m precisionai.magnidata.training.train --run-id … --data-root …`), output
  to a local log file that is also logged as artifact `logs/trainer.log` at the end (and
  every few minutes while running). A subprocess isolates CUDA/OOM crashes.
- If the subprocess exits without the run reaching a terminal status → `FAILED`, tag
  `magnidata.error` = exit code + last log line.
- **Startup recovery**: `RUNNING` runs whose `magnidata.worker` names this host with a
  dead pid → `FAILED`, `magnidata.error="worker restarted during run"`.
- SIGTERM: forward to the child, wait up to 30 s, then `KILLED`.
- Small, testable functions: `find_scheduled`, `claim`, `recover_stale`, `run_one`,
  `loop(max_iterations=None)`.

Docker:
- `Dockerfile.trainer` (repo root): `python:3.12-slim`; CPU torch by default with build arg
  `TORCH_INDEX_URL` to switch to a CUDA wheel index; install the `training` extra
  (torch, torchvision, mlflow-skinny); copy `precisionai/`; `CMD` the worker.
- `docker-compose.yml`: `trainer` service started by plain `docker compose up` (no
  profile — everything must be reachable from one command), `restart: unless-stopped`,
  mounts the same `data/`, `image_sets/` (ro) and
  `data_user/` (rw) as the API, env `MLFLOW_TRACKING_URI=http://mlflow:5000`,
  `depends_on: mlflow: condition: service_healthy`, commented GPU
  `deploy.resources.reservations.devices` block with a note that CPU is the default.

## Testing

`precisionai/magnidata/training/test_worker.py` (real sqlite MLflow fixture):
- `find_scheduled` returns only `SCHEDULED` + `origin=ui` runs, oldest first.
- `claim` refuses a run already tagged by another worker id or no longer `SCHEDULED`.
- `recover_stale` fails a `RUNNING` run with a dead pid on this host; leaves a live-pid run
  (`os.getpid()`) alone; ignores runs from other hosts.
- `run_one` with a fake trainer argv (`[sys.executable, "-c", "import sys; sys.exit(3)"]`)
  → `FAILED` with the exit code in `magnidata.error`.
- `run_one` with the real trainer on the synthetic fixture (`tiny_unet`, 1 epoch) → `FINISHED`,
  `logs/trainer.log` artifact present.
- `loop(max_iterations=1)` touches the heartbeat.
- Manual: `docker compose up --build` starts all four services; `docker compose logs trainer`
  shows the worker polling; the heartbeat file appears under `data_user/`.
