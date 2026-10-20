# task-18 — GPU trainer image, device visibility, and progress logging

**Depends on:** task-09
**Commit message:** `Run the trainer on the GPU and report device and progress`

## Description

Training on AgriStress-500 "takes forever" because the trainer never touches the GPU.
Measured on the dev host (WSL2): an RTX 5070 Laptop GPU (8 GB, Blackwell / sm_120, driver
CUDA 13.2) sits idle; `docker info` lists the `nvidia` runtime; but inside the `trainer`
container torch is `2.14.0+cpu` and `torch.cuda.is_available()` is `False`.

Root causes:
- `Dockerfile.trainer:6` defaults `TORCH_INDEX_URL` to the CPU wheel index.
- The GPU `deploy.resources.reservations.devices` block in `docker-compose.yml` (trainer
  service) is commented out.
- `train.py` `resolve_device("auto")` silently falls back to CPU; the device is only an
  MLflow param, never printed, never shown in the UI.
- `train.py` prints nothing during epochs/batches/scoring, so `trainer.log` stays 0 bytes
  for the first 15+ minutes and the user cannot tell "slow" from "stuck".

## Proposed solution

Docker / compose (GPU must not break CPU-only hosts):
- `Dockerfile.trainer`: keep the `TORCH_INDEX_URL` build arg; document cu128 as the
  minimum for RTX 50xx (Blackwell). Verify that `torch==2.14.0`/`torchvision==0.29.0` exist
  on the cu128 index; if not, pin the closest versions that do and note it in CHANGELOG.
- New `docker-compose.gpu.yml` override: sets `build.args.TORCH_INDEX_URL:
  https://download.pytorch.org/whl/cu128` and the nvidia `deploy` devices block for
  `trainer`, plus env `MAGNIDATA_TRAIN_DEVICE=cuda`. Leave the base `docker-compose.yml`
  CPU-safe (remove the commented block, point to the override in its comment).
- GPU hosts keep the "one command" rule by setting `COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml`
  in the (git-ignored) `.env`. Document this in README/CONTRIBUTING and add it to the local
  `.env` on this machine (do not commit `.env`).

Trainer (`precisionai/magnidata/training/train.py`, `worker.py`):
- The worker passes `--device` from `MAGNIDATA_TRAIN_DEVICE` (default `auto`).
- `resolve_device`: `cuda` requested but unavailable → raise a clear error so the run
  goes to `FAILED` with `magnidata.error="CUDA requested but not available (torch <ver>)"`.
  `auto` keeps the CPU fallback, but logs a warning line.
- At run start print (with `flush=True`) one line: `device=cuda:0 (NVIDIA GeForce RTX 5070 Laptop GPU) torch=2.14.0+cu128`,
  and set tag `magnidata.device` (add the key to `contract.py`, task-00 contract is extended,
  not changed).
- Progress lines, all `flush=True`: every N batches (N so that ~10 lines/epoch)
  `epoch 3/20 train batch 40/45 loss=0.412 2.1 it/s`; at epoch end
  `epoch 3/20 done train=… val=… 41.2s ETA 11m`; scoring pass start/end with sample count
  and seconds. Log MLflow metrics `perf/epoch_seconds`, `perf/samples_per_sec` per epoch
  and tag `magnidata.eta_seconds` (updated each epoch).
- The worker already uploads `logs/trainer.log` periodically — confirm the interval is ≤ 60 s
  while running so the run detail log is near-live.

Dashboard:
- Run detail and the Experiments runs table show `magnidata.device` (badge `GPU` / `CPU`)
  and, for RUNNING runs, `ETA` from `magnidata.eta_seconds`.

## Testing

- `training/test_train.py`: `resolve_device("cuda")` raises with `match="CUDA requested"`
  when CUDA is unavailable (skip if it is available); a 1-epoch tiny run on the synthetic
  fixture sets `magnidata.device`, logs `perf/epoch_seconds`, and its captured stdout
  (`capsys`) contains `device=` and `epoch 1/1 done`.
- `training/test_worker.py`: worker builds the trainer argv with `--device` from the env.
- vitest for the device/ETA formatting helper (pure function in `lib/experiments.ts`).
- Manual (record outputs in the commit-free task notes / final report):
  `docker compose build trainer && docker compose up -d`, then
  `docker compose exec trainer python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
  → `+cu128 True …RTX 5070…`; during a run `nvidia-smi` shows a python process;
  `trainer.log` has a `device=` line within seconds of the run starting.
