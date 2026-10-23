# task-19 — Fast data pipeline: resize-once cache, parallel loading, mixed precision

**Depends on:** task-18
**Commit message:** `Speed up training with cached resized images, loader workers, and AMP`

## Description

Even on GPU the trainer would be input-bound. AgriStress-500 is 451 PNGs at 4000×3000
(~12 GB). `data.py:~260` decodes the full 12 MP image and mask and resizes to
`image_size` for **every sample on every pass**: train + val + a full scoring pass every
epoch (`score_every=1`), plus ~450 decodes to build the palette (`data.py:~125`). Loading is
single-process (`num_workers=0`, `train.py:75`), not exposed in presets, no `pin_memory`.
No AMP. Measured: trainer at ~116% CPU on a 20-core host with 19 GB read from disk.

## Proposed solution

`precisionai/magnidata/training/data.py`:
- **Resized cache**: decode each image/mask once, resize to `image_size` (image: bilinear,
  mask: nearest on decoded class indices), store as uint8 `.npy` under
  `DATA_ROOT/data_user/.train_cache/<image_size>/<sha1(abs path + mtime + size)>.npy`
  (data_user is the only rw volume). Datasets read from the cache; a miss populates it.
  Use `Image.draft()`/`reduce()` for JPEG/PNG where it helps. Bound the cache size
  (`TRAIN_CACHE_MAX_MB`, default 4096, LRU eviction by atime) — mirror the approach of
  `api/artifact_cache.py`.
- **Palette**: build from the cached masks (or from the first pass) instead of a separate
  full-resolution decode of every mask; cache the palette per dataset source + mtime.
- Pre-warm: optional `warm_cache(paths, workers)` using a `concurrent.futures` process pool,
  called by `prepare_data` with a progress line every 50 images.

`train.py`:
- `num_workers` default `min(8, (os.cpu_count() or 2) - 1)`; `pin_memory=device.type == "cuda"`,
  `persistent_workers=num_workers > 0`. Same for eval and scoring loaders.
- AMP on CUDA only: `torch.autocast("cuda", dtype=torch.bfloat16 if supported else float16)`
  in train/eval/score, `torch.amp.GradScaler` for fp16. Per-sample losses must still be
  computed and stored in float32 (cast logits before the loss terms in `losses.py`) so CSV
  values are comparable with earlier CPU runs. New param `amp` (bool, default true).
- `torch.backends.cudnn.benchmark = True` on CUDA.

`api/training_presets.py`: add `num_workers` (0..16) and `amp` to `BOUNDS`/defaults and to the
Train dialog's advanced section (task-14 UI).

## Testing

- `training/test_data.py`: cache hit returns identical arrays to an uncached decode;
  changing the source mtime invalidates; eviction above the size bound; mask resize keeps
  only valid class indices (no interpolated colours).
- `training/test_train.py`: 1-epoch tiny run with `num_workers=2` on the synthetic fixture
  finishes and produces the same per-sample CSV columns; AMP path skipped without CUDA.
- `api/test_training_launch.py`: `num_workers` and `amp` validated against bounds.
- Manual on AgriStress-500 @256 with the GPU stack from task-18: report epoch-1 time vs
  epoch-2 time (cache warm) and steady-state `perf/samples_per_sec`. Target: epoch 2+ is
  at least 10× faster than the measured CPU baseline (~10 min train+val for epoch 1).
