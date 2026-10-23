# task-36 — Reference trainer: native-resolution training

**Depends on:** task-35
**Commit message:** `Train the reference models at the images' own size`

## Description

The reference trainer (Tiny UNet, LR-ASPP) always resized to a square `image_size`, so it could not train SEEDPLANT
at its full 798×532 for comparison with AgriTune's full-size runs.

- `input_resolution` becomes a setting of both engines. `native` makes the trainer read every image's header, require
  one common `(width, height)`, and cache and train at that size; `image_size` keeps the square behaviour.
- The checkpoint records `input_size`; on-demand inference resizes to it (older checkpoints fall back to the square
  `image_size`).
- API: `input_resolution` accepted for reference presets (default `image_size`); Train dialog shows *Input resolution*
  for every preset.

## Testing

- `common_image_size` (uniform, mixed, parallel), a native end-to-end training run recording `input_size`, rejected
  unknown values, inference from a non-square checkpoint, API launch of a native reference run, dashboard launch
  params. Real run: LR-ASPP on SEEDPLANT at 798×532.
