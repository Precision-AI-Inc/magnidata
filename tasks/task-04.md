# task-04 — Segmentation dataset, mask decoding, and splits

**Depends on:** task-00
**Commit message:** `Add segmentation training dataset with deterministic splits`

## Description

The trainer must read any MagniData dataset that has masks (e.g. the AgriStress-500 demo:
`data_user/agristress500.csv`, images in `agristress500_images/images/`, colour masks in
`agristress500_images/masks/`) and yield `(image, target, image_name)` so every loss can be
keyed back to a dataset row. Masks are **colour PNGs** (the API's `_mask_as_rgba` treats
near-black as background), so a colour → class-index mapping is required. The train/val
split must be deterministic and logged, because "hard on train" and "hard on val" mean
different things.

## Proposed solution

`precisionai/magnidata/training/data.py`:

- `resolve_samples(dataset_csv, data_root) -> tuple[list[Sample], ResolveReport]` — read
  the CSV, find the image column with `contract.image_path_column`, locate the sibling
  mask (`…/images/<stem>.png` → `…/masks/<stem>.png`, or a `mask_path`-like column if
  present). Samples without a mask are skipped and counted in the report, not raised.
- `has_masks(dataset_csv, data_root) -> bool` — **torch-free** cheap check (first 20 rows)
  that the API uses in task-08 to reject datasets without masks. Keep it in a torch-free
  module (`training/dataset_check.py`) so the API can import it.
- `build_palette(mask_paths, max_classes=32) -> Palette` — unique RGB colours; near-black
  (all channels ≤ 10, same threshold as the API) → class 0; others → 1..N by frequency;
  more than `max_classes` → `ValueError`. Serialisable to `magnidata/palette.json`.
- `decode_mask(rgb, palette) -> np.ndarray[int64]` — unmapped colours → `ignore_index`.
- `make_split(image_names, val_fraction, seed, column_values=None) -> dict[str, str]` —
  hash-based (`sha1(f"{seed}:{name}")`) so existing assignments are stable when rows are
  added; `split_mode="column"` uses a `split` column from the CSV instead.
- `SegmentationDataset(torch.utils.data.Dataset)` — resize to `image_size` (bilinear
  image, nearest mask), ImageNet normalisation, horizontal flip only for the train split
  when `augment=True`; `__getitem__` → `dict(pixel_values, labels, image_name)`; a
  `collate_fn` that keeps `image_name` as a list.

## Testing

`precisionai/magnidata/training/conftest.py`: a `synthetic_dataset` fixture writing 8
small PNG images + colour masks + a CSV in the repo's layout under `tmp_path` (reused by
tasks 05, 09, 17).

`precisionai/magnidata/training/test_data.py`:
- `resolve_samples` finds all pairs; a sample with a missing mask is reported.
- `has_masks` true for the fixture, false for a CSV without masks; no torch imported
  (subprocess check on `dataset_check`).
- `build_palette`: black → 0, two colours → 1, 2; too many colours → `ValueError` with `match=`.
- `decode_mask` round-trips; unknown colour → `ignore_index`.
- `make_split` deterministic, honours `val_fraction` within ±1, stable when names are added;
  column mode uses the CSV column.
- dataset item shapes/dtypes (`[3,H,W]` float32, `[H,W]` int64); `image_name` is the basename;
  `collate_fn` keeps names.
