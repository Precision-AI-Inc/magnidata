# task-42 — Demo datasets: build again after a stale record, and train on COCO128

**Depends on:** task-41
**Commit message:** `Fix demo dataset builds and make COCO128 trainable`

## Description

Two problems showed up trying the demos on a fresh checkout.

1. **"The AgriStress-500 demo dataset has already been built"** while the dialog offered *Prepare*. Dataset records
   live in the `magnidata-api-db` volume, which survives a new clone, while `data_user/` is a git-ignored bind mount
   that does not. `/api/datasets` skips a record whose CSV is gone, but `start_demo_build` only looked at the record
   and refused with a 400. A demo now counts as built only if its CSV exists; otherwise the stale record is deleted
   and the build proceeds.
2. **"dataset has no segmentation masks to train on"** when training on the COCO128 demo. The build wrote COCO polygon
   labels only; the trainer reads colour masks (`mask_path` column or `masks/<stem>.png` beside `images/`). Staging
   now also writes one mask per image from those polygons. The trainer accepts at most 32 classes (background
   included) and COCO128 uses about 70, so the 15 classes with the most polygon area keep their own colour and every
   other class shares grey. Larger polygons are drawn first so small objects stay visible. The annotation overlay
   still prefers the polygons, so it does not change.

A COCO128 built earlier has no masks: delete it and prepare it again.

## Testing

- `test_dataset_builder.py`: a built demo is still refused (both demos, with its CSV on disk); a record without its
  CSV is deleted and the build starts (both demos).
- `test_prepare_coco128_dashboard.py`: staging writes a mask per image that `has_masks` finds; classes beyond the
  limit share one colour; a small object is drawn over a large one; an image without objects gets an empty mask.
- Staging the real COCO128-seg download writes 128 masks, `has_masks` is true and the masks use 17 colours (limit 32).
- Not done: delete COCO128 in the portal, prepare it again and start a Tiny UNet run.
