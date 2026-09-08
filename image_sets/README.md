# image_sets/

Drop a folder of images here to build a dataset from it in the portal, without uploading
anything through the browser. This directory is mounted read-only into the API container at
`/app/image_sets`, and it is git-ignored apart from this file.

## Layout

One directory per dataset, named however you want the dataset named:

```text
image_sets/
  MY-SET/
    images/            # the images (JPG, PNG, BMP, WEBP, TIFF)
    labels/            # optional — one COCO-format JSON per image
```

Labels are matched to images by filename stem, so `images/img001.png` pairs with
`labels/img001.json`. Images without a matching label still build — they get the
quality/exposure columns but not the instance/coverage/class ones.

Subfolders under `images/` become the `cluster`/`cluster_l2` grouping used by the 3D and
cluster views (`images/siteA/batch1/img.png` → cluster `siteA/batch1`). A flat `images/`
folder builds as a single `uncategorized` cluster.

## Building it

In the portal, open **BYOD → Prepare From Server Folder**. Every directory here that holds a
non-empty `images/` is listed with its image and label counts; pick one and press Prepare.

The build reads the images where they are — nothing is copied and nothing is written back
into this directory — so there is no size limit on this route. Only the derived feature CSV
and its manifest are written, to `precisionai/dataviz/data_user/`.

For the fuller pipeline (DINOv2 embeddings, provenance sidecars, checksums, a permanent
`confi.yaml` entry), see
[`../precisionai/dataviz/docs/recipes/from-images-to-dashboard.md`](../precisionai/dataviz/docs/recipes/from-images-to-dashboard.md).
