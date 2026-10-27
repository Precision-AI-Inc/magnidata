---
license: cc-by-nc-4.0
pretty_name: AgriStress-500-Downsized
task_categories:
  - image-segmentation
tags:
  - agriculture
  - magnidata
---

# AgriStress-500-Downsized

A downsized copy of [AgriStress-500](https://huggingface.co/datasets/precisionaiinc/AgriStress-500), prepared for
the MagniData demo Space. It exists only to feed that Space. **It is not the dataset**, and it is not meant for
training or research: for that, use
[precisionaiinc/AgriStress-500](https://huggingface.co/datasets/precisionaiinc/AgriStress-500).

## What is in it

`agristress500/` is a bundle written by `precisionai/magnidata/scripts/export_demo_bundle.py` in the
[MagniData repository](https://github.com/Precision-AI-Inc/magnidata) and installed by the Space's image build:

- `bundle.json`, the manifest (name, description, row count, image sizes).
- `agristress500.csv`, the feature table, and `agristress500.json`, the embeddings. Both were computed from the
  full-resolution AgriStress-500 images.
- `agristress500_images/`: `images/` (JPEGs, longest side at most 2048 px), `thumbnails/` (at most 720 px) and
  `masks/` (the segmentation masks, resized with nearest-neighbour so the class colours stay exact).

Images are renamed from `.png` to `.jpg`, and the embeddings' keys are renamed with them.

## Provenance and license

Derived from [precisionaiinc/AgriStress-500](https://huggingface.co/datasets/precisionaiinc/AgriStress-500)
(451 images and masks), which is released under CC-BY-NC-4.0. This derivative keeps that license.
