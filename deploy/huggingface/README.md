---
title: MagniData
emoji: 🔭
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: Explore image datasets by features, embeddings and clusters
---

# MagniData

Open-source data visualization for image datasets, by [Precision AI](https://precision.ai).
This Space runs the portal with two datasets already loaded, so you can open it and explore:

- **COCO128**: 128 everyday scenes with real instance segmentation polygons.
- **AgriStress-500**: 451 field images of crops and weeds with segmentation masks, from
  [Precision AI's dataset](https://huggingface.co/datasets/precisionaiinc/AgriStress-500)
  (CC-BY-NC-4.0). Images here are downsized for this demo; the feature table and embeddings come from the
  full-resolution originals.

For each dataset you get parallel coordinates, histograms and a table over per-image complexity, quality and
coverage metrics, image, mask and overlay previews, and a live 3D PCA / t-SNE / cluster view over the embeddings.

This is a demo on small CPU hardware, so it is read-only. Anything that adds, changes or deletes data (adding,
merging, saving or deleting datasets, creating partitions), and training and the Experiments tab, is disabled here: those
controls show "Disabled for the Demo". To use them, run the full stack with `docker compose up` from the
[GitHub repository](https://github.com/Precision-AI-Inc/magnidata), where nothing is turned off.

The Space repository holds only this card and a `Dockerfile` that builds MagniData from GitHub.
