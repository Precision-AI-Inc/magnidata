# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Build a dataset from a server-side folder in `image_sets/`: drop a folder holding
  `images/` (and optionally matching COCO `labels/`) on the host, and prepare it from
  the portal's "Bring Your Own Data" dialog. The folders are indexed by
  `GET /api/datasets/build/local-folders`, and the build runs in place — no upload, no
  copy of the image data, and none of the browser-upload size caps.
- Initial release: the `precisionai.dataviz` visual-query dashboard — a deterministic
  feature-extraction/embeddings toolkit (`tools/`), a Flask API (`api/`), a React/Vite/
  Three.js dashboard (`dashboard/`), and Docker Compose deployment.

### Fixed

- Portal served a blank page after a rebuild: `index.html` is now sent with
  `Cache-Control: no-cache` and missing `/assets/` bundles return 404 instead of
  falling back to `index.html`, which browsers refused to execute as a module script.

[Unreleased]: https://github.com/Precision-AI-Inc/dataviz/compare/v0.1.0...HEAD
