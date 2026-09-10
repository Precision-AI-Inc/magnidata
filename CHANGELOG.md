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
- Initial release: the `precisionai.magnidata` visual-query dashboard — a deterministic
  feature-extraction/embeddings toolkit (`tools/`), a Flask API (`api/`), a React/Vite/
  Three.js dashboard (`dashboard/`), and Docker Compose deployment.
- `docker compose up` now prints where to open the tool (`portal -> http://localhost:5175`,
  `api -> http://localhost:5051/api/health`) in the API's startup log, driven by the
  `MAGNIDATA_PORTAL_URL` / `MAGNIDATA_API_URL` variables in `docker-compose.yml`.

### Changed

- Renamed the project to MagniData throughout: the distribution is now `magnidata`
  (was `pai-dataviz`), the import path is `precisionai.magnidata` (was
  `precisionai.dataviz`), and the Compose project/containers are `magnidata-*`. The
  `dataviz-features/…` and `dataviz-embeddings/…` feature-schema identifiers are
  deliberately unchanged — they are a data contract recorded in provenance sidecars, so
  renaming them would misreport the schema version of already-generated datasets.
- Human-facing text now spells the product "MagniData" everywhere (READMEs, docs, CLI
  help, browser tab title "Precision AI - MagniData"); package and import paths stay
  lowercase `magnidata`. The README and `pyproject.toml` descriptions now match, and the
  root README no longer shows PyPI/Python-version badges.
- Landing page cards are wider, and the light- and dark-theme MagniData artwork now share
  one crop and scale, so both themes show the same logo at the same size.
- AgriStress-500 demo progress reads "Downloading AgriStress-500…" next to the overall
  percentage, instead of a per-image counter beside a different percentage.
- Inline errors in the Bring Your Own Data dialog are larger, bold, and boxed with an icon.

### Fixed

- Only one dataset build can run at a time, but the UI didn't say so: the Bring Your Own
  Data dialog now names the dataset being prepared and disables Prepare/Create until it
  finishes, a rejected request (HTTP 409) names the running build, and a rejected start no
  longer briefly replaces the running build's progress badge with an error.
- `SECURITY.md` listed the wrong report address and a template package name; it now points
  to reinier@precision.ai and `magnidata`.
- Portal served a blank page after a rebuild: `index.html` is now sent with
  `Cache-Control: no-cache` and missing `/assets/` bundles return 404 instead of
  falling back to `index.html`, which browsers refused to execute as a module script.

[Unreleased]: https://github.com/Precision-AI-Inc/magnidata/compare/0.0.3...HEAD
