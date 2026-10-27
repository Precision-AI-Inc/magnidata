# task-43 — Hugging Face Space with both demo datasets, sized for CPU Basic

**Depends on:** task-42
**Commit message:** `Add a Hugging Face Space build with both demo datasets`

## Description

A demo of MagniData as a Hugging Face Space in the `precisionaiinc` organisation, with COCO128 and AgriStress-500
already loaded. It is an add-on: the compose stack, its images and the app's behaviour are unchanged unless a
Space-only setting is switched on. The Space runs on the free **CPU Basic** tier (2 vCPUs, no GPU) and stays private
until it is ready to be released.

**One container.** `deploy/huggingface/Dockerfile` builds the API and the dashboard into one image that listens on
7860 as uid 1000. It clones the repository at `MAGNIDATA_REF` (the Space repository holds only the Dockerfile and its
card), builds the dashboard, and runs the app's own demo build for COCO128 while the image is built, so the first
visitor finds both datasets ready. There is no nginx, so the API serves the built dashboard itself when
`MAGNIDATA_STATIC_DIR` is set, with the same caching rules as `nginx.conf`: hashed assets cached forever, a missing
asset a 404, `index.html` always revalidated.

**AgriStress-500 from a bundle.** Its originals are 12 GB of PNGs, too much to download and extract features from
inside an image build. `scripts/export_demo_bundle.py` exports a built dataset as a bundle: images re-encoded as JPEGs
capped at `--max-size` pixels (default 2048, about 520 MB for the 451 images; the portal asks for up to 4096 px in its
image modal, so that view tops out at 2048 here), a 720 px thumbnail set for the grid, colour masks resized to match
with nearest-neighbour, and the feature table and embeddings that were computed from the full-resolution originals.
Embeddings are joined to the table by image basename, and a mismatch silently becomes zero rows, so the table's
`image_path` and `thumbnail_image` and the embeddings' keys are renamed from `.png` to `.jpg` together, and
`validate_bundle` fails if any row lacks an image, thumbnail, mask or embedding. It runs on export and again on install.
`api/preload_demo.py --bundle DIR` installs a bundle and registers it the way a demo build does (a first-class dataset
with its own embeddings) and precomputes the 3D projections, so the first view is instant.

The bundle lives in a public Hugging Face dataset repository used only by the Space, `precisionaiinc/AgriStress-500-Demo`
(public because AgriStress-500 is, which also means the build needs no token). It is deliberately not the published
`precisionaiinc/AgriStress-500`: that is a public dataset other people load as an `imagefolder`, extra image folders
would change what `load_dataset` returns, and a separate name keeps people from mistaking a downsized copy for the real
dataset. The derivative keeps CC-BY-NC-4.0 and its card says it is not the dataset. The image build downloads it with
`huggingface_hub` (a private repo would read a build secret named `HF_TOKEN`), a failed download fails the build, and
`--build-arg DEMO_DATA_REPO=` builds COCO128 only. COCO128 is not bundled: the exporter carries colour masks, while the
app's own build also keeps the polygons its overlay prefers.

**Demo mode.** `MAGNIDATA_DEMO_MODE`, off by default, is for a deployment open to anyone. The server answers with 403
`{"error": "Disabled for the Demo"}`, before it reads the request, every route that changes data shared by all visitors
or starts work, matched by method and URL rule so an id in the path does not matter: `POST /api/datasets/build` (an
upload, a server folder and the demo downloads), creating (`POST /api/datasets`), merging, re-parenting and deleting
datasets, creating, editing and deleting notes, `POST /api/partitions/commit`, `POST /api/training/runs`, and cancelling,
linking or running inference on a run; the unused `GET /api/mask/debug` is refused too. The image endpoints
(`/api/image`, `/api/mask`, `/api/overlay`, `/api/annotation/*`) open whichever file their `path`, `img` or `mask`
parameter names, with no containment, which on an open server is an arbitrary-file-read and a way to probe the
filesystem through the error messages; in demo mode a remote URL, an absolute path or a path that leaves `DATA_ROOT`
gets a 400. The 3D view starts one analysis thread per request and keeps every finished job forever, so in demo mode at
most 4 run at once (429 beyond that) and the 50 most recent finished ones are kept. The CSV, summary and embedding
routes already contain their paths. The refusal hook is only registered when the flag is on. `GET /api/config` reports
`demo_mode`, and the portal then shows Add dataset, View experiments, the Experiments section, every Train button, delete
(in the detail panel and on the graph), Merge, Save as dataset, Create complemented dataset and Create partitions
disabled, still visible, with the tooltip "Disabled for the Demo" (a React context carries the reason, so no props were
threaded and nothing changes without it). The Dockerfile sets the flag after the preload step, because that step drives
the endpoint demo mode refuses; a test pins that order.

**CPU Basic.** `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` are 2, because the container sees the
host's cores while it is given two. `MLFLOW_HTTP_REQUEST_MAX_RETRIES=0` and a 3 s timeout, because there is no MLflow
server and a request to its default address has to fail at once (24 ms, instead of a retry stall of about 2 s).
Thumbnails are their own 720 px set so the grid never decodes a 2048 px JPEG.

Left open on purpose, because they are what the demo shows: every read-only request, including the embedding
projections, clusters and analyses. They are cached, and the analysis jobs are bounded in demo mode.

Two things found while preparing the data: the builder expects the working directory to be `DATA_ROOT` (it is in
Docker), and the feature extractor spawns worker processes, so a script that drives a build needs an
`if __name__ == "__main__":` guard.

## Testing

- `test_export_demo_bundle.py` (18 tests): images capped and masks the same size, no enlarging, table and embeddings
  renamed together, thumbnail column rewritten, a dataset without masks, several worker processes, and every failure
  the validator is there to catch (missing embedding, image or file, unknown format, row count, non-empty output).
- `test_preload_demo.py` (19 tests with the installer): an installed bundle is served by the real API (dataset list,
  CSV, image, overlay, embeddings), installing twice keeps one record, an incomplete bundle installs nothing, failing
  projections do not stop an install, and the command-line exit codes.
- `test_demo_mode.py`: `/api/config` both ways. Every refused route is spelled out in the test (not imported, so dropping
  one from the server fails it) and checked twice: 403 in demo mode, and reached (not a 403) in a normal deployment. A
  note is never written and a dataset is never deleted in demo mode, while a normal deployment still adds a note, deletes
  a dataset and reads an image by absolute path. In demo mode an image inside the data folder is served, and a URL, an
  absolute path and `..` are refused on every image endpoint and on both `/api/overlay` parameters. The analysis cap
  refuses a fifth concurrent job, keeps the 50 latest finished ones, and a normal deployment is never limited.
- `test_dashboard_static.py`: dashboard routes, fallback, cache headers, a JSON 404 for unknown `/api` paths, real API
  routes still win, no routes without the variable. `test_space_dockerfile.py`: demo mode is set once, after the preload.
- Dashboard: `tsc --noEmit` is clean and `npm test` passes (150 tests, including the new Train reason for the demo).
- The Space image was built locally from the working tree with both datasets (the clone and the bundle download swapped
  for local copies) and run with `--cpus=2 --memory=16g`: ready in 6 s, image 0.74 GB, about 266 MiB of memory after
  the run. COCO128 (128 rows) and AgriStress-500 (451 rows) are listed with embeddings, 26 of 26 sampled overlays render
  for each, all 451 AgriStress thumbnails load in 5.6 s six at a time with no failures, and a 1280 px AgriStress
  preview takes 160 to 450 ms and its overlay 300 to 800 ms cold. On the real image, builds, training, notes,
  dataset deletion, merging, partition commits and the debug endpoint return 403, and an absolute or `..` image path
  returns 400.
- A browser session against the image (Playwright, Chromium): the seven disabled controls the UI shows are `disabled`
  and carry the title "Disabled for the Demo" (Add dataset, the Experiments item, View experiments, delete, Train model,
  Save selection as a dataset, Create complemented dataset). A visitor's session of 10 actions (brush a filter on an
  axis, sort, Charts, Table, Overview, Schema, back to the Explorer, the 3D embeddings, and an image preview stepped
  through) completed with no page errors and no refused request. The only request that was not a plain GET was the
  allowed `POST /api/embedding/analysis`; the two requests that answered with an error were the workspace's background
  poll for MLflow runs (`/api/experiments`, `/api/runs`, 503), which the Space has no MLflow to answer. Merge, creating
  a dataset from removals or additions and Create partitions were checked in code and through the server refusals, not
  in a browser.
- The regular `Dockerfile.api` image: `demo_mode` is false, build requests are validated as before, the training
  endpoint is reachable (no 403) and the API serves no dashboard.
- Full suite: 693 passed, 50 skipped (the torch and AgriTune tests, skipped as before), coverage 71.77% against the
  60% gate. `main` before this task: 589 passed, 69.51%. Ruff, ruff format and pyright are clean.
- On Hugging Face: the public data repo `precisionaiinc/AgriStress-500-Demo` (1358 files, `cc-by-nc-4.0`) and the
  private Space `precisionaiinc/magnidata` were created, the Space building this branch. The first build, which
  includes `RUN --mount=type=bind,from=` and the bundle download, finished in about a minute and a half and the Space
  reached RUNNING on `cpu-basic`. A smoke test over HTTPS shows both datasets (128 and 451 rows, with embeddings), every
  refusal answering 403 or 400 as in the local image, 26 of 26 sampled overlays on each dataset and no failed read.
  Before anything was published the diff, the Space files and the bundle were scanned for secrets, personal details and
  local paths, and the bundle for EXIF and GPS data: none were found.
- MLflow in demo mode: the Space has no MLflow server, so the dashboard's polling (`/api/experiments`, `/api/runs`,
  `/api/training/presets`, every 15 s) was answered with a 503 and a logged failure each time. In demo mode the server
  now installs an `OfflineStore` that answers every MLflow call at once and without logging (status `reachable: false`),
  and the dashboard, told by `/api/config`, does not send those requests at all (`setMlflowOffline` in `api.ts`).
  A normal deployment keeps the real store and the polling; tests cover both.
- Upload CSV is disabled in demo mode, with the same tooltip and cursor as the other controls. It only reads a file in
  the browser, but a visitor's upload leads to Add dataset, which is off, so leaving it enabled was a dead end.
- The portal's footer links to the GitHub repository in demo mode, and the Space card was rewritten to follow the
  repository's README (logo, GitHub and licence badges, the same description) and to list both datasets in its
  front matter, which is what shows them on the Space page.
- Access once the Space is public is documented in `MAINTAINING.md`: organization roles, a Resource Group for the Space
  and the data repo, disabling discussions, and that whoever can merge to `main` controls what the Space runs.
- Not done: switching the Space's Dockerfile from `feature/hf-space` to `main` once this merges.
