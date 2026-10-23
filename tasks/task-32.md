# task-32 — Self-contained portal guide: from AgriStress-500 to a trained, curated dataset

**Status:** completed

**Depends on:** task-31
**Commit message:** `Add a self-contained guide to the portal`

## Description

Replace the current Help & Setup page with a guide inside the portal that explains how every piece fits
together and walks a new user from `docker compose up` to a curated, retrained dataset using the
AgriStress-500 dataset. Everything except dataset partitioning is done in the UI; partitioning is done
with a command-line tool whose output appears as lineage in the dataset graph. The guide must be
self-contained: a reader with only the repository and the portal can complete every step without
leaving it, and every screenshot comes from the real UI.

Audience: a new user (data scientist or agronomist) who has not seen MagniData; no prior knowledge of
MLflow, Docker internals, or the codebase.

## Proposed solution

### Guide structure (portal, **Help & Setup**)

A chaptered guide with a left-hand table of contents, anchors for deep links, previous/next navigation,
and "copy" buttons on every command. Content lives in the dashboard bundle (works offline, versioned with
the code). Chapters:

1. **How the pieces fit** — one diagram (inline SVG) of `web`, `api`, `mlflow`, `trainer`, the data volume
   and the browser, plus the data flow: images → features/embeddings → dataset CSV → lineage graph →
   Visual Explorer → training run → per-image losses → curated dataset → retrain. A glossary (dataset,
   lineage, run, experiment, hardness, ground truth vs prediction, complement, partition).
2. **Install and start** — prerequisites, `docker compose up --build`, optional GPU compose file, the four
   ports, how to tell the stack is healthy (summary strip, Trainer card), where data lives, how to stop and
   reset. Includes the "502 while the API is still starting" note.
3. **Get AgriStress-500** — the one-click demo (BYOD → Load Demo) and what it downloads, how long it takes,
   how to confirm it appears in the workspace; Git LFS pointer problems and fix.
4. **Partition it from the command line** — the new CLI (below): why partition, copy-paste commands for the
   Docker stack and for a local checkout, expected output, and how the resulting sub-datasets appear in the
   lineage graph.
5. **Tour the workspace** — summary strip, lineage graph (select, Shift-click multi-select, double-click to
   open), detail panel tabs, experiment badges (yellow = live, gray = none), merge, delete.
6. **Explore in the Visual Explorer** — parallel coordinates and brushing, 3D embeddings, table, image
   preview (image / mask / overlay / prediction, with the ground-truth vs prediction caption), removing
   images, saving a selection as a dataset, exporting CSV.
7. **Train and read the results** — Train dialog and presets, live run status, experiments table, run
   detail, compare, overlaying a run (loss columns, epoch scrubber, `split`), reading predictions.
8. **Close the loop** — find persistently hard images, curate or complement from Cloud, save as a child
   dataset, train on it, compare on the dataset, merge datasets; each step ends with what to look for.
9. **Reference and troubleshooting** — CSV/embeddings data contract, CLI reference, ports and environment
   variables, common errors (trainer offline, MLflow unreachable, 502, LFS pointer, empty graph) with fixes.

Each chapter opens with a goal and an expected result, uses numbered steps, and ends with "what you should
see" so a reader can check themselves. No step may say "see the README" for something needed to proceed.

### New command-line tool: dataset partitioning

`python -m precisionai.magnidata.scripts.partition_dataset` (run in the `api` container with
`docker compose exec api …`, or locally):

- Input: a registered dataset (default `data_user/agristress500.csv`).
- Tiles images and masks to a fixed size (default 256×256, mask tiles use nearest-neighbour), recomputes
  the per-tile feature columns, and writes tiles under `data_user/<name>/{images,masks}`.
- Splits the tiles into N balanced sub-datasets (default 2; `--parts`, `--seed`, optional `--stratify-by`
  column) so that every part is a valid dataset on its own.
- Registers each part through the API (`POST /api/datasets`) as a child of the source dataset, and
  optionally registers the union as a Cloud-style parent, so the lineage graph shows source → parts.
- Idempotent and resumable: re-running skips existing parts; `--dry-run` prints the plan and counts;
  prints a summary table and the URL of the workspace.
- Documented in chapter 4 with the exact commands used to create the screenshots.

### Screenshots and samples

- Captured from the running UI against AgriStress-500 by a repeatable script
  (`scripts/capture_docs_screenshots.py` driving headless Chrome), 1440×900, light theme, stored under
  `dashboard/public/guide/`; each stays under the 1,800 KB pre-commit limit (optimised PNG or WebP).
- Minimum set: workspace with the lineage graph (source → parts → merged, with badges), detail panel
  (statistics, images, annotations), multi-select merge panel, Visual Explorer overview, 3D embeddings
  with a brushed selection, image preview in each of image / mask / overlay / prediction (the same sample,
  showing ground truth next to the prediction), Train dialog, a live run, run detail curves, compare view,
  loss-column overlay with the epoch scrubber, saving a selection as a dataset, and the Experiments view
  filtered to one dataset.
- "Cool samples": pick real, visually clear AgriStress images (a healthy field, a stressed region, a hard
  sample the model gets wrong) and reuse the same few images across chapters so the story is continuous.
- Alt text for every image; callouts (numbered badges) added in CSS, not baked into the PNG, so they stay
  readable and translatable.

### Keeping the guide correct

- Every command block is checked in a test against the tool's `--help` output (flags exist) and every
  screenshot reference is checked for an existing file.
- The capture script is the single source for screenshots, so they can be refreshed after UI changes.
- Existing recipes (`docs/recipes/*.md`, `docs/*-contract.md`) stay as detailed references; the guide links
  to them but never depends on them for a required step. The root README quickstart points to the guide.

## Testing

- Python tests for the partition CLI: tiling sizes, balanced split counts, deterministic seed, lineage
  registration through the API (child parents), idempotent re-run, `--dry-run`.
- Frontend tests: chapter list and anchor generation, copy-button text, referenced images exist, every
  documented flag appears in `--help`.
- Acceptance run (recorded in the task): on a fresh checkout with the compose volumes removed, follow the
  guide verbatim from chapter 2 through chapter 8 — start the stack, load AgriStress-500, run the CLI,
  explore, train, curate, retrain — without consulting any other document, fixing the guide wherever a step
  was unclear or wrong.
- Dashboard build, test suite and repository gates.

## Open questions to settle while implementing

- Whether partitioning should recompute all feature columns per tile (slow, needs the feature extractor in
  the `api` image) or copy parent features and recompute only a small set; the guide must state which.
- Whether the guide also ships as Markdown in `docs/` for the Sphinx build, or only in the portal.
- AgriStress-500 licensing/attribution text to show on the Get AgriStress-500 chapter.

## Decisions on the open questions

- Partitioning recomputes the per-tile features with the portal's existing server-folder build (fast pass, no NIMA),
  rather than copying the parent's whole-image features, which would be wrong for a crop. About a minute per 1,000 tiles.
- The guide ships in the portal only (bundled in the dashboard image); the README points to it.
- Attribution: the guide names the Hugging Face dataset and tells readers to check its dataset card for terms.

## Implementation notes

- Guide: `dashboard/src/components/guide/` (`chapters.tsx`, `blocks.tsx`, `commands.ts`, `shots.json`, `GuidePage`),
  deep links `#guide=<chapter>`, replacing `HelpPage`.
- CLI: `precisionai/magnidata/scripts/partition_dataset.py`; lineage through the new `POST /api/datasets/<id>/parents`.
- Screenshots: `dashboard/scripts/capture-guide.mjs` (playwright-core) with stages `source,byod,workspace,explorer,train,
  runs,overlay,selection,loop,filtered`; needs a stack with the partitioned data and a trainer worker.
- Tests keep the guide honest: every screenshot referenced/declared/present and < 1,800 KB, every command's module exists,
  every documented CLI flag exists in the parser and every parser flag is documented.
- Found while capturing: the ground-truth preview caption was unreadable and card names truncated; both fixed.

## Acceptance run

On an isolated second Compose project (own volumes and ports, API and web built from this tree, existing trainer image):
`partition_dataset --limit-images 8 --parts 2` from the repository root produced the output shown in chapter 4 (1,771 tiles,
909 + 862), both parts registered under AgriStress-500; the trainer container then trained a 2-epoch run on part 1 to FINISHED;
the portal served the guide images. Deviation: the 24-image AgriStress sample was seeded through `confi.yaml` instead of the
12 GB Load Demo download, so chapter 3's download itself was not re-run; its UI (Load Demo → Prepare) is captured from a fresh
install.

## Follow-up: own data and embeddings (added after review)

- Two chapters added (9 Bring your own data, 10 Add embeddings; Reference is now 11), with five more screenshots.
- Verified in a real stack while writing them, which exposed two product gaps, both fixed with tests:
  uploaded embeddings were stored under the original file names while images are renamed on staging (0 of 40 matched);
  and created datasets could not gain embeddings after creation (now a `<stem>.json` beside the CSV is discovered).
- Also verified: embeddings tool on 40 tiles (DINOv2, CPU, ~13 s), the Embeddings source and PCA projection, the
  server-folder build, and that deleting a dataset lets its folder be prepared again.

## Follow-up: rename and extension chapter (added after review)

- The navigation entry and page title **Help & Setup** are now **User Guide** (book icon); all screenshots were regenerated
  from a clean stack so none show the old label.
- New chapter 11, **Add a model or a metric** (Reference is now 12). Verified in a throwaway copy with a `tiny_unet_wide` preset,
  a `jaccard` loss term and a `val_metric/pixel_accuracy` metric: the trainer ran, the Train dialog listed the preset and
  term, `loss_jaccard` (+ mean/delta/slope/rank) appeared as Explorer axes, the metric appeared as a chart and a Columns
  option, on-demand inference worked on the new model, and the trainer's 137 tests still passed. A run logged with plain
  MLflow (custom `custom/seconds_per_image` metric) appeared in Experiments.
- Limit documented: per-image values in the Explorer come only from `loss_<term>`, `iou` and `confidence` columns.
