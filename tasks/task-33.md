# task-33 — AgriTune training engine integration and responsive portal controls

**Status:** completed

**Depends on:** task-32
**Commit message (AgriTune):** `Support MagniData-managed training runs`
**Commit message (MagniData):** `Control AgriTune training from the portal`

## Description

Make AgriTune a first-class training engine behind MagniData's existing MLflow queue. A user selects
an AgriTune decoder in the same Train dialog, controls its relevant training/feature/augmentation
settings, watches live progress and loss curves, and receives the same split, per-image loss,
prediction, checkpoint and provenance experience as the reference MagniData trainer.

The implementation spans two sibling repositories. AgriTune is developed on
`feature/magnidata-integration`; MagniData remains on the current feature branch. Each repository gets
one focused commit. The existing untracked AgriTune `REPORT.md` and `mlflow.db` are not touched.

## AgriTune branch

- Add a MagniData integration module which attaches to an existing MLflow run instead of creating a
  second run, translates aggregate metrics to the MagniData contract, and owns terminal status.
- Add an epoch observer to the trainer/service layer so the adapter can report epoch progress, honour
  cancellation, score every sample in eval mode, and publish artifacts without coupling AgriTune's
  core trainer to MagniData.
- Produce `magnidata/split.csv` and `magnidata/per_sample/epoch_NNNN.csv` with `image_name`, `split`,
  `loss_total`, configured component loss columns, per-image IoU and confidence.
- Save top-K colourized prediction masks and upload the last/best decoder checkpoints and run
  provenance. Set schema/scored-epoch/device/progress tags expected by MagniData.
- Support MagniData dataset CSVs by generating an AgriTune manifest whose stable `sample_id` and image
  names continue to join against Explorer rows.
- Test the MLflow attach behavior, metric/tag/artifact contract, component losses, cancellation, and a
  tiny end-to-end fake-encoder run against a real local MLflow store.

## MagniData

- Add an explicit engine to preset definitions. Keep the existing Tiny UNet and LR-ASPP presets on the
  reference engine; add AgriTune MLP Probe and TokenFPN presets with decoder, feature-provider,
  augmentation and loss choices.
- Validate/serialize engine-specific fields server-side. The API never accepts a command or arbitrary
  path from the browser.
- Dispatch AgriTune runs from the existing worker based on the preset's engine while preserving the
  one-worker queue, heartbeat, cancellation, logs and recovery semantics.
- Install the sibling AgriTune checkout in the trainer image through a named Compose build context, so
  both local branches are tested together without vendoring one product into the other.
- Make the Train dialog conditional and responsive: show only applicable settings, explain frozen
  encoder features, preserve settings for retraining, and work at phone through desktop widths.
- Existing Experiments/run-detail/compare/overlay views consume the common contract and therefore show
  AgriTune losses, metrics, predictions and per-image axes without an engine-specific display path.
- Add a User Guide chapter with clone/build/configuration steps, the end-to-end workflow, field meanings,
  expected UI results, offline fake-encoder behavior, hosted encoder configuration and troubleshooting.
- Update README and CHANGELOG.

## Verification

- AgriTune: focused tests plus full `pytest`, ruff/format, pyright and pre-commit.
- MagniData: API/worker tests, dashboard tests/build, full Python suite and pre-commit.
- Build the combined trainer image from the two sibling checkouts.
- Launch a tiny AgriTune run from the portal/API on a fixture dataset with the fake encoder; verify it
  reaches FINISHED and the run detail contains loss curves, metrics and artifacts; verify Explorer
  overlay exposes per-image losses and prediction access remains separate from ground truth.

## Assessment and verification (MagniData side)

Reviewed the uncommitted work from the parallel session, completed it, and verified on a real combined trainer image
built from `../agritune` (isolated Compose project, own ports and volumes) with the offline fake encoder:

- Lifecycle through the existing worker: SCHEDULED -> RUNNING with `AgriTune epoch n/m` and scoring stages -> FINISHED;
  both presets (MLP Probe, TokenFPN), `bce_dice` objective with weights, online augmentation, online features.
- Contract: `magnidata.engine`, split and per-sample files (`loss_total`, component losses, `iou`, `confidence`),
  top-K predictions served by the normal prediction endpoint, `last.ckpt` and provenance artifacts, derived Explorer axes.
- Reference engine still dispatches to the reference trainer and finishes; cancelling an AgriTune run ends KILLED.
- Phone-width Train dialog is a single-column bottom sheet; AgriTune controls replace the loss-term checkboxes.

Changes made while assessing:

- AgriTune is now opt-in. The named build context made `docker compose build` fail for everyone without a sibling
  checkout ("failed to get build context agritune"). `Dockerfile.trainer` now has an empty default `agritune` stage and
  installs from it only when present (a bind mount, so nothing is baked into a layer); `docker-compose.agritune.yml`
  supplies the context and encoder variables. Verified: default build succeeds, override build has AgriTune.
- API validation by engine: AgriTune presets reject `loss_terms`, `num_workers`, `split_mode: column`, `pretrained: true`;
  reference presets reject the AgriTune fields. Weights for BCE objectives were rejected (validated against the reference
  default terms) and are now accepted; the adapter scaled the BCE term with `weight_cce` and now uses `weight_bce`.
- On-demand inference for AgriTune runs returns an explicit 409 (previously a misleading "no checkpoint").
- Guide chapter reworked (opt-in setup, settings, real screenshots); contract doc gains `magnidata.engine`.

Open items for the AgriTune repository (not changed here):

- `train_loss/loss` is logged per optimizer step (14 points for 2 epochs) while every other loss is per epoch, so the
  run-detail "loss (total)" chart mixes time axes; the contract says step is the epoch. The run-detail screenshot is
  therefore left out of the guide until fixed.
- Per-class IoU is published as `val_metric/iou_0..N` rather than `iou_<class name>`.
