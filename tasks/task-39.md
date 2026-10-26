# task-39 — Advanced AgriTune training settings in the Train dialog

**Depends on:** task-38
**Commit message:** `Expose AgriTune's full training configuration as advanced Train dialog settings`

## Description

The Train dialog exposed a handful of AgriTune options. AgriTune's `TrainingRunConfig` has many more: optimizer name,
betas and momentum; the learning-rate schedule; precision, early stopping, step checkpoints, best-checkpoint metric and
class weights; decoder options per preset; geometric and photometric image augmentation; feature augmentation; data
loading and preload precision.

- `training/agritune_settings.py` (torch-free) declares each setting once: MLflow key, target path in AgriTune's
  config, type, bounds, choices, preset scope, group, label, help and default.
- API: validates advanced keys for AgriTune presets (refused for reference presets and other presets' decoder
  settings), records only those a run sets, and serves the specs with the presets.
- Trainer: applies every setting (the run's value or its default) by target path, with the few that need logic
  (constant schedule = none and total steps from the run when 0; precision auto; shuffle auto; preload precision only
  with preload; class weights per class; CLS fusion enum; image augmentation only when augmenting).
- Dashboard: collapsible *Advanced AgriTune settings*, grouped, with placeholders, help, client validation, field-level
  server errors, reset, and retrain restore from the earlier run's params.

## Testing

- Spec module: well-formed, defaults round-trip through MLflow strings, invalid values, unset lists, preset scope.
- API: served specs, validation and recording, refusals.
- Adapter (with AgriTune): every target exists in a real `TrainingRunConfig` for both presets; values reach AgriTune;
  defaults keep the schedule constant and augmentation off; class weights; CLS fusion; preload precision.
- Dashboard: parsing, errors, payload, retrain restore.
