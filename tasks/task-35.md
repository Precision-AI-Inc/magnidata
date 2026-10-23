# task-35 — AgriTune: full-resolution runs, existing feature stores, and the report's training settings

**Depends on:** task-34
**Commit message:** `Train AgriTune at native resolution from existing feature stores with the report's settings`

## Description

The MagniData AgriTune adapter squashed every image to `image_size × image_size` before encoding (SEEDPLANT 798×532
→ 256×256, an 18×18 patch grid instead of 57×38), which capped a linear probe near 62% val mIoU against the
AgriTune report's 77.6–77.7%. It also could not express the report's optimiser settings or reuse the feature store
those runs already built.

- `input_resolution` (AgriTune presets): `image_size` keeps today's square staging; `native` encodes the original
  image files unchanged and trains on masks at their original size. All images must share one size (checked).
- Existing feature store: with `AGRITUNE_EXISTING_MANIFEST` / `AGRITUNE_EXISTING_STORE` (+ store type, shard size,
  preprocessing label) set for the trainer, a `native` + `cached` run whose every image matches a row of that
  manifest by file name uses that manifest's sample IDs and image files and reads the store in place (no encoding);
  per-sample rows keep MagniData image names through AgriTune's `MagniDataOptions.image_names`. Otherwise the run
  encodes as usual and the log says why.
- Settings: `accumulation_steps`, `grad_clip_norm` (AgriTune), `weight_decay` (both engines); batch size up to 256.

## Testing

- API bounds/validation for the new fields; reference presets reject the AgriTune-only ones.
- Adapter: native manifest points at original files with masks at original size; mixed sizes rejected; existing-store
  match uses foreign IDs + name map, a partial match falls back; trainer/optimiser settings reach AgriTune's config.
- Real run: SEEDPLANT with the `seedplant-sanity-mlp_probe_test_1` configuration reading `features-sharded-test`.
