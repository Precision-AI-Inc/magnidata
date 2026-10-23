# task-34 — AgriTune: hosted encoder by default and preloaded cached features

**Depends on:** task-33
**Commit message:** `Encode AgriTune datasets once with the hosted encoder and train from preloaded features`

## Description

AgriTune runs launched from the portal fell back to the offline fake encoder (its features carry no image
information, so the decoder only learns background) and trained through the `hybrid` provider, which reads
or encodes every batch's features each epoch (~33 min per epoch on SEEDPLANT7_19_v2, 151k images).

- **Hosted encoder in the integration.** `docker-compose.agritune.yml` defaults
  `AGRITUNE_ENCODER_BASE_URL` to `https://embeddings.precision.ai/v1` (an explicitly empty value still selects
  the offline fake encoder) and reads `AGRITUNE_ENCODER_API_KEY` from the sibling `../agritune/.env`, the
  file AgriTune's own CLI uses, so the key is never copied into this repository.
- **Cached provider with preload.** A new `cached` feature provider encodes the whole staged dataset into the
  feature store once, before training (resumable: already-encoded samples are skipped, so later runs on the
  same data skip it), then trains from the store. A new `feature_preload` setting (`none`, `host`, `device`)
  loads every feature and mask into RAM or GPU memory once (float16), with per-epoch shuffling. AgriTune only
  allows preload with `cached`; the API enforces that. AgriTune presets default to `cached` + `host`.
- Train dialog: the provider gains *Encode once, then train from the store*; a *Preload* select appears for
  it. Retrain restores both settings.
- Guide (Train with AgriTune): hosted encoder setup from AgriTune's `.env`, the new settings, memory sizing.

## Testing

- API: defaults, `cached`/`feature_preload` choices, preload rejected without `cached`, reference presets reject
  `feature_preload`.
- Adapter (trainer image with AgriTune): `_config` maps `cached` + preload to AgriTune's `feature_preload`,
  `feature_preload_dtype`, `shuffle`; the encode-once step fills a real store from a synthetic dataset with the
  fake encoder and skips it on a second call.
- Dashboard: `validateParams`/`paramsFromRun` for the new fields.
- Repo gates per README.
