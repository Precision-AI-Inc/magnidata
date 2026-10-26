# task-40 — Every AgriTune decoder as a preset

**Depends on:** task-39
**Commit message:** `Offer every AgriTune decoder as a training preset`

## Description

AgriTune ships six segmentation decoders (`build_decoder`: `mlp_probe`, `token_fpn`, `aspp`, `ppm`, `segmenter`,
`mask_former`); MagniData offered two. Add presets for ASPP, Pyramid Pooling, Segmenter and MaskFormer, and give each
decoder's constructor options an advanced Decoder setting scoped to the decoders that have it: hidden width, layers,
dropout, CLS fusion (TokenFPN), atrous rates (ASPP), pool sizes (PPM), attention heads and MLP ratio (Segmenter,
MaskFormer), mask queries (MaskFormer). Options whose defaults differ between decoders default to unset, and an unset
option is not passed, so each decoder keeps its own default.

## Testing

- Every preset maps to its AgriTune decoder; every decoder builds through AgriTune's `build_decoder` with each of its
  settings changed and with none set; API serves and launches all six presets with their defaults.
