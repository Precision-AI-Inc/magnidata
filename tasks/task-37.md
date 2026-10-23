# task-37 — Native resolution with mixed image sizes

**Depends on:** task-36
**Commit message:** `Train native-resolution runs at the most common image size`

## Description

task-35/36 required every image to share one size for `input_resolution: native`. SEEDPLANT7_19_v2 mixes 798×532
(~79%) and 810×540 tiles, so the full-size AgriTune run failed in staging. The AgriTune runs being reproduced handled
this with `geometric.resize: [798, 532]`: cached features come from the unresized images, and only the masks are
resized to one size.

- `data.native_size` returns the size most images share (ties: the larger) and how many differ.
- AgriTune adapter: native staging leaves images untouched and resizes masks of other sizes to that size.
- Reference trainer: caches and trains at that size, resizing the others; both log the size and the count.
- Guide, dialog label and CHANGELOG describe "the size most images share".

## Testing

- `native_size` (uniform, majority, tie), native AgriTune staging and native reference training with a mixed-size
  sample; repo gates. Real runs: the SEEDPLANT AgriTune full-size and LR-ASPP full-size runs.
