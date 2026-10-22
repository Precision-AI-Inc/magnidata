# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for sample discovery, palette, splits and the segmentation dataset."""

import csv
import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from precisionai.magnidata.training.dataset_check import find_mask, has_masks, resolve_under

torch = pytest.importorskip("torch")
data = pytest.importorskip("precisionai.magnidata.training.data")


def _mask_files(ds):
    return [ds.root / "data_user" / "syn" / "masks" / n for n in ds.names]


def test_resolve_samples_finds_all_pairs(synthetic_dataset):
    samples, report = data.resolve_samples(synthetic_dataset.csv, synthetic_dataset.root)
    assert [s.image_name for s in samples] == synthetic_dataset.names
    assert report.total_rows == report.resolved == 8
    assert report.missing_mask == []
    assert all(s.mask_path.is_file() and s.image_path.is_file() for s in samples)


def test_resolve_samples_reports_missing_mask_image_and_duplicates(synthetic_dataset):
    ds = synthetic_dataset
    _mask_files(ds)[3].unlink()
    (ds.root / "data_user" / "syn" / "images" / ds.names[5]).unlink()
    with open(ds.csv, "a", newline="") as fh:
        csv.writer(fh).writerow([f"data_user/syn/images/{ds.names[0]}", 32, 24])
    samples, report = data.resolve_samples(ds.csv, ds.root)
    assert len(samples) == 6
    assert report.missing_mask == [ds.names[3]]
    assert report.missing_image == [ds.names[5]]
    assert report.duplicate_names == [ds.names[0]]
    assert report.total_rows == 9


def test_resolve_samples_uses_mask_path_column(synthetic_dataset):
    ds = synthetic_dataset
    other = ds.root / "elsewhere.png"
    _mask_files(ds)[0].rename(other)
    sibling_missing, _ = data.resolve_samples(ds.csv, ds.root)
    assert ds.names[0] not in [s.image_name for s in sibling_missing]

    with open(ds.csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows[0]["mask_path"] = "elsewhere.png"
    with open(ds.csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[*rows[0], "mask_path"], restval="")
        writer.writeheader()
        writer.writerows(rows)
    samples, _ = data.resolve_samples(ds.csv, ds.root)
    assert samples[0].mask_path == other.resolve()


def test_paths_cannot_escape_data_root(synthetic_dataset):
    assert resolve_under(synthetic_dataset.root, "../outside.png") is None
    assert resolve_under(synthetic_dataset.root, "/etc/passwd") is None
    assert resolve_under(synthetic_dataset.root, "https://host/a.png") is None
    assert find_mask("../x/images/a.png", synthetic_dataset.root) is None


def test_has_masks(synthetic_dataset, tmp_path):
    assert has_masks(synthetic_dataset.csv, synthetic_dataset.root)
    plain = tmp_path / "plain.csv"
    plain.write_text("image_path,width\ndata_user/nomask/images/a.png,10\n")
    assert not has_masks(plain, tmp_path)
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    assert not has_masks(empty, tmp_path)


def test_dataset_check_does_not_import_torch():
    code = "import sys; import precisionai.magnidata.training.dataset_check; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_build_palette_orders_by_frequency(synthetic_dataset):
    palette = data.build_palette(_mask_files(synthetic_dataset))
    # the left third (class A) is 32//3 * 24 px, the bottom-right quarter (class B) is 16 * 12 px
    assert palette.colors == (synthetic_dataset.colors[0], synthetic_dataset.colors[1])
    assert palette.num_classes == 3
    assert palette.as_mapping()[0] == {"name": "background", "rgb": [0, 0, 0]}
    assert palette.as_mapping()[2]["rgb"] == list(synthetic_dataset.colors[1])


def test_build_palette_in_parallel_matches_serial_and_reports_progress(synthetic_dataset):
    masks = _mask_files(synthetic_dataset)
    lines: list[str] = []

    palette = data.build_palette(masks, workers=2, progress=lines.append)

    assert palette == data.build_palette(masks)
    assert lines[-1] == f"scanning mask colours {len(masks)}/{len(masks)}"


def test_progress_every_caps_the_number_of_lines():
    assert data.progress_every(10) == data.WARM_PROGRESS_EVERY
    assert data.progress_every(151_291) == 1512


def test_build_palette_treats_near_black_as_background(tmp_path):
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    arr[0] = (9, 10, 3)
    arr[1] = (11, 0, 0)
    Image.fromarray(arr).save(tmp_path / "m.png")
    palette = data.build_palette([tmp_path / "m.png"])
    assert palette.colors == ((11, 0, 0),)


def test_build_palette_too_many_colours(tmp_path):
    arr = np.zeros((1, 40, 3), dtype=np.uint8)
    arr[0, :, 0] = np.arange(40) + 50
    Image.fromarray(arr).save(tmp_path / "m.png")
    with pytest.raises(ValueError, match="too many mask colours"):
        data.build_palette([tmp_path / "m.png"], max_classes=32)


def test_decode_mask_round_trip_and_unknown_colour(synthetic_dataset):
    palette = data.build_palette(_mask_files(synthetic_dataset))
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[0, 0] = synthetic_dataset.colors[0]
    rgb[0, 1] = synthetic_dataset.colors[1]
    rgb[1, 0] = (1, 2, 3)
    rgb[1, 1] = (99, 98, 97)
    out = data.decode_mask(rgb, palette)
    assert out.dtype == np.int64
    # (1, 2, 3) is near-black so it is background; (99, 98, 97) is unknown
    assert out.tolist() == [[1, 2, 0], [0, data.IGNORE_INDEX, 0]]


def test_decode_mask_without_palette_colours():
    out = data.decode_mask(np.full((2, 2, 3), 200, dtype=np.uint8), data.Palette(()))
    assert (out == data.IGNORE_INDEX).all()


def test_make_split_is_deterministic_and_respects_fraction():
    names = [f"img_{i}.png" for i in range(2000)]
    first = data.make_split(names, 0.2, seed=1)
    assert first == data.make_split(names, 0.2, seed=1)
    assert first != data.make_split(names, 0.2, seed=2)
    assert abs(sum(v == "val" for v in first.values()) - 400) < 80


def test_make_split_is_stable_when_names_are_added():
    names = [f"img_{i}.png" for i in range(500)]
    before = data.make_split(names, 0.25, seed=7)
    after = data.make_split([*names, *(f"new_{i}.png" for i in range(100))], 0.25, seed=7)
    assert all(after[n] == before[n] for n in names)


def test_make_split_small_dataset_has_both_splits(synthetic_dataset):
    split = data.make_split(synthetic_dataset.names, 0.25, seed=0)
    assert set(split.values()) == {"train", "val"}
    assert set(data.make_split(["a.png", "b.png"], 0.5, seed=0).values()) == {"train", "val"}


def test_make_split_column_mode():
    column = {"a.png": "val", "b.png": "test", "c.png": "train"}
    assert data.make_split(["a.png", "b.png", "c.png"], 0.5, seed=0, column_values=column) == column
    with pytest.raises(ValueError, match="invalid split"):
        data.make_split(["a.png"], 0.5, seed=0, column_values={"a.png": "holdout"})


def test_dataset_items(synthetic_dataset):
    ds = synthetic_dataset
    samples, _ = data.resolve_samples(ds.csv, ds.root)
    palette = data.build_palette([s.mask_path for s in samples])
    splits = data.make_split([s.image_name for s in samples], 0.25, seed=0)
    train = data.SegmentationDataset(samples, palette, splits, "train", image_size=(16, 20), augment=True)
    val = data.SegmentationDataset(samples, palette, splits, "val", image_size=(16, 20))
    assert len(train) + len(val) == 8
    assert {s.image_name for s in train.samples}.isdisjoint(s.image_name for s in val.samples)

    item = val[0]
    assert item["pixel_values"].shape == (3, 16, 20)
    assert item["pixel_values"].dtype == torch.float32
    assert item["labels"].shape == (16, 20)
    assert item["labels"].dtype == torch.int64
    assert item["image_name"] == val.samples[0].image_name
    assert set(item["labels"].unique().tolist()) <= {0, 1, 2}


def test_only_train_split_is_augmented(synthetic_dataset, monkeypatch):
    ds = synthetic_dataset
    samples, _ = data.resolve_samples(ds.csv, ds.root)
    palette = data.build_palette([s.mask_path for s in samples])
    splits = dict.fromkeys(ds.names, "train")
    monkeypatch.setattr(data.random, "random", lambda: 0.0)
    plain = data.SegmentationDataset(samples, palette, splits, "train", (16, 16), augment=False)[0]
    flipped = data.SegmentationDataset(samples, palette, splits, "train", (16, 16), augment=True)[0]
    assert torch.equal(flipped["labels"], plain["labels"].flip(-1))
    val_item = data.SegmentationDataset(samples, palette, dict.fromkeys(ds.names, "val"), "val", 16, augment=True)[0]
    assert torch.equal(val_item["labels"], plain["labels"])


def test_collate_keeps_names(synthetic_dataset):
    ds = synthetic_dataset
    samples, _ = data.resolve_samples(ds.csv, ds.root)
    palette = data.build_palette([s.mask_path for s in samples])
    dataset = data.SegmentationDataset(samples, palette, {}, None, image_size=16)
    batch = data.collate_fn([dataset[0], dataset[1]])
    assert batch["pixel_values"].shape == (2, 3, 16, 16)
    assert batch["labels"].shape == (2, 16, 16)
    assert batch["image_name"] == ds.names[:2]


def _samples(ds):
    return data.resolve_samples(ds.csv, ds.root)[0]


def test_cache_hit_matches_uncached_decode(synthetic_dataset, tmp_path):
    samples = _samples(synthetic_dataset)
    cache = data.ResizedCache(tmp_path / "cache", (16, 16))
    for sample in samples[:3]:
        miss = cache.get(sample)
        hit = cache.get(sample)
        fresh_image, fresh_mask, _ = data._decode_pair(sample, (16, 16))
        assert np.array_equal(miss[0], fresh_image)
        assert np.array_equal(hit[1], fresh_mask)
    assert len(list((tmp_path / "cache" / "16x16").glob("*.counts.npy"))) == 3


def test_cached_dataset_equals_uncached(synthetic_dataset, tmp_path):
    samples = _samples(synthetic_dataset)
    palette = data.build_palette([s.mask_path for s in samples])
    splits = {s.image_name: "train" for s in samples}
    plain = data.SegmentationDataset(samples, palette, splits, None, 16)
    cached = data.SegmentationDataset(samples, palette, splits, None, 16, cache=data.ResizedCache(tmp_path, (16, 16)))
    for i in range(len(plain)):
        assert torch.equal(plain[i]["pixel_values"], cached[i]["pixel_values"])
        assert torch.equal(plain[i]["labels"], cached[i]["labels"])


def test_changed_source_invalidates_entry(synthetic_dataset, tmp_path):
    sample = _samples(synthetic_dataset)[0]
    cache = data.ResizedCache(tmp_path, (16, 16))
    before = cache.key(sample)
    stat = sample.image_path.stat()
    os.utime(sample.image_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    assert cache.key(sample) != before


def test_eviction_keeps_cache_under_bound_and_prefers_recent(synthetic_dataset, tmp_path):
    samples = _samples(synthetic_dataset)
    cache = data.ResizedCache(tmp_path, (16, 16), max_bytes=10**9)
    data.warm_cache(cache, samples)
    one_entry = cache.size_bytes() // len(samples)
    for age, sample in enumerate(samples):
        entry = cache.directory / f"{cache.key(sample)}.npy"
        os.utime(entry, (1_000_000 + age, 1_000_000 + age))
    cache.max_bytes = one_entry * 3
    removed = cache.evict()
    assert removed >= len(samples) - 4
    assert cache.size_bytes() <= cache.max_bytes
    newest = cache.directory / f"{cache.key(samples[-1])}.npy"
    oldest = cache.directory / f"{cache.key(samples[0])}.npy"
    assert newest.exists()
    assert not oldest.exists()


def test_cache_max_bytes_env(monkeypatch):
    monkeypatch.setenv("TRAIN_CACHE_MAX_MB", "2")
    assert data.cache_max_bytes() == 2 * 1024 * 1024
    monkeypatch.setenv("TRAIN_CACHE_MAX_MB", "lots")
    assert data.cache_max_bytes() == 4096 * 1024 * 1024


def test_cached_mask_has_only_palette_colours(synthetic_dataset, tmp_path):
    samples = _samples(synthetic_dataset)
    palette = data.build_palette([s.mask_path for s in samples])
    cache = data.ResizedCache(tmp_path, (12, 12))
    allowed = {(0, 0, 0), *palette.colors}
    for sample in samples:
        _, mask = cache.get(sample)
        assert {tuple(c) for c in mask.reshape(-1, 3).tolist()} <= allowed
        labels = data.decode_mask(mask, palette)
        assert set(np.unique(labels)) <= set(range(palette.num_classes))


def test_warm_cache_parallel_reports_progress_and_palette_matches(synthetic_dataset, tmp_path):
    samples = _samples(synthetic_dataset)
    cache = data.ResizedCache(tmp_path, (16, 16))
    lines: list[str] = []
    data.warm_cache(cache, samples, workers=2, progress=lines.append)
    assert lines[-1] == f"caching resized images {len(samples)}/{len(samples)}"
    palette_progress: list[tuple[int, int]] = []
    cached = data.build_palette_cached(
        cache, samples, progress=lambda done, total: palette_progress.append((done, total))
    )
    assert palette_progress[-1] == (len(samples), len(samples))
    assert cached == data.build_palette([s.mask_path for s in samples])
