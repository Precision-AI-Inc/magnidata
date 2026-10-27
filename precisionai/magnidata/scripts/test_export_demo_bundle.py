# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for export_demo_bundle: capped images, masks kept in step, and the embedding join.

Run:  python -m pytest precisionai/magnidata/scripts/test_export_demo_bundle.py -q
"""

import csv
import json
import re
import shutil
from pathlib import Path

import pytest
from PIL import Image

from precisionai.magnidata.scripts import export_demo_bundle as bundles


def _write_embeddings(root: Path, names: list[str], stem: str = "syn") -> None:
    table = {name: [float(i), 1.0, 0.5] for i, name in enumerate(names)}
    (root / "data_user" / f"{stem}.json").write_text(json.dumps({"embeddings": table}))


def _export(dataset, out: Path, **options):
    _write_embeddings(dataset.root, dataset.names)
    options.setdefault("workers", 1)
    return bundles.export_bundle(dataset.root, "syn", out, name="Synthetic", description="eight tiles", **options)


def test_images_are_capped_and_masks_stay_the_same_size(synthetic_dataset, tmp_path):
    out = tmp_path / "bundle"

    _export(synthetic_dataset, out, max_size=16, thumbnail_size=8)

    images = out / "syn_images"
    with Image.open(images / "images" / "img_00.jpg") as image:
        assert image.format == "JPEG"
        assert image.size == (16, 12)
    with Image.open(images / "thumbnails" / "img_00.jpg") as thumbnail:
        assert thumbnail.size == (8, 6)
    with Image.open(images / "masks" / "img_00.png") as mask:
        assert mask.size == (16, 12)
        colours = {colour for _, colour in mask.convert("RGB").getcolors()}
    assert colours <= {(0, 0, 0), *synthetic_dataset.colors}


def test_images_smaller_than_the_cap_are_not_enlarged(synthetic_dataset, tmp_path):
    _export(synthetic_dataset, tmp_path / "bundle", max_size=1000)

    with Image.open(tmp_path / "bundle" / "syn_images" / "images" / "img_00.jpg") as image:
        assert image.size == (32, 24)


def test_table_and_embeddings_are_renamed_together(synthetic_dataset, tmp_path):
    out = tmp_path / "bundle"

    manifest = _export(synthetic_dataset, out, max_size=16)

    with open(out / "syn.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["image_path"] for row in rows] == [f"data_user/syn_images/images/img_{i:02d}.jpg" for i in range(8)]
    assert [row["width"] for row in rows] == ["32"] * 8  # features keep the values of the full-size images
    embeddings = json.loads((out / "syn.json").read_text())["embeddings"]
    assert sorted(embeddings) == [f"img_{i:02d}.jpg" for i in range(8)]
    assert embeddings["img_03.jpg"] == [3.0, 1.0, 0.5]
    assert manifest["row_count"] == 8
    assert bundles.validate_bundle(out) == manifest


def test_a_thumbnail_column_is_pointed_at_the_new_thumbnails(synthetic_dataset, tmp_path):
    table = synthetic_dataset.csv
    with open(table, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["thumbnail_image"] = "data_user/syn/thumbs/old.jpg"
    with open(table, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    out = tmp_path / "bundle"
    _export(synthetic_dataset, out, max_size=16)

    with open(out / "syn.csv", newline="") as handle:
        first = next(csv.DictReader(handle))
    assert first["thumbnail_image"] == "data_user/syn_images/thumbnails/img_00.jpg"
    assert (out / "syn_images" / "thumbnails" / "img_00.jpg").is_file()


def test_a_dataset_without_masks_exports_images_only(synthetic_dataset, tmp_path):
    shutil.rmtree(synthetic_dataset.root / "data_user" / "syn" / "masks")
    out = tmp_path / "bundle"

    manifest = _export(synthetic_dataset, out, max_size=16)

    assert manifest["has_masks"] is False
    assert not (out / "syn_images" / "masks").exists()


def test_conversion_in_several_processes_gives_the_same_bundle(synthetic_dataset, tmp_path):
    out = tmp_path / "bundle"

    _export(synthetic_dataset, out, max_size=16, workers=2)

    assert len(list((out / "syn_images" / "images").glob("*.jpg"))) == 8
    assert len(list((out / "syn_images" / "masks").glob("*.png"))) == 8


def test_an_output_directory_that_is_not_empty_is_refused(synthetic_dataset, tmp_path):
    out = tmp_path / "bundle"
    out.mkdir()
    (out / "leftover.txt").write_text("x")

    with pytest.raises(bundles.BundleError, match="is not empty"):
        _export(synthetic_dataset, out)


def test_a_missing_embedding_stops_the_export(synthetic_dataset, tmp_path):
    names = [name for name in synthetic_dataset.names if name != "img_03.png"]
    _write_embeddings(synthetic_dataset.root, names)

    with pytest.raises(bundles.BundleError, match=re.escape("no embedding for img_03.png")):
        bundles.export_bundle(synthetic_dataset.root, "syn", tmp_path / "bundle", name="n", description="d", workers=1)


def test_a_missing_image_stops_the_export(synthetic_dataset, tmp_path):
    (synthetic_dataset.root / "data_user" / "syn" / "images" / "img_05.png").unlink()

    with pytest.raises(bundles.BundleError, match=re.escape("image not found: data_user/syn/images/img_05.png")):
        _export(synthetic_dataset, tmp_path / "bundle")


def test_two_images_that_differ_only_by_extension_are_refused(synthetic_dataset, tmp_path):
    images = synthetic_dataset.root / "data_user" / "syn" / "images"
    shutil.copy(images / "img_00.png", images / "img_00.bmp")
    with open(synthetic_dataset.csv, "a", newline="") as handle:
        handle.write("data_user/syn/images/img_00.bmp,32,24\n")

    with pytest.raises(bundles.BundleError, match="share a name"):
        _export(synthetic_dataset, tmp_path / "bundle")


def test_validation_names_a_file_that_went_missing(synthetic_dataset, tmp_path):
    out = tmp_path / "bundle"
    _export(synthetic_dataset, out, max_size=16)
    (out / "syn_images" / "images" / "img_00.jpg").unlink()

    with pytest.raises(bundles.BundleError, match=re.escape("missing data_user/syn_images/images/img_00.jpg")):
        bundles.validate_bundle(out)


def test_validation_names_an_embedding_that_went_missing(synthetic_dataset, tmp_path):
    out = tmp_path / "bundle"
    _export(synthetic_dataset, out, max_size=16)
    table = json.loads((out / "syn.json").read_text())
    del table["embeddings"]["img_02.jpg"]
    (out / "syn.json").write_text(json.dumps(table))

    with pytest.raises(bundles.BundleError, match=re.escape("no embedding for img_02.jpg")):
        bundles.validate_bundle(out)


def test_validation_rejects_a_directory_that_is_not_a_bundle(tmp_path):
    with pytest.raises(bundles.BundleError, match=re.escape("has no bundle.json")):
        bundles.validate_bundle(tmp_path)


def test_validation_rejects_a_newer_bundle_format(synthetic_dataset, tmp_path):
    out = tmp_path / "bundle"
    _export(synthetic_dataset, out, max_size=16)
    manifest = json.loads((out / "bundle.json").read_text())
    manifest["format"] = 99
    (out / "bundle.json").write_text(json.dumps(manifest))

    with pytest.raises(bundles.BundleError, match="unsupported bundle format 99"):
        bundles.validate_bundle(out)


def test_validation_reports_a_row_count_that_disagrees_with_the_manifest(synthetic_dataset, tmp_path):
    out = tmp_path / "bundle"
    _export(synthetic_dataset, out, max_size=16)
    manifest = json.loads((out / "bundle.json").read_text())
    manifest["row_count"] = 9
    (out / "bundle.json").write_text(json.dumps(manifest))

    with pytest.raises(bundles.BundleError, match="table has 8 rows, the manifest says 9"):
        bundles.validate_bundle(out)


def test_many_problems_are_summarised(synthetic_dataset, tmp_path):
    out = tmp_path / "bundle"
    _export(synthetic_dataset, out, max_size=16)
    for path in (out / "syn_images" / "images").glob("*.jpg"):
        path.unlink()

    with pytest.raises(bundles.BundleError, match=r"\(and \d+ more\)"):
        bundles.validate_bundle(out)


def test_command_line_exports_a_bundle(synthetic_dataset, tmp_path, capsys):
    _write_embeddings(synthetic_dataset.root, synthetic_dataset.names)
    out = tmp_path / "bundle"
    argv = [
        "--data-root", str(synthetic_dataset.root), "--stem", "syn", "--name", "Synthetic",
        "--description", "eight tiles", "--out", str(out), "--max-size", "16", "--workers", "1",
    ]  # fmt: skip

    assert bundles.main(argv) == 0

    assert "exported 8 images" in capsys.readouterr().out
    assert (out / "bundle.json").is_file()


def test_command_line_reports_a_failure_without_a_traceback(synthetic_dataset, tmp_path, capsys):
    out = tmp_path / "bundle"
    out.mkdir()
    (out / "leftover.txt").write_text("x")
    argv = [
        "--data-root", str(synthetic_dataset.root), "--stem", "syn", "--name", "n",
        "--description", "d", "--out", str(out),
    ]  # fmt: skip

    assert bundles.main(argv) == 1

    assert "export failed" in capsys.readouterr().err
