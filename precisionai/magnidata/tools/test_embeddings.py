# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the embeddings tool. Uses a fake in-memory embedding model —
no real DINOv2 download, no torch/timm/network required.

Run:  python -m pytest precisionai/magnidata/tools/test_embeddings.py -q
"""

import csv
import json
import os

import numpy as np
import pytest
from PIL import Image

from . import embedding_models
from .embedding_models import EmbeddingModel
from .embeddings import AllRowsFailedError, embed_row, main, run


def _fake_model(dim=4):
    def load(device):
        return {"device": device}

    def embed(handle, rgb):
        vec = np.array([rgb[..., 0].mean(), rgb[..., 1].mean(), rgb[..., 2].mean(), 1.0], dtype=np.float32)
        return vec[:dim]

    return EmbeddingModel(name="fake", dim=dim, load=load, embed=embed)


def _write_image(path, rgb):
    Image.fromarray(rgb, "RGB").save(path)


def _write_csv(path, rows, fieldname="image_path"):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[fieldname])
        w.writeheader()
        for r in rows:
            w.writerow({fieldname: r})


def test_embed_row_reads_image_and_calls_model(tmp_path):
    img_path = os.path.join(str(tmp_path), "a.png")
    _write_image(img_path, np.full((4, 4, 3), 100, np.uint8))
    model = _fake_model()
    handle = model.load("cpu")

    name, vec, source, err = embed_row(img_path, model, handle)

    assert name == "a.png"
    assert vec is not None
    assert len(vec) == 4
    assert source == "fullres"
    assert err == ""


def test_embed_row_missing_image_returns_none(tmp_path):
    model = _fake_model()
    handle = model.load("cpu")

    name, vec, source, err = embed_row(os.path.join(str(tmp_path), "missing.png"), model, handle)

    assert name == "missing.png"
    assert vec is None
    assert source == "missing"
    assert err == "image:not_found (full-res and thumbnails)"


def test_embed_row_unreadable_image_reports_reason(tmp_path):
    """A file that exists but is not a decodable image surfaces the read failure."""
    bad_path = os.path.join(str(tmp_path), "bad.png")
    with open(bad_path, "w") as f:
        f.write("this is not a PNG")
    model = _fake_model()
    handle = model.load("cpu")

    name, vec, source, err = embed_row(bad_path, model, handle)

    assert (name, vec, source) == ("bad.png", None, "fullres")
    assert err.startswith("embed:")
    assert err != "embed:"


def test_run_writes_embeddings_json_and_provenance(tmp_path, monkeypatch):
    monkeypatch.setitem(embedding_models.MODELS, "fake", _fake_model())
    ds = str(tmp_path)
    img1, img2 = os.path.join(ds, "a.png"), os.path.join(ds, "b.png")
    _write_image(img1, np.full((4, 4, 3), 50, np.uint8))
    _write_image(img2, np.full((4, 4, 3), 200, np.uint8))
    csv_path = os.path.join(ds, "in.csv")
    _write_csv(csv_path, [img1, img2])
    out_path = os.path.join(ds, "out.json")

    n = run(csv_path, out_path, model="fake", device="cpu")

    assert n == 2
    with open(out_path) as f:
        data = json.load(f)
    assert set(data["embeddings"].keys()) == {"a.png", "b.png"}
    assert len(data["embeddings"]["a.png"]) == 4

    with open(out_path + ".provenance.json") as f:
        prov = json.load(f)
    assert prov["feature_schema_version"] == "dataviz-embeddings/fake"
    assert prov["params"]["model"] == "fake"

    # Records what the run actually did, not just its declared config.
    stats = prov["params"]["stats"]
    assert stats["rows"] == 2
    assert stats["embedded"] == 2
    assert stats["skipped"] == 0
    assert stats["sources"] == {"fullres": 2}
    assert stats["observed_dim"] == 4  # observed, not the registry's declared dim

    # Embeddings-specific determinism notes, not the feature extractor's.
    notes = " ".join(prov["determinism_notes"]).lower()
    assert "resize" in notes
    assert "nima" not in notes
    assert "polygon" not in notes


def test_run_skips_missing_images(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(embedding_models.MODELS, "fake", _fake_model())
    ds = str(tmp_path)
    img1 = os.path.join(ds, "a.png")
    _write_image(img1, np.full((4, 4, 3), 50, np.uint8))
    csv_path = os.path.join(ds, "in.csv")
    _write_csv(csv_path, [img1, os.path.join(ds, "missing.png")])
    out_path = os.path.join(ds, "out.json")

    n = run(csv_path, out_path, model="fake", device="cpu")

    assert n == 1
    with open(out_path) as f:
        data = json.load(f)
    assert set(data["embeddings"].keys()) == {"a.png"}

    # The reason is reported per row, not just counted (features.py's convention).
    err = capsys.readouterr().err
    assert "WARNING: 1/2 rows had errors:" in err
    assert "missing.png: image:not_found (full-res and thumbnails)" in err

    with open(out_path + ".provenance.json") as f:
        stats = json.load(f)["params"]["stats"]
    assert stats == {"rows": 2, "embedded": 1, "skipped": 1, "sources": {"fullres": 1, "missing": 1}, "observed_dim": 4}


def test_run_total_failure_refuses_to_overwrite_existing_output(tmp_path, monkeypatch):
    """Every row failing must not clobber a good pre-existing embeddings file."""
    monkeypatch.setitem(embedding_models.MODELS, "fake", _fake_model())
    ds = str(tmp_path)
    csv_path = os.path.join(ds, "in.csv")
    _write_csv(csv_path, [os.path.join(ds, "gone1.png"), os.path.join(ds, "gone2.png")])
    out_path = os.path.join(ds, "out.json")
    prov_path = out_path + ".provenance.json"
    existing = {"embeddings": {"good.png": [1.0, 2.0, 3.0, 4.0]}}
    with open(out_path, "w") as f:
        json.dump(existing, f)

    with pytest.raises(AllRowsFailedError):
        run(csv_path, out_path, model="fake", device="cpu")

    with open(out_path) as f:
        assert json.load(f) == existing  # untouched
    assert not os.path.exists(prov_path)  # no sidecar for an unwritten output


def test_main_exits_nonzero_on_total_failure(tmp_path, monkeypatch):
    monkeypatch.setitem(embedding_models.MODELS, "fake", _fake_model())
    ds = str(tmp_path)
    csv_path = os.path.join(ds, "in.csv")
    _write_csv(csv_path, [os.path.join(ds, "gone.png")])
    out_path = os.path.join(ds, "out.json")

    with pytest.raises(SystemExit) as exc:
        main(["--input", csv_path, "--output", out_path, "--model", "fake"])

    assert exc.value.code == 1
    assert not os.path.exists(out_path)


def test_run_empty_input_csv_succeeds(tmp_path, monkeypatch):
    """Zero rows is not a failure — nothing could fail — so write an empty file."""
    monkeypatch.setitem(embedding_models.MODELS, "fake", _fake_model())
    ds = str(tmp_path)
    csv_path = os.path.join(ds, "in.csv")
    _write_csv(csv_path, [])
    out_path = os.path.join(ds, "out.json")

    n = run(csv_path, out_path, model="fake", device="cpu")

    assert n == 0
    with open(out_path) as f:
        assert json.load(f) == {"embeddings": {}}
    with open(out_path + ".provenance.json") as f:
        stats = json.load(f)["params"]["stats"]
    assert stats["rows"] == 0
    assert stats["embedded"] == 0
    assert stats["observed_dim"] is None


def test_run_unknown_model_raises(tmp_path):
    csv_path = os.path.join(str(tmp_path), "in.csv")
    _write_csv(csv_path, [])
    with pytest.raises(ValueError, match="unknown model"):
        run(csv_path, os.path.join(str(tmp_path), "out.json"), model="not-a-model")
