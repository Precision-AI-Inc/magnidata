# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the provenance sidecar builder. Pure stdlib + a tiny file
fixture — no torch/pyiqa/timm required (an unlisted/uninstalled module in
version_modules just reports version "absent", which is fine here)."""

import os

from .provenance import build


def test_build_uses_generic_param_names(tmp_path):
    input_path = os.path.join(str(tmp_path), "in.csv")
    with open(input_path, "w") as f:
        f.write("a,b\n1,2\n")
    output_path = os.path.join(str(tmp_path), "out.csv")

    prov = build(input_path, output_path, {"k": "v"}, "some-schema/1.0.0", weight_paths=[])

    assert prov["feature_schema_version"] == "some-schema/1.0.0"
    assert prov["input_path"]["path"] == input_path
    assert prov["output_path"] == output_path
    assert prov["params"] == {"k": "v"}
    assert prov["model_weights"] == []


def test_build_custom_version_modules(tmp_path):
    input_path = os.path.join(str(tmp_path), "in.csv")
    with open(input_path, "w") as f:
        f.write("a\n1\n")
    output_path = os.path.join(str(tmp_path), "out.json")

    prov = build(input_path, output_path, {}, "dataviz-embeddings/dinov2", version_modules=("numpy", "PIL"))

    assert set(prov["versions"].keys()) == {"numpy", "PIL", "python"}


def test_build_default_version_modules_unchanged(tmp_path):
    input_path = os.path.join(str(tmp_path), "in.csv")
    with open(input_path, "w") as f:
        f.write("a\n1\n")
    output_path = os.path.join(str(tmp_path), "out.csv")

    prov = build(input_path, output_path, {}, "dataviz-features/2.1.0")

    assert set(prov["versions"].keys()) == {"numpy", "PIL", "torch", "pyiqa", "python"}
