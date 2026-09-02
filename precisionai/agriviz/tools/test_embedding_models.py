# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Structural tests for the embedding-model registry. Does not call load() or
embed() on the real entries — no model download, no network, no torch/timm
required for this test to run."""
from .embedding_models import MODELS, EmbeddingModel


def test_dinov2_registered():
    assert "dinov2" in MODELS
    entry = MODELS["dinov2"]
    assert isinstance(entry, EmbeddingModel)
    assert entry.name == "dinov2"
    assert entry.dim == 384
    assert callable(entry.load)
    assert callable(entry.embed)


def test_embedding_model_is_a_plain_dataclass():
    fake = EmbeddingModel(name="fake", dim=4, load=lambda device: None, embed=lambda h, rgb: rgb)
    assert fake.name == "fake"
    assert fake.dim == 4


def test_resnet18_registered():
    assert "resnet18" in MODELS
    entry = MODELS["resnet18"]
    assert isinstance(entry, EmbeddingModel)
    assert entry.name == "resnet18"
    assert entry.dim == 512
    assert callable(entry.load)
    assert callable(entry.embed)


def test_mobilenet_registered():
    assert "mobilenet" in MODELS
    entry = MODELS["mobilenet"]
    assert isinstance(entry, EmbeddingModel)
    assert entry.name == "mobilenet"
    assert entry.dim == 576
    assert callable(entry.load)
    assert callable(entry.embed)


def test_all_models_share_the_timm_embed_function():
    """dinov2/resnet18/mobilenet all embed the same way — only the backbone differs."""
    assert MODELS["dinov2"].embed is MODELS["resnet18"].embed is MODELS["mobilenet"].embed
