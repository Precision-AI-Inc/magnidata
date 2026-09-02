# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Registry of embedding-generation backbones: name -> load/embed functions.

Heavy deps (torch, timm) are imported lazily inside each model's load()/embed()
— matching nima.py's pattern — so importing this module (and MODELS) never
requires them; only actually running a model does.

Adding a future model (e.g. an agriculture-trained checkpoint) means adding one
more MODELS entry here — nothing in embeddings.py or the API changes.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EmbeddingModel:
    name: str
    dim: int
    load: Callable[[str], Any]                      # device -> model handle
    embed: Callable[[Any, np.ndarray], np.ndarray]   # (handle, rgb uint8 HxWx3) -> float32 vector


def _load_timm_model(model_name: str, device: str):
    import timm
    import torch

    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval()
    model.to(device)
    cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**cfg)
    return model, transform, torch.device(device)


def _embed_timm_model(handle, rgb: np.ndarray) -> np.ndarray:
    import torch
    from PIL import Image

    model, transform, device = handle
    tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
    return out.squeeze(0).cpu().numpy().astype(np.float32)


def _load_dinov2(device: str):
    return _load_timm_model("vit_small_patch14_dinov2.lvd142m", device)


def _load_resnet18(device: str):
    return _load_timm_model("resnet18", device)


def _load_mobilenet(device: str):
    return _load_timm_model("mobilenetv3_small_100", device)


MODELS: dict[str, EmbeddingModel] = {
    "dinov2": EmbeddingModel(name="dinov2", dim=384, load=_load_dinov2, embed=_embed_timm_model),
    "resnet18": EmbeddingModel(name="resnet18", dim=512, load=_load_resnet18, embed=_embed_timm_model),
    "mobilenet": EmbeddingModel(name="mobilenet", dim=576, load=_load_mobilenet, embed=_embed_timm_model),
}
