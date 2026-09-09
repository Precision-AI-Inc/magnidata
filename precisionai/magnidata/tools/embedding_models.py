# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Registry of embedding-generation backbones: name -> load/embed functions.

Heavy deps (torch, timm) are imported at module level behind a try/except with an
availability flag — matching nima.py's pattern — so importing this module (and
MODELS) never requires them; only actually calling a model's ``load``/``embed``
does, and those raise ``ImportError`` if the deps are unavailable.

Adding a future model (e.g. an agriculture-trained checkpoint) means adding one
more MODELS entry here — nothing in embeddings.py or the API changes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

try:
    import timm
    import timm.data
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    timm = None
    torch = None
    _TORCH_AVAILABLE = False


@dataclass
class EmbeddingModel:
    """A registered embedding backbone: its declared dim plus load/embed callables."""

    name: str
    dim: int
    load: Callable[[str], Any]  # device -> model handle
    embed: Callable[[Any, np.ndarray], np.ndarray]  # (handle, rgb uint8 HxWx3) -> float32 vector


def _load_timm_model(model_name: str, device: str) -> tuple[Any, Any, Any]:
    if not _TORCH_AVAILABLE or timm is None or torch is None:
        raise ImportError("torch and timm are required for embeddings: pip install 'magnidata[ml]'") from None
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval()
    model.to(device)
    # timm.data re-exports these via a wildcard import with no __all__, so pyright
    # treats them as private even though they're timm's documented public API for
    # building a model-matched preprocessing transform.
    cfg = timm.data.resolve_data_config({}, model=model)  # pyright: ignore[reportPrivateImportUsage]
    transform = timm.data.create_transform(**cfg)  # pyright: ignore[reportPrivateImportUsage]
    return model, transform, torch.device(device)


def _embed_timm_model(handle: Any, rgb: np.ndarray) -> np.ndarray:
    if not _TORCH_AVAILABLE or torch is None:
        raise ImportError("torch is required for embeddings: pip install 'magnidata[ml]'") from None
    model, transform, device = handle
    tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
    return out.squeeze(0).cpu().numpy().astype(np.float32)


def _load_dinov2(device: str) -> tuple[Any, Any, Any]:
    return _load_timm_model("vit_small_patch14_dinov2.lvd142m", device)


def _load_resnet18(device: str) -> tuple[Any, Any, Any]:
    return _load_timm_model("resnet18", device)


def _load_mobilenet(device: str) -> tuple[Any, Any, Any]:
    return _load_timm_model("mobilenetv3_small_100", device)


MODELS: dict[str, EmbeddingModel] = {
    "dinov2": EmbeddingModel(name="dinov2", dim=384, load=_load_dinov2, embed=_embed_timm_model),
    "resnet18": EmbeddingModel(name="resnet18", dim=512, load=_load_resnet18, embed=_embed_timm_model),
    "mobilenet": EmbeddingModel(name="mobilenet", dim=576, load=_load_mobilenet, embed=_embed_timm_model),
}
