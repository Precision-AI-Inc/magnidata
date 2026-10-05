# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Segmentation models available as training presets.

Every model maps ``[B, 3, H, W]`` images to ``[B, C, H, W]`` logits at the input resolution.
"""

import torch
from torch import nn
from torch.nn import functional

from precisionai.magnidata.training import contract

try:
    import torchvision.models.segmentation as tv_segmentation
    from torchvision.models import MobileNet_V3_Large_Weights
except ImportError:
    raise ImportError("torchvision is required for training models: pip install -e '.[training]'") from None


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class TinyUNet(nn.Module):
    """Three-level U-Net with about 0.1M parameters: no downloads, fast on CPU."""

    def __init__(self, num_classes: int, widths: tuple[int, int, int] = (16, 32, 64)) -> None:
        super().__init__()
        w1, w2, w3 = widths
        self.enc1 = _conv_block(3, w1)
        self.enc2 = _conv_block(w1, w2)
        self.bottom = _conv_block(w2, w3)
        self.dec2 = _conv_block(w3 + w2, w2)
        self.dec1 = _conv_block(w2 + w1, w1)
        self.head = nn.Conv2d(w1, num_classes, 1)

    @staticmethod
    def _up(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up = functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([up, skip], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for a batch of images."""
        e1 = self.enc1(x)
        e2 = self.enc2(functional.max_pool2d(e1, 2))
        mid = self.bottom(functional.max_pool2d(e2, 2))
        d2 = self.dec2(self._up(mid, e2))
        d1 = self.dec1(self._up(d2, e1))
        return self.head(d1)


class _TorchvisionSegmenter(nn.Module):
    """Adapter turning torchvision's ``{"out": logits}`` output into a plain tensor."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for a batch of images."""
        return self.model(x)["out"]


def build_model(preset: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build the model of a preset.

    Parameters
    ----------
    preset : str
        One of `contract.MODEL_PRESETS`.
    num_classes : int
        Number of classes, background included.
    pretrained : bool
        Load ImageNet backbone weights where the preset has a backbone (downloads on first use).
    """
    if preset == "tiny_unet":
        return TinyUNet(num_classes)
    if preset == "lraspp_mobilenet":
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_segmentation.lraspp_mobilenet_v3_large(
            weights=None, weights_backbone=weights, num_classes=num_classes
        )
        return _TorchvisionSegmenter(model)
    raise ValueError(f"unknown preset {preset!r}: choose from {', '.join(contract.MODEL_PRESETS)}")
