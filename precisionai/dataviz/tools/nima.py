# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Learned no-reference image-quality scoring via pyiqa (PyTorch).

Runs the NIMA aesthetic models plus general technical-quality metrics (NIQE, BRISQUE).
Heavy deps (torch + pyiqa) are imported at module level behind a try/except with an
availability flag (not inside functions), so importing this module never requires
them — the rest of the tool, and the pixel/COCO/metadata/quality unit tests, still
work without them (use ``--skip-nima``, which skips *all* learned IQA here).
Constructing ``NimaScorer`` is what actually needs them; it raises ``ImportError``
if they are unavailable.

Caveat: NIMA/MUSIQ-style models are trained on consumer/aesthetic photos and can be
out-of-distribution on this tool's typically top-down, technical imagery. NIQE is
fully blind (no aesthetic labels), so it is the least domain-biased single score.
Keep them as separate columns.

Determinism: models run in ``eval()`` under ``no_grad``, with deterministic algorithms
enabled and a fixed seed. Reproducibility across machines additionally requires a pinned
torch/pyiqa and identical model weights (hashed into the provenance sidecar).
"""

from __future__ import annotations

import contextlib
import glob
import os
from typing import Any

import numpy as np
from PIL import Image

try:
    import pyiqa
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    pyiqa = None
    torch = None
    _TORCH_AVAILABLE = False

# pyiqa metric name -> output column
VARIANTS = [
    ("nima", "nima_ava"),  # Inception backbone, AVA (aesthetic, higher=better)
    ("nima-vgg16-ava", "nima_vgg16_ava"),  # VGG16 backbone, AVA (aesthetic)
    ("niqe", "niqe"),  # blind NSS, no labels (lower=better) — most general
    ("brisque", "brisque"),  # NSS distortion model (lower=better) — technical
]

# Learned IQA models are trained on normal-sized photos; the full ~16 MP images this
# tool handles are out-of-distribution and also make the NSS metrics (BRISQUE/NIQE) numerically
# fragile. Score them on a fixed, recorded downscale (deterministic LANCZOS) — this is
# the standard preprocessing for these metrics, not the full-res path used by the
# deterministic pixel features.
PROC_LONGEST = 512
RESAMPLE = "lanczos"
_RESAMPLE_FILTER = Image.Resampling.LANCZOS

# pyiqa's BRISQUE instance is stateful: a reused instance fails on every other call
# ("size of tensor a (36) must match b (774)"). It is a tiny SVM, so we recreate it
# per image instead of caching it. (NIQE/NIMA are safe to reuse.)
_PER_CALL = {"brisque"}


class NimaScorer:
    """Loads and runs the configured pyiqa metrics (NIMA variants, NIQE, BRISQUE)."""

    def __init__(self, device: str = "cpu") -> None:
        if not _TORCH_AVAILABLE or torch is None or pyiqa is None:
            raise ImportError("torch and pyiqa are required for NIMA scoring: pip install 'pai-dataviz[ml]'") from None

        torch.manual_seed(0)
        with contextlib.suppress(Exception):
            torch.use_deterministic_algorithms(True, warn_only=True)

        self._torch = torch
        self._pyiqa = pyiqa
        self.device = device
        # Cache reusable metrics; defer stateful ones (recreated per call in score()).
        self.metrics = {
            col: pyiqa.create_metric(name, device=device).eval() for name, col in VARIANTS if name not in _PER_CALL
        }

    @property
    def columns(self) -> list[str]:
        """Return the output column names, in ``VARIANTS`` order."""
        return [col for _name, col in VARIANTS]

    def proc_params(self) -> dict:
        """Return the deterministic preprocessing params to record in the provenance sidecar."""
        return {"iqa_proc_longest": PROC_LONGEST, "iqa_resample": RESAMPLE}

    def _to_tensor(self, image_path: str) -> Any:
        im = Image.open(image_path).convert("RGB")
        w, h = im.size
        s = PROC_LONGEST / max(w, h)
        if s < 1.0:
            im = im.resize((max(1, round(w * s)), max(1, round(h * s))), _RESAMPLE_FILTER)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        return self._torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def weight_paths(self) -> list[str]:
        """Best-effort list of loaded weight files, for provenance hashing.

        pyiqa caches weights under torch's hub dir (e.g. ~/.cache/torch/hub/pyiqa).
        """
        cache = os.path.join(self._torch.hub.get_dir(), "pyiqa")
        return sorted(glob.glob(os.path.join(cache, "*.pth"))) if os.path.isdir(cache) else []

    def score(self, image_path: str) -> dict:
        """Score one image on every metric in ``VARIANTS``, keyed by output column name."""
        tensor = self._to_tensor(image_path)
        out = {}
        with self._torch.no_grad():
            for name, col in VARIANTS:
                metric = (
                    self._pyiqa.create_metric(name, device=self.device).eval()
                    if name in _PER_CALL
                    else self.metrics[col]
                )
                out[col] = float(metric(tensor).item())
        return out
