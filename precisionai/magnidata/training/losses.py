# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Per-sample semantic-segmentation losses.

Every term is computed per image (shape ``[B]``) so the training loop can record which
samples are hard, and is reduced to the usual scalar only for backpropagation. Logits are
``[B, C, H, W]`` and targets are ``[B, H, W]`` class indices; pixels equal to
``ignore_index`` are excluded everywhere.

Imported only by trainer code, never by the training package itself, because it needs torch.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from precisionai.magnidata.training import contract

try:
    import torch
    from torch.nn import functional
except ImportError:
    raise ImportError("torch is required for training losses: pip install -e '.[ml]'") from None

IGNORE_INDEX = 255

LossFn = Callable[..., torch.Tensor]


def _valid_mask(target: torch.Tensor, ignore_index: int) -> torch.Tensor:
    return target != ignore_index


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Average ``[B, H, W]`` values over each image's valid pixels; 0 where none are valid."""
    count = valid.flatten(1).sum(1)
    total = (values * valid).flatten(1).sum(1)
    return torch.where(count > 0, total / count.clamp(min=1), torch.zeros_like(total))


def cce_per_sample(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """Return each image's pixel-wise cross-entropy averaged over its valid pixels.

    An image with no valid pixels scores 0 rather than NaN.
    """
    pixel = functional.cross_entropy(logits, target, ignore_index=ignore_index, reduction="none")
    return _masked_mean(pixel, _valid_mask(target, ignore_index))


def dice_per_sample(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int | None = None,
    ignore_index: int = IGNORE_INDEX,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return each image's soft multi-class Dice loss, ``1 - mean_c dice_c``.

    The softmax prediction is compared with the one-hot target on valid pixels. A class
    absent from both the hard prediction and the target of an image is left out of that
    image's mean, so rare classes do not drag every image toward a constant. An image with
    no valid pixels scores 0.
    """
    classes = num_classes or logits.shape[1]
    valid = _valid_mask(target, ignore_index)
    mask = valid.unsqueeze(1)
    probs = logits.softmax(1) * mask
    safe = target.masked_fill(~valid, 0)
    onehot = functional.one_hot(safe, classes).permute(0, 3, 1, 2).to(probs.dtype) * mask
    hard = functional.one_hot(logits.argmax(1), classes).permute(0, 3, 1, 2).to(probs.dtype) * mask

    inter = (probs * onehot).flatten(2).sum(2)
    denom = probs.flatten(2).sum(2) + onehot.flatten(2).sum(2)
    dice = (2 * inter + eps) / (denom + eps)
    present = (hard.flatten(2).sum(2) + onehot.flatten(2).sum(2)) > 0
    n_present = present.sum(1)
    loss = 1 - (dice * present).sum(1) / n_present.clamp(min=1)
    return torch.where(n_present > 0, loss, torch.zeros_like(loss))


def bce_fg_per_sample(
    logits: torch.Tensor,
    target: torch.Tensor,
    background_index: int = 0,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Return each image's foreground-vs-background BCE, using ``1 - p(background)``.

    Collapsing all non-background classes into one isolates how well the model separates
    crop from soil, independent of the finer class split.
    """
    valid = _valid_mask(target, ignore_index)
    p_fg = (1 - logits.softmax(1)[:, background_index]).clamp(1e-7, 1 - 1e-7)
    is_fg = (target != background_index).to(p_fg.dtype)
    pixel = functional.binary_cross_entropy(p_fg, is_fg, reduction="none")
    return _masked_mean(pixel, valid)


def iou_per_sample(
    logits: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = IGNORE_INDEX
) -> torch.Tensor:
    """Return each image's hard mean IoU over the classes present in it (no gradient).

    An image with no valid pixels scores 1, consistent with its zero loss.
    """
    with torch.no_grad():
        valid = _valid_mask(target, ignore_index)
        mask = valid.unsqueeze(1)
        pred = functional.one_hot(logits.argmax(1), num_classes).permute(0, 3, 1, 2) * mask
        truth = functional.one_hot(target.masked_fill(~valid, 0), num_classes).permute(0, 3, 1, 2) * mask
        inter = (pred * truth).flatten(2).sum(2).to(logits.dtype)
        union = (pred + truth).clamp(max=1).flatten(2).sum(2).to(logits.dtype)
        present = union > 0
        iou = inter / union.clamp(min=1)
        n_present = present.sum(1)
        mean = (iou * present).sum(1) / n_present.clamp(min=1)
        return torch.where(n_present > 0, mean, torch.ones_like(mean))


def confusion_counts(
    logits: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = IGNORE_INDEX
) -> torch.Tensor:
    """Return a ``[C, C]`` int64 matrix (rows: target, columns: prediction) over valid pixels.

    Matrices from several batches add up, so a trainer can derive dataset-level mIoU and
    per-class IoU after a full pass.
    """
    with torch.no_grad():
        valid = _valid_mask(target, ignore_index)
        pred = logits.argmax(1)[valid]
        truth = target[valid]
        flat = truth * num_classes + pred
        return torch.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


LOSS_REGISTRY: dict[str, LossFn] = {
    "dice": dice_per_sample,
    "cce": cce_per_sample,
    "bce": bce_fg_per_sample,
}


@dataclass
class LossOutput:
    """Result of `SegmentationCriterion`: the scalar to backpropagate and the per-image terms."""

    total: torch.Tensor
    per_sample: dict[str, torch.Tensor]


class SegmentationCriterion:
    """Weighted sum of per-sample loss terms.

    Parameters
    ----------
    terms : Sequence[str]
        Names from `LOSS_REGISTRY`, in the order they are recorded.
    weights : Mapping[str, float] | None
        Weight per term; missing terms weigh 1.
    ignore_index : int
        Target value of pixels to skip.
    """

    def __init__(
        self, terms: Sequence[str], weights: Mapping[str, float] | None = None, ignore_index: int = IGNORE_INDEX
    ) -> None:
        weights = dict(weights or {})
        unknown = sorted({*terms, *weights} - set(LOSS_REGISTRY))
        if unknown:
            raise ValueError(f"unknown loss term(s): {', '.join(unknown)}")
        for term in terms:
            contract.check_term(term)
        self.terms = list(terms)
        self.weights = {t: float(weights.get(t, 1.0)) for t in self.terms}
        self.ignore_index = ignore_index

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> LossOutput:
        """Return the scalar total and each term (plus ``total``) per sample."""
        per_sample = {t: LOSS_REGISTRY[t](logits, target, ignore_index=self.ignore_index) for t in self.terms}
        weighted = torch.stack([self.weights[t] * per_sample[t] for t in self.terms]).sum(0)
        per_sample["total"] = weighted
        return LossOutput(total=weighted.mean(), per_sample=per_sample)
