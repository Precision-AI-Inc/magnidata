# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-sample segmentation losses (CPU, tiny seeded tensors)."""

import pytest

torch = pytest.importorskip("torch")
losses = pytest.importorskip("precisionai.magnidata.training.losses")
F = torch.nn.functional

B, C, H, W = 3, 3, 8, 8


@pytest.fixture
def batch():
    gen = torch.Generator().manual_seed(0)
    target = torch.randint(0, C, (B, H, W), generator=gen)
    # every class present in every image so no class is skipped by Dice
    target[:, 0, :3] = torch.arange(C)
    logits = torch.randn(B, C, H, W, generator=gen)
    return logits, target


def _confident(target, correct=True):
    onehot = F.one_hot(target, C).permute(0, 3, 1, 2).float()
    sign = 1.0 if correct else -1.0
    return sign * 20 * onehot - sign * 10


@pytest.mark.parametrize(
    "fn",
    [
        losses.cce_per_sample,
        losses.dice_per_sample,
        losses.bce_fg_per_sample,
        lambda lg, tg: losses.iou_per_sample(lg, tg, C),
    ],
)
def test_returns_one_value_per_image(batch, fn):
    out = fn(*batch)
    assert out.shape == (B,)
    assert torch.isfinite(out).all()


def test_perfect_prediction(batch):
    _, target = batch
    logits = _confident(target)
    assert losses.cce_per_sample(logits, target).max() < 1e-4
    assert losses.dice_per_sample(logits, target).max() < 1e-3
    assert losses.bce_fg_per_sample(logits, target).max() < 1e-4
    assert torch.allclose(losses.iou_per_sample(logits, target, C), torch.ones(B))


def test_inverted_prediction(batch):
    _, target = batch
    # a different wrong class everywhere: shift each pixel's class by one
    wrong = _confident((target + 1) % C)
    assert losses.dice_per_sample(wrong, target).min() > 0.99
    assert torch.allclose(losses.iou_per_sample(wrong, target, C), torch.zeros(B))


def test_cce_mean_matches_torch_cross_entropy(batch):
    logits, target = batch
    expected = F.cross_entropy(logits, target, reduction="mean")
    assert abs(losses.cce_per_sample(logits, target).mean() - expected) < 1e-6


@pytest.mark.parametrize(
    "fn",
    [
        losses.cce_per_sample,
        losses.dice_per_sample,
        losses.bce_fg_per_sample,
        lambda lg, tg: losses.iou_per_sample(lg, tg, C),
    ],
)
def test_images_are_independent(batch, fn):
    logits, target = batch
    before = fn(logits, target)
    changed = logits.clone()
    changed[2] = torch.randn(C, H, W)
    after = fn(changed, target)
    assert torch.equal(before[:2], after[:2])
    assert not torch.equal(before[2], after[2])


def test_ignored_pixels_do_not_count(batch):
    logits, target = batch
    ignored = target.clone()
    ignored[0, :4] = losses.IGNORE_INDEX
    cropped = losses.cce_per_sample(logits[:1, :, 4:], target[:1, 4:])
    assert torch.allclose(losses.cce_per_sample(logits, ignored)[0], cropped[0], atol=1e-6)


@pytest.mark.parametrize(
    "fn",
    [
        losses.cce_per_sample,
        losses.dice_per_sample,
        losses.bce_fg_per_sample,
        lambda lg, tg: losses.iou_per_sample(lg, tg, C),
    ],
)
def test_all_ignored_image_is_finite(batch, fn):
    logits, target = batch
    target = target.clone()
    target[1] = losses.IGNORE_INDEX
    out = fn(logits, target)
    assert torch.isfinite(out).all()
    if fn is losses.cce_per_sample or fn is losses.dice_per_sample or fn is losses.bce_fg_per_sample:
        assert out[1] == 0


def test_confusion_counts_add_up_across_batches(batch):
    logits, target = batch
    target = target.clone()
    target[0, 0, 0] = losses.IGNORE_INDEX
    whole = losses.confusion_counts(logits, target, C)
    parts = losses.confusion_counts(logits[:1], target[:1], C) + losses.confusion_counts(logits[1:], target[1:], C)
    assert torch.equal(whole, parts)
    assert whole.sum() == (target != losses.IGNORE_INDEX).sum()


def test_criterion_total_is_weighted_sum_and_backpropagates(batch):
    logits, target = batch
    logits = logits.clone().requires_grad_(True)
    criterion = losses.SegmentationCriterion(["dice", "cce"], {"dice": 2.0, "cce": 0.5})
    out = criterion(logits, target)
    expected = 2.0 * losses.dice_per_sample(logits, target) + 0.5 * losses.cce_per_sample(logits, target)
    assert out.per_sample["total"].shape == (B,)
    assert torch.allclose(out.per_sample["total"], expected)
    assert torch.allclose(out.total, expected.mean())
    assert set(out.per_sample) == {"dice", "cce", "total"}
    out.total.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_criterion_rejects_unknown_terms():
    with pytest.raises(ValueError, match="unknown loss term"):
        losses.SegmentationCriterion(["dice", "focal"])
    with pytest.raises(ValueError, match="unknown loss term"):
        losses.SegmentationCriterion(["dice"], {"focal": 1.0})


def test_registry_names_match_contract():
    assert set(losses.LOSS_REGISTRY) == set(losses.contract.SEGMENTATION_LOSS_TERMS)
