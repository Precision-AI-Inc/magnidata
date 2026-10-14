# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AgriTune adapter; they need the sibling AgriTune checkout and skip without it."""

import importlib
import importlib.util
from types import ModuleType

import pytest

# Looked up without importing: importing the adapter pulls in torch, which must not happen at collection time.
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("precisionai.agritune") is None, reason="needs the sibling AgriTune checkout installed"
)


@pytest.fixture(scope="module")
def agritune() -> ModuleType:
    return importlib.import_module("precisionai.magnidata.training.agritune")


@pytest.mark.parametrize(
    ("loss", "term"),
    [("ce", "cce"), ("ce_dice", "cce"), ("dice", "cce"), ("bce", "bce"), ("bce_dice", "bce")],
)
def test_the_scaled_term_follows_the_objective(agritune, loss, term):
    assert agritune._primary_term({"agritune_loss": loss}) == term


def test_missing_objective_defaults_to_cross_entropy(agritune):
    assert agritune._primary_term({}) == "cce"


def test_every_agritune_preset_maps_to_a_decoder(agritune):
    assert set(agritune.PRESET_DECODERS) == {"agritune_mlp_probe", "agritune_token_fpn"}
