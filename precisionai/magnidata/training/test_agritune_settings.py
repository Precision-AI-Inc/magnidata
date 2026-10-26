# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the advanced AgriTune setting specs shared by the API, dashboard and trainer."""

import pytest

from precisionai.magnidata.training import agritune_settings as settings


def test_every_setting_is_well_formed():
    assert len(settings.BY_KEY) == len(settings.SETTINGS)
    for setting in settings.SETTINGS:
        assert setting.type in settings.SETTING_TYPES, setting.key
        assert setting.group, setting.key
        assert setting.label, setting.key
        assert setting.help, setting.key
        if setting.type == "choice":
            assert setting.default in setting.choices, setting.key


@pytest.mark.parametrize("setting", settings.SETTINGS, ids=lambda s: s.key)
def test_defaults_survive_the_mlflow_round_trip(setting):
    preset = setting.presets[0] if setting.presets else "agritune_mlp_probe"
    default = list(setting.default) if isinstance(setting.default, tuple) else setting.default
    if default is None:
        return
    validated = settings.validate(setting.key, default, preset)

    assert settings.from_param(setting, settings.to_param(validated)) == pytest.approx(validated)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("optimizer.name", "lion", "must be one of adamw, adam, sgd"),
        ("optimizer.momentum", 1.5, "between 0 and 0.9999"),
        ("scheduler.warmup_steps", 2.5, "whole number"),
        ("optimizer.betas", [0.9], "two numbers"),
        ("augmentation.brightness_range", [1.5, 0.5], "must not exceed"),
        ("loss.class_weights", "1,2", "list of numbers"),
        ("data.pin_memory", "yes", "true or false"),
        ("decoder.hidden_dim", 64, "only available for agritune_token_fpn"),
    ],
)
def test_invalid_values_name_the_field(key, value, message):
    with pytest.raises(ValueError, match=message):
        settings.validate(key, value, "agritune_mlp_probe")


def test_lists_and_pairs_can_be_unset():
    assert settings.validate("loss.class_weights", [], "agritune_mlp_probe") is None
    assert settings.validate("augmentation.random_crop", None, "agritune_mlp_probe") is None
    assert settings.to_param(None) == ""
    assert settings.from_param(settings.BY_KEY["decoder.hidden_dims"], "") is None


def test_lists_keep_their_element_type():
    hidden = settings.validate("decoder.hidden_dims", [256, 128], "agritune_mlp_probe")

    assert settings.from_param(settings.BY_KEY["decoder.hidden_dims"], settings.to_param(hidden)) == [256, 128]
    assert settings.to_param(True) == "true"


def test_decoder_settings_apply_only_to_their_preset():
    hidden_dims = settings.BY_KEY["decoder.hidden_dims"]

    assert settings.applies_to(hidden_dims, "agritune_mlp_probe")
    assert not settings.applies_to(hidden_dims, "agritune_token_fpn")
    assert settings.applies_to(settings.BY_KEY["optimizer.name"], "agritune_token_fpn")
