# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Training presets and validation of launch requests.

Users pick a preset and may tune only whitelisted parameters within bounds; anything else
is rejected rather than ignored, and nothing from a request is ever executed.
"""

import re
from dataclasses import dataclass
from typing import Any

from precisionai.magnidata.training import contract

RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")
SPLIT_MODES = ("random", "column")
MAX_NAME_LENGTH = 100
MAX_SOURCE_LENGTH = 500

BOUNDS: dict[str, dict[str, Any]] = {
    "epochs": {"type": "int", "min": 1, "max": 200},
    "score_every": {"type": "int", "min": 1, "max": "epochs"},
    "batch_size": {"type": "int", "min": 1, "max": 64},
    "image_size": {"type": "int", "min": 128, "max": 1024, "multiple_of": 32},
    "lr": {"type": "float", "min": 1e-6, "max": 1e-1},
    "val_fraction": {"type": "float", "min": 0.05, "max": 0.5},
    "seed": {"type": "int", "min": 0, "max": 2**31 - 1},
    "split_mode": {"type": "choice", "choices": list(SPLIT_MODES)},
    "loss_terms": {"type": "list", "choices": list(contract.SEGMENTATION_LOSS_TERMS), "min_items": 1},
    "loss_weights": {"type": "map", "min": 0.0, "max": 10.0},
    "top_k_predictions": {"type": "int", "min": 0, "max": 200},
    "pretrained": {"type": "bool"},
    "amp": {"type": "bool"},
    "num_workers": {"type": "int", "min": 0, "max": 16},
    "feature_provider": {"type": "choice", "choices": ["hybrid", "online"]},
    "augmentation_mode": {"type": "choice", "choices": ["none", "offline", "online", "hybrid"]},
    "agritune_loss": {"type": "choice", "choices": ["ce", "bce", "dice", "ce_dice", "bce_dice"]},
}

AGRITUNE_FIELDS = ("feature_provider", "augmentation_mode", "agritune_loss")
AGRITUNE_LOSS_TERMS: dict[str, list[str]] = {
    "ce": ["cce"],
    "bce": ["bce"],
    "dice": ["dice"],
    "ce_dice": ["cce", "dice"],
    "bce_dice": ["bce", "dice"],
}
# Settings of the reference trainer that AgriTune presets cannot honour; rejected rather than silently ignored.
REFERENCE_ONLY_FIELDS = ("loss_terms", "num_workers")

COMMON_DEFAULTS: dict[str, Any] = {
    "score_every": 1,
    "val_fraction": 0.2,
    "seed": 0,
    "split_mode": "random",
    "loss_terms": ["dice", "cce"],
    "loss_weights": {},
    "top_k_predictions": 50,
    "amp": True,
}

PRESETS: dict[str, dict[str, Any]] = {
    "tiny_unet": {
        "engine": "reference",
        "label": "Tiny UNet",
        "description": "Small U-Net trained from scratch. CPU-friendly; good for smoke tests.",
        "defaults": {
            **COMMON_DEFAULTS,
            "epochs": 10,
            "batch_size": 8,
            "image_size": 128,
            "lr": 1e-3,
            "pretrained": False,
        },
    },
    "lraspp_mobilenet": {
        "engine": "reference",
        "label": "LR-ASPP MobileNetV3",
        "description": "MobileNetV3 backbone with an LR-ASPP head, ImageNet-pretrained. GPU recommended.",
        "defaults": {
            **COMMON_DEFAULTS,
            "epochs": 20,
            "batch_size": 8,
            "image_size": 256,
            "lr": 3e-4,
            "pretrained": True,
        },
    },
    "agritune_mlp_probe": {
        "engine": "agritune",
        "label": "AgriTune MLP Probe",
        "description": "Fast decoder trained on frozen encoder patch features. Works offline with the fake encoder.",
        "defaults": {
            **COMMON_DEFAULTS,
            "epochs": 10,
            "batch_size": 4,
            "image_size": 256,
            "lr": 3e-4,
            "pretrained": False,
            "feature_provider": "hybrid",
            "augmentation_mode": "none",
            "agritune_loss": "ce_dice",
        },
    },
    "agritune_token_fpn": {
        "engine": "agritune",
        "label": "AgriTune TokenFPN",
        "description": "Multi-scale decoder over frozen ViT features. A hosted encoder and GPU are recommended.",
        "defaults": {
            **COMMON_DEFAULTS,
            "epochs": 20,
            "batch_size": 4,
            "image_size": 256,
            "lr": 3e-4,
            "pretrained": False,
            "feature_provider": "hybrid",
            "augmentation_mode": "none",
            "agritune_loss": "ce_dice",
        },
    },
}


@dataclass(frozen=True)
class LaunchSpec:
    """A validated launch request; ``params`` are the complete settings as MLflow param strings."""

    name: str
    dataset_source: str
    preset: str
    engine: str
    params: dict[str, str]
    loss_terms: list[str]
    parent_run_id: str | None = None


def describe() -> dict[str, Any]:
    """Return the registry as served to the dashboard."""
    return {
        "presets": [{"key": key, **value} for key, value in PRESETS.items()],
        "bounds": BOUNDS,
        "loss_terms": list(contract.SEGMENTATION_LOSS_TERMS),
    }


def _integer(field: str, value: Any, low: int, high: int, multiple_of: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field}: must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{field}: must be between {low} and {high}")
    if value % multiple_of:
        raise ValueError(f"{field}: must be a multiple of {multiple_of}")
    return value


def _number(field: str, value: Any, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field}: must be a number")
    if not low <= value <= high:
        raise ValueError(f"{field}: must be between {low:g} and {high:g}")
    return float(value)


def _loss_terms(value: Any) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(t, str) for t in value):
        raise ValueError("params.loss_terms: must be a non-empty list of names")
    unknown = sorted(set(value) - set(contract.SEGMENTATION_LOSS_TERMS))
    if unknown:
        raise ValueError(f"params.loss_terms: unknown loss term(s) {', '.join(unknown)}")
    if len(set(value)) != len(value):
        raise ValueError("params.loss_terms: duplicate loss term")
    return list(value)


def _loss_weights(value: Any, terms: list[str]) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("params.loss_weights: must be an object of term → weight")
    out = {}
    for term, weight in value.items():
        if term not in terms:
            raise ValueError(f"params.loss_weights.{term}: not one of the selected loss terms")
        out[term] = _number(f"params.loss_weights.{term}", weight, 0.0, 10.0)
    return out


def _check_engine_fields(engine: str, overrides: dict[str, Any]) -> None:
    """Reject overrides that the preset's engine would ignore."""
    if engine == "reference":
        for field in AGRITUNE_FIELDS:
            if field in overrides:
                raise ValueError(f"params.{field}: only available for AgriTune presets")
        return
    if "loss_terms" in overrides:
        raise ValueError("params.loss_terms: AgriTune presets take an agritune_loss instead")
    if "num_workers" in overrides:
        raise ValueError("params.num_workers: not used by AgriTune presets")
    if overrides.get("split_mode", "random") != "random":
        raise ValueError("params.split_mode: AgriTune presets split randomly by validation fraction")
    if overrides.get("pretrained", False) is not False:
        raise ValueError("params.pretrained: AgriTune presets use a frozen encoder, not ImageNet weights")


def _resolve(preset: str, overrides: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(overrides) - set(BOUNDS))
    if unknown:
        raise ValueError(f"params: unknown parameter(s) {', '.join(unknown)}")
    engine = PRESETS[preset]["engine"]
    _check_engine_fields(engine, overrides)
    p = {**PRESETS[preset]["defaults"], **overrides}
    choices: dict[str, str] = {}
    for field in AGRITUNE_FIELDS:
        if field in p:
            allowed = BOUNDS[field]["choices"]
            if p[field] not in allowed:
                raise ValueError(f"params.{field}: must be one of {', '.join(allowed)}")
            choices[field] = p[field]
    # An AgriTune objective fixes which loss terms exist, so weights are checked against those, not the reference defaults.
    terms = AGRITUNE_LOSS_TERMS[choices["agritune_loss"]] if engine == "agritune" else _loss_terms(p["loss_terms"])
    out: dict[str, Any] = {
        "epochs": _integer("params.epochs", p["epochs"], 1, 200),
        "batch_size": _integer("params.batch_size", p["batch_size"], 1, 64),
        "image_size": _integer("params.image_size", p["image_size"], 128, 1024, 32),
        "lr": _number("params.lr", p["lr"], 1e-6, 1e-1),
        "val_fraction": _number("params.val_fraction", p["val_fraction"], 0.05, 0.5),
        "seed": _integer("params.seed", p["seed"], 0, 2**31 - 1),
        "top_k_predictions": _integer("params.top_k_predictions", p["top_k_predictions"], 0, 200),
        "loss_terms": terms,
    }
    out["score_every"] = _integer("params.score_every", p["score_every"], 1, out["epochs"])
    out["loss_weights"] = _loss_weights(p["loss_weights"], out["loss_terms"])
    if p["split_mode"] not in SPLIT_MODES:
        raise ValueError(f"params.split_mode: must be one of {', '.join(SPLIT_MODES)}")
    out["split_mode"] = p["split_mode"]
    if not isinstance(p["pretrained"], bool):
        raise ValueError("params.pretrained: must be true or false")
    out["pretrained"] = p["pretrained"]
    if not isinstance(p["amp"], bool):
        raise ValueError("params.amp: must be true or false")
    out["amp"] = p["amp"]
    if "num_workers" in p:
        out["num_workers"] = _integer("params.num_workers", p["num_workers"], 0, 16)
    out.update(choices)
    return out


def _flat(resolved: dict[str, Any]) -> dict[str, str]:
    params = {}
    for key, value in resolved.items():
        if key == "loss_terms":
            params[key] = ",".join(value)
        elif key == "loss_weights":
            params.update({f"weight_{term}": str(weight) for term, weight in value.items()})
        elif isinstance(value, bool):
            params[key] = str(value).lower()
        else:
            params[key] = str(value)
    return params


def validate_launch(body: Any) -> LaunchSpec:
    """Validate a launch request body.

    Raises
    ------
    ValueError
        Naming the offending field, for unknown keys, unknown presets and out-of-bounds values.
    """
    if not isinstance(body, dict):
        raise ValueError("body: must be a JSON object")
    unknown = sorted(set(body) - {"name", "dataset_source", "preset", "params", "parent_run_id"})
    if unknown:
        raise ValueError(f"unknown field(s): {', '.join(unknown)}")

    source = body.get("dataset_source")
    if not isinstance(source, str) or not source or len(source) > MAX_SOURCE_LENGTH:
        raise ValueError("dataset_source: required")
    preset = body.get("preset")
    if preset not in PRESETS:
        raise ValueError(f"preset: must be one of {', '.join(PRESETS)}")
    overrides = body.get("params", {})
    if not isinstance(overrides, dict):
        raise ValueError("params: must be an object")
    name = body.get("name") or f"{preset} on {source.rsplit('/', 1)[-1].removesuffix('.csv')}"
    if not isinstance(name, str) or len(name) > MAX_NAME_LENGTH:
        raise ValueError(f"name: must be text of at most {MAX_NAME_LENGTH} characters")
    parent = body.get("parent_run_id")
    if parent is not None and (not isinstance(parent, str) or not RUN_ID_RE.match(parent)):
        raise ValueError("parent_run_id: must be a run id")

    engine = PRESETS[preset]["engine"]
    resolved = _resolve(preset, overrides)
    return LaunchSpec(
        name, source, preset, engine, _flat({"preset": preset, **resolved}), resolved["loss_terms"], parent
    )
