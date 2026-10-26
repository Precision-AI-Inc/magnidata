# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Advanced AgriTune training settings, declared once for the API, the Train dialog and the trainer.

Each :class:`Setting` names an MLflow parameter (``key``), where its value lands in AgriTune's
``TrainingRunConfig`` (``target``), and how it is validated and shown. The API validates launch requests
against these specs and serves them to the dashboard, which renders the *Advanced* section from them; the
AgriTune adapter applies each run's values by ``target``. Torch-free, so the API can import it.
"""

from dataclasses import asdict, dataclass
from typing import Any

# Value types: scalars, a choice, a list ("ints"/"floats", empty = unset) or an optional pair ("range"/"size").
SETTING_TYPES = ("int", "float", "bool", "choice", "ints", "floats", "range", "size")


@dataclass(frozen=True)
class Setting:
    """One advanced AgriTune setting.

    Attributes
    ----------
    key : str
        MLflow parameter name, also the request field (``optimizer.name``).
    target : str
        Dotted path in AgriTune's ``TrainingRunConfig`` the value is written to (``optimizer.name``,
        ``decoder_kwargs.dropout``).
    group : str
        Dialog section.
    label : str
        Dialog label.
    type : str
        One of :data:`SETTING_TYPES`.
    default : Any
        Value used when the run does not set it (``None`` = AgriTune's own default or unset).
    help : str
        One-line explanation shown under the field.
    minimum, maximum : float | None
        Bounds for numbers and list or pair elements.
    choices : tuple[str, ...]
        Allowed values of a choice.
    presets : tuple[str, ...]
        Presets the setting applies to; empty means every AgriTune preset.
    zero_is_none : bool
        For numbers, ``0`` means "off" and becomes ``None`` in AgriTune.
    """

    key: str
    target: str
    group: str
    label: str
    type: str
    default: Any
    help: str
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()
    presets: tuple[str, ...] = ()
    zero_is_none: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON form served to the dashboard."""
        out = asdict(self)
        out["choices"] = list(self.choices)
        out["presets"] = list(self.presets)
        return out


_MLP = ("agritune_mlp_probe",)
_FPN = ("agritune_token_fpn",)
_ASPP = ("agritune_aspp",)
_PPM = ("agritune_ppm",)
_SEGMENTER = ("agritune_segmenter",)
_MASK_FORMER = ("agritune_mask_former",)
_AUG = "Applies when Image augmentation is not None."

SETTINGS: tuple[Setting, ...] = (
    # Optimizer
    Setting(
        "optimizer.name",
        "optimizer.name",
        "Optimizer",
        "Optimizer",
        "choice",
        "adamw",
        "AdamW, Adam or SGD; learning rate and weight decay are set above.",
        choices=("adamw", "adam", "sgd"),
    ),
    Setting(
        "optimizer.betas",
        "optimizer.betas",
        "Optimizer",
        "Adam betas",
        "range",
        (0.9, 0.999),
        "First and second moment decay of Adam and AdamW.",
        0.0,
        0.9999,
    ),
    Setting(
        "optimizer.momentum",
        "optimizer.momentum",
        "Optimizer",
        "SGD momentum",
        "float",
        0.9,
        "Used by SGD only.",
        0.0,
        0.9999,
    ),
    # Scheduler
    Setting(
        "scheduler.name",
        "scheduler.name",
        "Learning-rate schedule",
        "Schedule",
        "choice",
        "constant",
        "Constant keeps the learning rate fixed.",
        choices=("constant", "cosine", "linear_warmup", "polynomial", "plateau"),
    ),
    Setting(
        "scheduler.warmup_steps",
        "scheduler.warmup_steps",
        "Learning-rate schedule",
        "Warm-up steps",
        "int",
        0,
        "Optimizer steps of linear warm-up (cosine, linear warm-up, polynomial).",
        0,
        1_000_000,
    ),
    Setting(
        "scheduler.total_steps",
        "scheduler.total_steps",
        "Learning-rate schedule",
        "Total steps",
        "int",
        0,
        "Length of the cosine, linear warm-up or polynomial schedule; 0 uses every optimizer step of the run.",
        0,
        100_000_000,
    ),
    Setting(
        "scheduler.polynomial_power",
        "scheduler.polynomial_power",
        "Learning-rate schedule",
        "Polynomial power",
        "float",
        1.0,
        "Decay exponent of the polynomial schedule.",
        0.0,
        10.0,
    ),
    Setting(
        "scheduler.plateau_factor",
        "scheduler.plateau_factor",
        "Learning-rate schedule",
        "Plateau factor",
        "float",
        0.5,
        "Multiplier applied when the validation metric stops improving.",
        0.0,
        1.0,
    ),
    Setting(
        "scheduler.plateau_patience",
        "scheduler.plateau_patience",
        "Learning-rate schedule",
        "Plateau patience",
        "int",
        10,
        "Validation passes without improvement before the plateau schedule lowers the rate.",
        0,
        1000,
    ),
    # Trainer
    Setting(
        "trainer.precision",
        "trainer.precision.mode",
        "Training",
        "Precision",
        "choice",
        "auto",
        "Auto follows Mixed precision (fp16 on a GPU, else fp32); bf16 needs a GPU that supports it.",
        choices=("auto", "fp32", "fp16", "bf16"),
    ),
    Setting(
        "trainer.early_stopping_patience",
        "trainer.early_stopping_patience",
        "Training",
        "Early-stopping patience",
        "int",
        0,
        "Stop after this many validation passes without improvement; 0 never stops early.",
        0,
        1000,
        zero_is_none=True,
    ),
    Setting(
        "trainer.checkpoint_every_n_steps",
        "trainer.checkpoint_every_n_steps",
        "Training",
        "Checkpoint every N steps",
        "int",
        0,
        "Extra mid-epoch checkpoints; 0 saves only at epoch ends.",
        0,
        10_000_000,
        zero_is_none=True,
    ),
    Setting(
        "val_metric_name",
        "val_metric_name",
        "Training",
        "Best-checkpoint metric",
        "choice",
        "mean_iou",
        "Validation metric that picks the best checkpoint; loss is minimized, the others maximized.",
        choices=("mean_iou", "mean_dice", "pixel_accuracy", "loss"),
    ),
    Setting(
        "checkpoint_top_k",
        "checkpoint_top_k",
        "Training",
        "Best checkpoints kept",
        "int",
        3,
        "How many best-by-metric checkpoints to keep; 0 keeps every one.",
        0,
        100,
    ),
    Setting(
        "loss.class_weights",
        "loss.class_weights",
        "Training",
        "Class weights",
        "floats",
        None,
        "One weight per class, background first, comma-separated; empty weighs every class equally.",
        0.0,
        1000.0,
    ),
    # Decoder
    Setting(
        "decoder.hidden_dims",
        "decoder_kwargs.hidden_dims",
        "Decoder",
        "Hidden layers",
        "ints",
        None,
        "Widths of hidden layers, comma-separated; empty is a linear probe.",
        1,
        8192,
        presets=_MLP,
    ),
    Setting(
        "decoder.hidden_dim",
        "decoder_kwargs.hidden_dim",
        "Decoder",
        "Hidden width",
        "int",
        None,
        "Channels inside the decoder; empty keeps its default (192 for Segmenter, 128 for the others).",
        1,
        4096,
        presets=_FPN + _ASPP + _PPM + _SEGMENTER + _MASK_FORMER,
    ),
    Setting(
        "decoder.num_layers",
        "decoder_kwargs.num_layers",
        "Decoder",
        "Layers",
        "int",
        None,
        "Pyramid levels (TokenFPN), refinement convolutions (ASPP, Pyramid Pooling) or transformer layers "
        "(Segmenter, MaskFormer); empty keeps the default (1 for ASPP and Pyramid Pooling, else 2).",
        1,
        24,
        presets=_FPN + _ASPP + _PPM + _SEGMENTER + _MASK_FORMER,
    ),
    Setting(
        "decoder.dropout",
        "decoder_kwargs.dropout",
        "Decoder",
        "Dropout",
        "float",
        0.0,
        "Dropout inside the decoder.",
        0.0,
        0.95,
        presets=_MLP + _SEGMENTER + _MASK_FORMER,
    ),
    Setting(
        "decoder.cls_fusion",
        "decoder_kwargs.cls_fusion",
        "Decoder",
        "CLS-token fusion",
        "choice",
        "none",
        "How the image-level CLS token conditions the patch features.",
        choices=("none", "concat", "film"),
        presets=_FPN,
    ),
    Setting(
        "decoder.atrous_rates",
        "decoder_kwargs.atrous_rates",
        "Decoder",
        "Atrous rates",
        "ints",
        None,
        "Dilation rates of the parallel branches, comma-separated; empty keeps 6, 12, 18.",
        1,
        64,
        presets=_ASPP,
    ),
    Setting(
        "decoder.pool_sizes",
        "decoder_kwargs.pool_sizes",
        "Decoder",
        "Pool sizes",
        "ints",
        None,
        "Output grid of each pooling branch, comma-separated; empty keeps 1, 2, 3, 6.",
        1,
        64,
        presets=_PPM,
    ),
    Setting(
        "decoder.num_heads",
        "decoder_kwargs.num_heads",
        "Decoder",
        "Attention heads",
        "int",
        None,
        "Must divide the hidden width; empty keeps the default (3 for Segmenter, 4 for MaskFormer).",
        1,
        64,
        presets=_SEGMENTER + _MASK_FORMER,
    ),
    Setting(
        "decoder.mlp_ratio",
        "decoder_kwargs.mlp_ratio",
        "Decoder",
        "MLP ratio",
        "int",
        None,
        "Width of each transformer feed-forward block relative to the hidden width; empty keeps 4.",
        1,
        16,
        presets=_SEGMENTER + _MASK_FORMER,
    ),
    Setting(
        "decoder.num_queries",
        "decoder_kwargs.num_queries",
        "Decoder",
        "Mask queries",
        "int",
        None,
        "Learned mask queries; empty keeps 32.",
        1,
        1024,
        presets=_MASK_FORMER,
    ),
    # Image augmentation
    Setting(
        "augmentation.variant",
        "augmentation.variant",
        "Image augmentation",
        "Offline variant",
        "int",
        0,
        "Which fixed augmented view to encode under offline augmentation. " + _AUG,
        0,
        1000,
    ),
    Setting(
        "augmentation.hybrid_online_probability",
        "augmentation.hybrid_online_probability",
        "Image augmentation",
        "Hybrid online probability",
        "float",
        0.3,
        "Share of fresh online views under hybrid augmentation. " + _AUG,
        0.0,
        1.0,
    ),
    Setting(
        "augmentation.random_crop",
        "augmentation.geometric.random_crop",
        "Image augmentation",
        "Random crop",
        "size",
        None,
        "Width,height of a random crop; empty disables it. " + _AUG,
        1,
        8192,
    ),
    Setting(
        "augmentation.horizontal_flip_probability",
        "augmentation.geometric.horizontal_flip_probability",
        "Image augmentation",
        "Horizontal flip",
        "float",
        0.5,
        "Probability of a left-right flip. " + _AUG,
        0.0,
        1.0,
    ),
    Setting(
        "augmentation.vertical_flip_probability",
        "augmentation.geometric.vertical_flip_probability",
        "Image augmentation",
        "Vertical flip",
        "float",
        0.0,
        "Probability of an up-down flip. " + _AUG,
        0.0,
        1.0,
    ),
    Setting(
        "augmentation.rotation_max_degrees",
        "augmentation.geometric.rotation_max_degrees",
        "Image augmentation",
        "Rotation",
        "float",
        0.0,
        "Largest random rotation in degrees. " + _AUG,
        0.0,
        180.0,
    ),
    Setting(
        "augmentation.brightness_range",
        "augmentation.photometric.brightness_range",
        "Image augmentation",
        "Brightness",
        "range",
        None,
        "Min,max brightness factor; empty disables it. " + _AUG,
        0.0,
        5.0,
    ),
    Setting(
        "augmentation.contrast_range",
        "augmentation.photometric.contrast_range",
        "Image augmentation",
        "Contrast",
        "range",
        None,
        "Min,max contrast factor; empty disables it. " + _AUG,
        0.0,
        5.0,
    ),
    Setting(
        "augmentation.saturation_range",
        "augmentation.photometric.saturation_range",
        "Image augmentation",
        "Saturation",
        "range",
        None,
        "Min,max saturation factor; empty disables it. " + _AUG,
        0.0,
        5.0,
    ),
    Setting(
        "augmentation.hue_max_degrees",
        "augmentation.photometric.hue_max_degrees",
        "Image augmentation",
        "Hue shift",
        "float",
        0.0,
        "Largest hue shift in degrees. " + _AUG,
        0.0,
        180.0,
    ),
    Setting(
        "augmentation.blur_probability",
        "augmentation.photometric.blur_probability",
        "Image augmentation",
        "Blur probability",
        "float",
        0.0,
        _AUG,
        0.0,
        1.0,
    ),
    Setting(
        "augmentation.blur_radius_range",
        "augmentation.photometric.blur_radius_range",
        "Image augmentation",
        "Blur radius",
        "range",
        (0.1, 2.0),
        "Min,max Gaussian blur radius in pixels. " + _AUG,
        0.0,
        50.0,
    ),
    Setting(
        "augmentation.noise_probability",
        "augmentation.photometric.noise_probability",
        "Image augmentation",
        "Noise probability",
        "float",
        0.0,
        _AUG,
        0.0,
        1.0,
    ),
    Setting(
        "augmentation.noise_std_range",
        "augmentation.photometric.noise_std_range",
        "Image augmentation",
        "Noise strength",
        "range",
        (0.01, 0.05),
        "Min,max Gaussian noise standard deviation. " + _AUG,
        0.0,
        1.0,
    ),
    Setting(
        "augmentation.channel_dropout_probability",
        "augmentation.photometric.channel_dropout_probability",
        "Image augmentation",
        "Colour-channel dropout",
        "float",
        0.0,
        _AUG,
        0.0,
        1.0,
    ),
    Setting(
        "augmentation.gsd_jitter_probability",
        "augmentation.photometric.gsd_jitter_probability",
        "Image augmentation",
        "Ground-sampling jitter",
        "float",
        0.0,
        "Probability of simulating a coarser ground sampling distance. " + _AUG,
        0.0,
        1.0,
    ),
    Setting(
        "augmentation.gsd_jitter_scale_range",
        "augmentation.photometric.gsd_jitter_scale_range",
        "Image augmentation",
        "Ground-sampling scale",
        "range",
        (0.5, 0.9),
        "Min,max downscale factor. " + _AUG,
        0.01,
        1.0,
    ),
    Setting(
        "augmentation.vegetation_index_jitter_probability",
        "augmentation.photometric.vegetation_index_jitter_probability",
        "Image augmentation",
        "Vegetation-index jitter",
        "float",
        0.0,
        _AUG,
        0.0,
        1.0,
    ),
    Setting(
        "augmentation.vegetation_index_jitter_range",
        "augmentation.photometric.vegetation_index_jitter_range",
        "Image augmentation",
        "Vegetation-index range",
        "range",
        (-15.0, 15.0),
        "Min,max shift. " + _AUG,
        -100.0,
        100.0,
    ),
    Setting(
        "augmentation.seasonal_color_shift_probability",
        "augmentation.photometric.seasonal_color_shift_probability",
        "Image augmentation",
        "Seasonal colour shift",
        "float",
        0.0,
        _AUG,
        0.0,
        1.0,
    ),
    Setting(
        "augmentation.seasonal_hue_shift_degrees",
        "augmentation.photometric.seasonal_hue_shift_degrees",
        "Image augmentation",
        "Seasonal hue shift",
        "float",
        0.0,
        "Hue shift in degrees. " + _AUG,
        0.0,
        180.0,
    ),
    # Feature augmentation
    Setting(
        "feature_augmentation.patch_dropout_probability",
        "feature_augmentation.patch_dropout_probability",
        "Feature augmentation",
        "Patch dropout",
        "float",
        0.0,
        "Drop each patch token with this probability.",
        0.0,
        0.99,
    ),
    Setting(
        "feature_augmentation.token_masking_probability",
        "feature_augmentation.token_masking_probability",
        "Feature augmentation",
        "Token masking",
        "float",
        0.0,
        "Replace each patch token with the mask value.",
        0.0,
        1.0,
    ),
    Setting(
        "feature_augmentation.token_mask_value",
        "feature_augmentation.token_mask_value",
        "Feature augmentation",
        "Mask value",
        "float",
        0.0,
        "Value masked tokens are set to.",
        -1000.0,
        1000.0,
    ),
    Setting(
        "feature_augmentation.gaussian_noise_std",
        "feature_augmentation.gaussian_noise_std",
        "Feature augmentation",
        "Feature noise",
        "float",
        0.0,
        "Gaussian noise added to every token.",
        0.0,
        100.0,
    ),
    Setting(
        "feature_augmentation.cls_dropout_probability",
        "feature_augmentation.cls_dropout_probability",
        "Feature augmentation",
        "CLS dropout",
        "float",
        0.0,
        "Drop each sample's CLS token.",
        0.0,
        0.99,
    ),
    Setting(
        "feature_augmentation.channel_dropout_probability",
        "feature_augmentation.channel_dropout_probability",
        "Feature augmentation",
        "Channel dropout",
        "float",
        0.0,
        "Drop each embedding channel.",
        0.0,
        0.99,
    ),
    # Data loading
    Setting(
        "data.num_workers",
        "num_workers",
        "Data loading",
        "Loader processes",
        "int",
        0,
        "Processes decoding images and masks (online or hybrid features); 0 loads in the trainer.",
        0,
        64,
    ),
    Setting(
        "data.pin_memory",
        "pin_memory",
        "Data loading",
        "Pin host memory",
        "bool",
        False,
        "Faster copies to the GPU at the cost of locked RAM.",
    ),
    Setting(
        "data.prefetch_factor",
        "prefetch_factor",
        "Data loading",
        "Prefetch batches",
        "int",
        0,
        "Batches each loader process prepares ahead; 0 uses PyTorch's default.",
        0,
        1024,
        zero_is_none=True,
    ),
    Setting(
        "data.feature_read_workers",
        "feature_read_workers",
        "Data loading",
        "Feature read threads",
        "int",
        32,
        "Concurrent feature-store reads per batch when features are not preloaded.",
        1,
        1024,
    ),
    Setting(
        "data.feature_preload_dtype",
        "feature_preload_dtype",
        "Data loading",
        "Preload precision",
        "choice",
        "float16",
        "Precision of preloaded features: float16 halves memory, float32 keeps the store's values.",
        choices=("float16", "bfloat16", "float32"),
    ),
    Setting(
        "data.shuffle",
        "shuffle",
        "Data loading",
        "Shuffle every epoch",
        "choice",
        "auto",
        "Auto shuffles whenever features are preloaded (AgriTune requires preload for shuffling).",
        choices=("auto", "true", "false"),
    ),
)

BY_KEY: dict[str, Setting] = {setting.key: setting for setting in SETTINGS}


def _number(setting: Setting, value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field}: must be a number")
    if setting.type in ("int", "ints", "size") and not float(value).is_integer():
        raise ValueError(f"{field}: must be a whole number")
    low = setting.minimum if setting.minimum is not None else float("-inf")
    high = setting.maximum if setting.maximum is not None else float("inf")
    if not low <= value <= high:
        raise ValueError(f"{field}: must be between {low:g} and {high:g}")
    return int(value) if setting.type in ("int", "ints", "size") else float(value)


def validate(key: str, value: Any, preset: str) -> Any:
    """Return ``value`` checked against the setting ``key`` for ``preset``; raise ``ValueError`` naming the field.

    Lists and pairs come as JSON arrays; an empty array or ``None`` unsets them.
    """
    setting = BY_KEY[key]
    field = f"params.{key}"
    if setting.presets and preset not in setting.presets:
        raise ValueError(f"{field}: only available for {', '.join(setting.presets)}")
    if setting.type == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{field}: must be true or false")
        return value
    if setting.type == "choice":
        if value not in setting.choices:
            raise ValueError(f"{field}: must be one of {', '.join(setting.choices)}")
        return value
    if setting.type in ("int", "float"):
        return _number(setting, value, field)
    return _numbers(setting, value, field)


def _numbers(setting: Setting, value: Any, field: str) -> list[float] | None:
    if value is None or value == []:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{field}: must be a list of numbers")
    if setting.type in ("range", "size") and len(value) != 2:
        raise ValueError(f"{field}: must be two numbers")
    items = [_number(setting, item, field) for item in value]
    if setting.type == "range" and items[0] > items[1]:
        raise ValueError(f"{field}: the first number must not exceed the second")
    return items


def to_param(value: Any) -> str:
    """Return the MLflow parameter string of a validated value (lists comma-joined, unset as empty)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list | tuple):
        return ",".join(str(item) for item in value)
    return str(value)


def from_param(setting: Setting, text: str) -> Any:
    """Return the typed value of an MLflow parameter string, or ``None`` for an unset list or pair."""
    if setting.type == "bool":
        return text.lower() == "true"
    if setting.type == "choice":
        return text
    if setting.type == "int":
        return int(float(text))
    if setting.type == "float":
        return float(text)
    if not text.strip():
        return None
    numbers = [float(item) for item in text.split(",")]
    return [int(item) for item in numbers] if setting.type in ("ints", "size") else numbers


def applies_to(setting: Setting, preset: str) -> bool:
    """Return whether ``setting`` is offered for ``preset``."""
    return not setting.presets or preset in setting.presets
