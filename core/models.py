from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SUPPORTED_IMAGE_FORMATS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
DETECTOR_BACKENDS = (
    "stardist",
)
STARDIST_PRETRAINED_MODELS = (
    "2D_versatile_he",
    "2D_versatile_fluo",
    "2D_paper_dsb2018",
    "2D_demo",
)
STARDIST_PREPROCESS_MODES = (
    "rgb",
)
DETECTION_PRESETS = (
    "точный",
    "чувствительный",
)
DEFAULT_ENHANCEMENT_PARAMS = {
    "saturation": 1.0,
    "brightness": 0.0,
    "contrast": 1.0,
    "sharpness": 1.0,
}
DETECTION_RUNTIME_PRESETS = (
    "legacy",
    "balanced",
    "high_recall",
)
DETECTION_RUNTIME_PRESET_VALUES = {
    "legacy": {
        "purple_filter_enabled": True,
        "strict_purple_filter": True,
        "require_center_purple": True,
        "min_purple_ratio": 0.20,
        "purple_s_min": 45,
        "purple_v_max": 200,
        "prob_thresh": 0.15,
        "nms_thresh": 0.55,
        "min_area_px": 18,
        "scale": 1.0,
    },
    "balanced": {
        "purple_filter_enabled": True,
        "strict_purple_filter": False,
        "require_center_purple": False,
        "min_purple_ratio": 0.10,
        "purple_s_min": 30,
        "purple_v_max": 255,
        "prob_thresh": 0.10,
        "nms_thresh": 0.40,
        "min_area_px": 10,
        "scale": 1.0,
    },
    "high_recall": {
        "purple_filter_enabled": False,
        "strict_purple_filter": False,
        "require_center_purple": False,
        "min_purple_ratio": 0.05,
        "purple_s_min": 15,
        "purple_v_max": 255,
        "prob_thresh": 0.08,
        "nms_thresh": 0.35,
        "min_area_px": 0,
        "scale": 1.0,
    },
}


class ModelValidationError(ValueError):
    """Raised when model path/runtime is invalid."""


@dataclass
class LoadedModel:
    path: str
    model_format: str
    runtime: str
    model: object
    input_name: str | None = None
    output_name: str | None = None


@dataclass
class StarDistConfig:
    detector_backend: str = "stardist"
    model_name: str = "2D_versatile_he"
    prob_thresh: float = 0.15
    nms_thresh: float = 0.29
    scale: float = 1.0
    min_area_px: int = 10
    max_area_px: int = 0
    n_tiles_x: int = 3
    n_tiles_y: int = 3
    preprocess_mode: str = "rgb"
    norm_p_low: float = 1.0
    norm_p_high: float = 99.8
    purple_filter_enabled: bool = True
    purple_h_min: int = 118
    purple_h_max: int = 172
    purple_s_min: int = 30
    purple_v_max: int = 215
    min_purple_ratio: float = 0.12
    require_center_purple: bool = True
    strict_purple_filter: bool = True
    upscale_factor: float = 2.05
    stain_norm_enabled: bool = True
    cellpose_model_type: str = "nuclei"
    cellpose_diameter_px: float = 14.0
    cellpose_flow_threshold: float = 0.40
    cellpose_cellprob_threshold: float = -0.50


_HE_REF_MEAN_LAB = np.array([173.0, 140.0, 123.0], dtype=np.float32)
_HE_REF_STD_LAB = np.array([46.0, 12.0, 10.0], dtype=np.float32)


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return bool(value)


def normalize_runtime_preset_name(preset_name: str) -> str:
    key = str(preset_name).strip().lower()
    if key not in DETECTION_RUNTIME_PRESETS:
        raise ValueError(
            f"Неизвестный runtime-пресет: {preset_name}. "
            f"Доступно: {', '.join(DETECTION_RUNTIME_PRESETS)}"
        )
    return key


def get_runtime_preset_values(preset_name: str) -> dict:
    key = normalize_runtime_preset_name(preset_name)
    values = DETECTION_RUNTIME_PRESET_VALUES.get(key, {})
    return dict(values)
