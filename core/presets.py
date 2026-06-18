from __future__ import annotations

import cv2
import numpy as np

from .models import DETECTION_PRESETS
from .detector import get_detection_params as _get_detection_params_internal


def _percentile_or(values: np.ndarray, q: float, fallback: float) -> float:
    if values.size == 0:
        return float(fallback)
    return float(np.percentile(values, q))


def recommend_detection_params_by_cell(
    cell_diameter_px: float,
    preset: str = "точный",
) -> dict:
    d = float(cell_diameter_px)
    if d <= 1.0:
        raise ValueError("Диаметр клетки должен быть больше 1 пикселя")

    p = str(preset).strip().lower()
    if p not in DETECTION_PRESETS:
        raise ValueError(f"Неизвестный пресет: {preset}")

    target_diameter = 14.0
    scale = float(np.clip(target_diameter / d, 0.65, 1.05))
    area = float(np.pi * (d * 0.5) ** 2)

    if p == "точный":
        prob_thresh = 0.15
        nms_thresh = 0.55
        cellpose_flow_threshold = 0.50
        cellpose_cellprob_threshold = -0.20
        min_area_px = int(max(8.0, round(area * 0.38)))
        max_area_px = int(round(area * 3.2))
        min_purple_ratio = 0.20
        purple_s_min = 42
        purple_v_max = 200
    else:
        prob_thresh = 0.15
        nms_thresh = 0.70
        cellpose_flow_threshold = 0.30
        cellpose_cellprob_threshold = -1.00
        min_area_px = int(max(6.0, round(area * 0.25)))
        max_area_px = int(round(area * 4.5))
        min_purple_ratio = 0.12
        purple_s_min = 30
        purple_v_max = 215

    min_area_px = int(max(0, min_area_px))
    max_area_px = int(max(0, max_area_px))
    if max_area_px > 0 and max_area_px <= min_area_px:
        max_area_px = min_area_px + 1

    params = _get_detection_params_internal()
    params.update(
        {
            "model_name": "2D_versatile_he",
            "prob_thresh": prob_thresh,
            "nms_thresh": nms_thresh,
            "scale": scale,
            "min_area_px": min_area_px,
            "max_area_px": max_area_px,
            "n_tiles_x": 3,
            "n_tiles_y": 3,
            "preprocess_mode": "rgb",
            "norm_p_low": 1.0,
            "norm_p_high": 99.8,
            "purple_filter_enabled": True,
            "purple_h_min": 120,
            "purple_h_max": 170,
            "purple_s_min": purple_s_min,
            "purple_v_max": purple_v_max,
            "min_purple_ratio": min_purple_ratio,
            "require_center_purple": True,
            "strict_purple_filter": True,
            "cellpose_model_type": "nuclei",
            "cellpose_diameter_px": float(np.clip(d, 4.0, 90.0)),
            "cellpose_flow_threshold": cellpose_flow_threshold,
            "cellpose_cellprob_threshold": cellpose_cellprob_threshold,
        }
    )
    return params


def _derive_color_thresholds_from_selection(
    image_bgr: np.ndarray,
    center: tuple[float, float],
    radius_px: float,
    preset: str,
) -> dict:
    h, w = image_bgr.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Пустое изображение для подбора цветовых параметров")

    cx = float(np.clip(float(center[0]), 0.0, float(w - 1)))
    cy = float(np.clip(float(center[1]), 0.0, float(h - 1)))
    radius = float(radius_px)
    if radius <= 1.0:
        raise ValueError("Радиус выделения должен быть больше 1 пикселя")

    yy, xx = np.ogrid[:h, :w]
    circle_mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (radius**2)
    circle_px = int(np.count_nonzero(circle_mask))
    if circle_px < 20:
        raise ValueError("Слишком маленькое выделение для подбора цвета ядра")

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h_ch = hsv[..., 0].astype(np.float32)
    s_ch = hsv[..., 1].astype(np.float32)
    v_ch = hsv[..., 2].astype(np.float32)

    s_circle = s_ch[circle_mask]
    v_circle = v_ch[circle_mask]

    sat_seed = max(20.0, _percentile_or(s_circle, 35.0, 20.0))
    val_seed = min(245.0, _percentile_or(v_circle, 85.0, 220.0))

    purple_hint_mask = h_ch >= 90.0
    color_mask = circle_mask & (s_ch >= sat_seed) & (v_ch <= val_seed) & purple_hint_mask
    if int(np.count_nonzero(color_mask)) < max(20, int(circle_px * 0.08)):
        color_mask = circle_mask & (s_ch >= sat_seed) & (v_ch <= val_seed)
    if int(np.count_nonzero(color_mask)) < max(20, int(circle_px * 0.12)):
        color_mask = circle_mask

    h_sel = h_ch[color_mask]
    s_sel = s_ch[color_mask]
    v_sel = v_ch[color_mask]

    hue_lo = _percentile_or(h_sel, 15.0, 120.0)
    hue_hi = _percentile_or(h_sel, 85.0, 170.0)
    if hue_hi < hue_lo:
        hue_lo, hue_hi = hue_hi, hue_lo

    hue_span = max(8.0, hue_hi - hue_lo)
    hue_mid = (hue_lo + hue_hi) * 0.5
    if preset == "точный":
        base_h_min = 118.0
        base_h_max = 170.0
    else:
        base_h_min = 114.0
        base_h_max = 174.0

    raw_h_min = hue_mid - hue_span * 0.8
    raw_h_max = hue_mid + hue_span * 0.8
    hue_min = int(round(np.clip(0.5 * raw_h_min + 0.5 * base_h_min, 80, 170)))
    hue_max = int(round(np.clip(0.5 * raw_h_max + 0.5 * base_h_max, 100, 179)))
    if hue_max <= hue_min:
        hue_max = min(179, hue_min + 10)
    if hue_max - hue_min < 28:
        pad = int((28 - (hue_max - hue_min)) * 0.5) + 1
        hue_min = max(80, hue_min - pad)
        hue_max = min(179, hue_max + pad)
    if preset == "точный":
        hue_min = int(np.clip(min(hue_min, 120), 70, 170))
        hue_max = int(np.clip(max(hue_max, 168), 110, 179))
    else:
        hue_min = int(np.clip(min(hue_min, 116), 70, 170))
        hue_max = int(np.clip(max(hue_max, 172), 110, 179))

    if preset == "точный":
        sat_floor = 26.0
        val_default = 200.0
    else:
        sat_floor = 18.0
        val_default = 215.0

    sat_min = int(round(np.clip(_percentile_or(s_sel, 30.0, sat_floor) * 0.80, sat_floor, 180.0)))
    val_max = int(
        round(
            np.clip(
                _percentile_or(v_sel, 90.0, val_default) + 20.0,
                130.0,
                min(235.0, val_default + 20.0),
            )
        )
    )

    color_ratio = float(np.count_nonzero(color_mask)) / float(max(1, circle_px))
    if preset == "точный":
        min_ratio = float(np.clip(color_ratio * 0.12, 0.05, 0.16))
    else:
        min_ratio = float(np.clip(color_ratio * 0.09, 0.04, 0.14))

    return {
        "purple_h_min": hue_min,
        "purple_h_max": hue_max,
        "purple_s_min": sat_min,
        "purple_v_max": val_max,
        "min_purple_ratio": min_ratio,
        "purple_filter_enabled": True,
        "require_center_purple": True,
        "strict_purple_filter": True,
    }


def recommend_detection_params_from_selection(
    image_bgr: np.ndarray,
    center: tuple[float, float],
    radius_px: float,
    preset: str = "точный",
) -> dict:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Передано пустое изображение для подбора параметров")

    diameter_px = float(radius_px) * 2.0
    params = recommend_detection_params_by_cell(diameter_px, preset=preset)
    params.update(
        _derive_color_thresholds_from_selection(
            image_bgr=image_bgr,
            center=center,
            radius_px=radius_px,
            preset=str(preset).strip().lower(),
        )
    )
    return params
