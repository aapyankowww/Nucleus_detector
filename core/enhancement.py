from __future__ import annotations

import cv2
import numpy as np

from .models import DEFAULT_ENHANCEMENT_PARAMS


# Custom H&E stain normalization reference (set by user via nucleus sample)
_custom_hes_mean: np.ndarray | None = None
_custom_hes_std: np.ndarray | None = None


def get_default_enhancement_params() -> dict:
    return dict(DEFAULT_ENHANCEMENT_PARAMS)


def normalize_enhancement_params(params: dict | None) -> dict:
    base = get_default_enhancement_params()
    if params is None:
        return base

    saturation = float(params.get("saturation", base["saturation"]))
    brightness = float(params.get("brightness", base["brightness"]))
    contrast = float(params.get("contrast", base["contrast"]))
    sharpness = float(params.get("sharpness", base["sharpness"]))
    white_balance = bool(params.get("white_balance", base["white_balance"]))
    white_balance_strength = float(params.get("white_balance_strength", base["white_balance_strength"]))
    white_balance_ref_bgr = params.get("white_balance_ref_bgr", base.get("white_balance_ref_bgr"))

    saturation = float(np.clip(saturation, 0.0, 3.0))
    brightness = float(np.clip(brightness, -100.0, 100.0))
    contrast = float(np.clip(contrast, 0.2, 3.0))
    sharpness = float(np.clip(sharpness, 0.0, 3.0))
    white_balance_strength = float(np.clip(white_balance_strength, 0.0, 1.0))

    if white_balance_ref_bgr is not None:
        if isinstance(white_balance_ref_bgr, (list, tuple)) and len(white_balance_ref_bgr) == 3:
            white_balance_ref_bgr = [float(v) for v in white_balance_ref_bgr]
        else:
            white_balance_ref_bgr = None

    return {
        "saturation": saturation,
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "white_balance": white_balance,
        "white_balance_strength": white_balance_strength,
        "white_balance_ref_bgr": white_balance_ref_bgr,
    }


def _apply_white_balance(bgr: np.ndarray, ref_bgr: tuple | list | None, strength: float) -> np.ndarray:
    if strength <= 0.0 or ref_bgr is None:
        return bgr
    ref = np.array([float(ref_bgr[0]), float(ref_bgr[1]), float(ref_bgr[2])], dtype=np.float32)
    target = ref.mean()
    scales = np.where(ref > 1e-6, target / ref, 1.0)
    scales = 1.0 + (scales - 1.0) * strength
    result = bgr.astype(np.float32) * scales.reshape(1, 1, 3)
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


def apply_image_enhancement(image_bgr: np.ndarray, params: dict | None) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Передано пустое изображение для цветокоррекции")

    cfg = normalize_enhancement_params(params)
    saturation = float(cfg["saturation"])
    brightness = float(cfg["brightness"])
    contrast = float(cfg["contrast"])
    sharpness = float(cfg["sharpness"])
    white_balance = bool(cfg["white_balance"])
    white_balance_strength = float(cfg["white_balance_strength"])
    white_balance_ref_bgr = cfg.get("white_balance_ref_bgr")

    work = image_bgr.astype(np.float32, copy=True)

    if white_balance and white_balance_strength > 0.0:
        work = _apply_white_balance(work.astype(np.uint8), white_balance_ref_bgr, white_balance_strength).astype(np.float32)

    if abs(brightness) > 1e-6:
        work += brightness
        np.clip(work, 0.0, 255.0, out=work)

    if abs(contrast - 1.0) > 1e-6:
        work = (work - 127.5) * contrast + 127.5
        np.clip(work, 0.0, 255.0, out=work)

    if abs(saturation - 1.0) > 1e-6:
        hsv = cv2.cvtColor(work.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= saturation
        np.clip(hsv[..., 1], 0.0, 255.0, out=hsv[..., 1])
        work = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    if abs(sharpness - 1.0) > 1e-6:
        blurred = cv2.GaussianBlur(work, (0, 0), sigmaX=1.2, sigmaY=1.2)
        if sharpness >= 1.0:
            amount = sharpness - 1.0
            work = cv2.addWeighted(work, 1.0 + amount, blurred, -amount, 0.0)
        else:
            work = cv2.addWeighted(work, sharpness, blurred, 1.0 - sharpness, 0.0)
        np.clip(work, 0.0, 255.0, out=work)

    return work.astype(np.uint8)


def set_custom_hes_reference(mean_lab: np.ndarray, std_lab: np.ndarray) -> None:
    global _custom_hes_mean, _custom_hes_std
    _custom_hes_mean = np.asarray(mean_lab, dtype=np.float32).copy().reshape(3)
    _custom_hes_std = np.asarray(std_lab, dtype=np.float32).copy().reshape(3)


def clear_custom_hes_reference() -> None:
    global _custom_hes_mean, _custom_hes_std
    _custom_hes_mean = None
    _custom_hes_std = None


def get_custom_hes_reference() -> tuple[np.ndarray | None, np.ndarray | None]:
    return _custom_hes_mean, _custom_hes_std


def percentile_normalize_rgb(image: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    if p_high <= p_low:
        return np.clip(image.astype(np.float32), 0.0, 1.0)

    data = image.astype(np.float32, copy=False)
    if data.ndim == 2:
        lo = np.percentile(data, p_low)
        hi = np.percentile(data, p_high)
        if hi - lo < 1e-6:
            return np.clip(data, 0.0, 1.0)
        return np.clip((data - lo) / (hi - lo), 0.0, 1.0)

    out = np.empty_like(data, dtype=np.float32)
    for c in range(data.shape[2]):
        ch = data[..., c]
        lo = np.percentile(ch, p_low)
        hi = np.percentile(ch, p_high)
        if hi - lo < 1e-6:
            out[..., c] = np.clip(ch, 0.0, 1.0)
            continue
        out[..., c] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
    return out
