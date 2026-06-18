from __future__ import annotations

import contextlib
import importlib.util
import io
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .enhancement import apply_image_enhancement, percentile_normalize_rgb, get_custom_hes_reference
from .models import (
    DETECTOR_BACKENDS,
    STARDIST_PREPROCESS_MODES,
    STARDIST_PRETRAINED_MODELS,
    SUPPORTED_IMAGE_FORMATS,
    LoadedModel,
    ModelValidationError,
    StarDistConfig,
    _HE_REF_MEAN_LAB,
    _HE_REF_STD_LAB,
    _to_bool,
    get_runtime_preset_values,
)
from .roi import (
    _normalize_poly_roi,
    _normalize_rect_roi,
    _point_in_any_roi,
    _point_in_roi,
    _roi_area_in_cells,
    _roi_to_infer_space,
    normalize_rois_for_image,
)

ort = None
_ONNX_IMPORT_ATTEMPTED = False

torch = None
_TORCH_IMPORT_ATTEMPTED = False

StarDist2D = None
_STARDIST_IMPORT_ATTEMPTED = False

Cellpose = None
_CELLPOSE_IMPORT_ATTEMPTED = False


class NucleiDetector:
    """Detector that uses custom ONNX/PT model or pre-trained StarDist model."""

    def __init__(self) -> None:
        self.loaded_model: object | None = None
        self.stardist_config = StarDistConfig()
        self._stardist_model = None
        self._stardist_model_name: str | None = None
        self._stardist_checked = False
        self._cellpose_model = None
        self._cellpose_model_type: str | None = None
        self._cellpose_checked = False

    def load_model(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.exists() or not path.is_file():
            raise ModelValidationError(f"Файл модели не найден: {model_path}")

        ext = path.suffix.lower()
        if ext not in {".onnx", ".pt"}:
            raise ModelValidationError("Поддерживаются только модели .onnx и .pt")

        if ext == ".onnx":
            ort_module = _get_onnxruntime_module()
            if ort_module is None:
                raise ModelValidationError("onnxruntime не установлен. Установите пакет onnxruntime.")
            session = ort_module.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            self.loaded_model = LoadedModel(
                path=str(path),
                model_format="onnx",
                runtime="onnxruntime",
                model=session,
                input_name=input_name,
                output_name=output_name,
            )
            return

        torch_module = _get_torch_module()
        if torch_module is None:
            raise ModelValidationError("PyTorch не установлен. Установите пакет torch.")

        model = None
        try:
            model = torch_module.jit.load(str(path), map_location="cpu")
        except Exception:
            model = torch_module.load(str(path), map_location="cpu")
        if model is None:
            raise ModelValidationError("Не удалось загрузить .pt модель")
        if hasattr(model, "eval"):
            model.eval()

        self.loaded_model = LoadedModel(
            path=str(path),
            model_format="pt",
            runtime="pytorch",
            model=model,
        )

    def reset_model(self) -> None:
        self.loaded_model = None

    def get_detection_params(self) -> dict:
        cfg = self.stardist_config
        return {
            "detector_backend": cfg.detector_backend,
            "model_name": cfg.model_name,
            "prob_thresh": cfg.prob_thresh,
            "nms_thresh": cfg.nms_thresh,
            "scale": cfg.scale,
            "min_area_px": cfg.min_area_px,
            "max_area_px": cfg.max_area_px,
            "n_tiles_x": cfg.n_tiles_x,
            "n_tiles_y": cfg.n_tiles_y,
            "preprocess_mode": cfg.preprocess_mode,
            "norm_p_low": cfg.norm_p_low,
            "norm_p_high": cfg.norm_p_high,
            "purple_filter_enabled": cfg.purple_filter_enabled,
            "purple_h_min": cfg.purple_h_min,
            "purple_h_max": cfg.purple_h_max,
            "purple_s_min": cfg.purple_s_min,
            "purple_v_max": cfg.purple_v_max,
            "min_purple_ratio": cfg.min_purple_ratio,
            "require_center_purple": cfg.require_center_purple,
            "strict_purple_filter": cfg.strict_purple_filter,
            "upscale_factor": cfg.upscale_factor,
            "stain_norm_enabled": cfg.stain_norm_enabled,
            "cellpose_model_type": cfg.cellpose_model_type,
            "cellpose_diameter_px": cfg.cellpose_diameter_px,
            "cellpose_flow_threshold": cfg.cellpose_flow_threshold,
            "cellpose_cellprob_threshold": cfg.cellpose_cellprob_threshold,
        }

    def set_detection_params(self, params: dict) -> None:
        cfg = self.stardist_config
        incoming = dict(params or {})
        preset_name = str(incoming.pop("preset", "")).strip()
        if preset_name:
            preset_values = get_runtime_preset_values(preset_name)
            preset_values.update(incoming)
            incoming = preset_values

        detector_backend = str(
            incoming.get("detector_backend", cfg.detector_backend)
        ).strip().lower()
        if not detector_backend:
            detector_backend = cfg.detector_backend
        if detector_backend not in DETECTOR_BACKENDS:
            raise ValueError("Выбранный встроенный детектор не поддерживается")

        new_model_name = str(incoming.get("model_name", cfg.model_name)).strip() or cfg.model_name

        prob_thresh = float(incoming.get("prob_thresh", cfg.prob_thresh))
        nms_thresh = float(incoming.get("nms_thresh", cfg.nms_thresh))
        scale = float(incoming.get("scale", cfg.scale))
        min_area_px = int(incoming.get("min_area_px", cfg.min_area_px))
        max_area_px = int(incoming.get("max_area_px", cfg.max_area_px))
        n_tiles_x = int(incoming.get("n_tiles_x", cfg.n_tiles_x))
        n_tiles_y = int(incoming.get("n_tiles_y", cfg.n_tiles_y))
        preprocess_mode = str(incoming.get("preprocess_mode", cfg.preprocess_mode)).strip().lower()
        if not preprocess_mode:
            preprocess_mode = cfg.preprocess_mode

        norm_p_low = float(incoming.get("norm_p_low", cfg.norm_p_low))
        norm_p_high = float(incoming.get("norm_p_high", cfg.norm_p_high))

        purple_filter_enabled = _to_bool(
            incoming.get("purple_filter_enabled", cfg.purple_filter_enabled)
        )
        purple_h_min = int(incoming.get("purple_h_min", cfg.purple_h_min))
        purple_h_max = int(incoming.get("purple_h_max", cfg.purple_h_max))
        purple_s_min = int(incoming.get("purple_s_min", cfg.purple_s_min))
        purple_v_max = int(incoming.get("purple_v_max", cfg.purple_v_max))
        min_purple_ratio = float(incoming.get("min_purple_ratio", cfg.min_purple_ratio))
        require_center_purple = _to_bool(
            incoming.get("require_center_purple", cfg.require_center_purple)
        )
        strict_purple_filter = _to_bool(
            incoming.get("strict_purple_filter", cfg.strict_purple_filter)
        )
        upscale_factor = float(incoming.get("upscale_factor", cfg.upscale_factor))
        stain_norm_enabled = _to_bool(incoming.get("stain_norm_enabled", cfg.stain_norm_enabled))
        cellpose_model_type = str(
            incoming.get("cellpose_model_type", cfg.cellpose_model_type)
        ).strip() or cfg.cellpose_model_type
        cellpose_diameter_px = float(
            incoming.get("cellpose_diameter_px", cfg.cellpose_diameter_px)
        )
        cellpose_flow_threshold = float(
            incoming.get("cellpose_flow_threshold", cfg.cellpose_flow_threshold)
        )
        cellpose_cellprob_threshold = float(
            incoming.get("cellpose_cellprob_threshold", cfg.cellpose_cellprob_threshold)
        )

        if not (0.0 <= prob_thresh <= 1.0):
            raise ValueError("Порог уверенности должен быть в диапазоне [0, 1]")
        if not (0.0 <= nms_thresh <= 1.0):
            raise ValueError("Порог разделения соседних ядер должен быть в диапазоне [0, 1]")
        if scale <= 0:
            raise ValueError("Масштаб для сети должен быть больше 0")
        if min_area_px < 0:
            raise ValueError("Минимальная площадь ядра должна быть >= 0")
        if max_area_px < 0:
            raise ValueError("Максимальная площадь ядра должна быть >= 0")
        if max_area_px > 0 and max_area_px <= min_area_px:
            raise ValueError("Максимальная площадь ядра должна быть больше минимальной")
        if n_tiles_x < 1 or n_tiles_y < 1:
            raise ValueError("Количество тайлов должно быть >= 1")
        if preprocess_mode not in STARDIST_PREPROCESS_MODES:
            raise ValueError(
                "Режим подготовки изображения указан неверно"
            )
        if not (0.0 <= norm_p_low < 100.0):
            raise ValueError("Нижний перцентиль нормализации должен быть в диапазоне [0, 100)")
        if not (0.0 < norm_p_high <= 100.0):
            raise ValueError("Верхний перцентиль нормализации должен быть в диапазоне (0, 100]")
        if norm_p_high <= norm_p_low:
            raise ValueError("Верхний перцентиль должен быть больше нижнего")
        if not (0 <= purple_h_min <= 179):
            raise ValueError("Минимальный оттенок фиолетового должен быть в диапазоне [0, 179]")
        if not (0 <= purple_h_max <= 179):
            raise ValueError("Максимальный оттенок фиолетового должен быть в диапазоне [0, 179]")
        if not (0 <= purple_s_min <= 255):
            raise ValueError("Минимальная насыщенность должна быть в диапазоне [0, 255]")
        if not (0 <= purple_v_max <= 255):
            raise ValueError("Максимальная яркость должна быть в диапазоне [0, 255]")
        if not (0.0 <= min_purple_ratio <= 1.0):
            raise ValueError("Минимальная доля окрашенных пикселей должна быть в диапазоне [0, 1]")
        if not (1.0 <= upscale_factor <= 4.0):
            raise ValueError("Коэффициент апскейла должен быть в диапазоне [1, 4]")
        if cellpose_diameter_px <= 0.0:
            raise ValueError("Диаметр ядра для Cellpose должен быть больше 0")
        if not (0.0 <= cellpose_flow_threshold <= 2.0):
            raise ValueError("Порог потока Cellpose должен быть в диапазоне [0, 2]")
        if not (-10.0 <= cellpose_cellprob_threshold <= 10.0):
            raise ValueError("Порог вероятности маски Cellpose должен быть в диапазоне [-10, 10]")

        old_backend = cfg.detector_backend
        old_model_name = cfg.model_name
        old_cellpose_model_type = cfg.cellpose_model_type
        cfg.detector_backend = detector_backend
        cfg.model_name = new_model_name
        cfg.prob_thresh = prob_thresh
        cfg.nms_thresh = nms_thresh
        cfg.scale = scale
        cfg.min_area_px = min_area_px
        cfg.max_area_px = max_area_px
        cfg.n_tiles_x = n_tiles_x
        cfg.n_tiles_y = n_tiles_y
        cfg.preprocess_mode = preprocess_mode
        cfg.norm_p_low = norm_p_low
        cfg.norm_p_high = norm_p_high
        cfg.purple_filter_enabled = purple_filter_enabled
        cfg.purple_h_min = purple_h_min
        cfg.purple_h_max = purple_h_max
        cfg.purple_s_min = purple_s_min
        cfg.purple_v_max = purple_v_max
        cfg.min_purple_ratio = min_purple_ratio
        cfg.require_center_purple = require_center_purple
        cfg.strict_purple_filter = strict_purple_filter
        cfg.upscale_factor = upscale_factor
        cfg.stain_norm_enabled = stain_norm_enabled
        cfg.cellpose_model_type = cellpose_model_type
        cfg.cellpose_diameter_px = cellpose_diameter_px
        cfg.cellpose_flow_threshold = cellpose_flow_threshold
        cfg.cellpose_cellprob_threshold = cellpose_cellprob_threshold

        if old_model_name != new_model_name:
            self._stardist_model = None
            self._stardist_model_name = None
            self._stardist_checked = False
        if old_backend != detector_backend:
            self._cellpose_checked = False
        if old_cellpose_model_type != cellpose_model_type:
            self._cellpose_model = None
            self._cellpose_model_type = None
            self._cellpose_checked = False

    def detect_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        nuclei = self.detect_nuclei_instances(image_bgr)
        return _nuclei_to_binary_mask(nuclei, image_bgr.shape[:2])

    def detect_nuclei_instances(self, image_bgr: np.ndarray) -> list[dict]:
        nuclei, _ = self.detect_nuclei_instances_with_diagnostics(image_bgr)
        return nuclei

    def detect_nuclei_instances_with_diagnostics(
        self,
        image_bgr: np.ndarray,
        area_small_px: float = 15.0,
        area_large_px: float = 2000.0,
    ) -> tuple[list[dict], dict]:
        cfg = self.stardist_config
        upscale_factor = float(cfg.upscale_factor)
        work_image = image_bgr
        orig_min_area = cfg.min_area_px
        orig_max_area = cfg.max_area_px
        orig_cellpose_diameter = cfg.cellpose_diameter_px

        if upscale_factor > 1.0 + 1e-4:
            work_image = _upscale_image(image_bgr, upscale_factor)
            scale_sq = upscale_factor * upscale_factor
            cfg.min_area_px = int(round(orig_min_area * scale_sq))
            if orig_max_area > 0:
                cfg.max_area_px = int(round(orig_max_area * scale_sq))
            cfg.cellpose_diameter_px = float(orig_cellpose_diameter * upscale_factor)

        try:
            diagnostics = _build_detection_diagnostics_template(area_small_px, area_large_px)
            if self.loaded_model is None:
                backend = str(cfg.detector_backend).strip().lower()
                diagnostics["backend"] = backend
                if backend == "cellpose_nuclei":
                    nuclei = self._cellpose_pretrained_instances(work_image, diagnostics=diagnostics)
                else:
                    nuclei = self._default_pretrained_instances(work_image, diagnostics=diagnostics)
            else:
                diagnostics["backend"] = "custom_model"
                nuclei_raw = _extract_nuclei_from_binary_mask(self._model_mask(work_image))
                diagnostics["model_candidates_before_nms"] = None
                diagnostics["after_nms_count"] = int(len(nuclei_raw))
                nuclei = self._postprocess_nuclei(nuclei_raw, work_image, diagnostics=diagnostics)

            if upscale_factor > 1.0 + 1e-4:
                nuclei = _scale_nuclei_coords(nuclei, upscale_factor)

            diagnostics["final_count"] = int(len(nuclei))
            diagnostics["final_area_stats"] = _summarize_nuclei_areas(
                nuclei,
                small_area_px=float(diagnostics["small_area_threshold_px"]),
                large_area_px=float(diagnostics["large_area_threshold_px"]),
            )
            return nuclei, diagnostics
        finally:
            cfg.min_area_px = orig_min_area
            cfg.max_area_px = orig_max_area
            cfg.cellpose_diameter_px = orig_cellpose_diameter

    def _preprocess_patch(self, image_bgr_patch: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image_bgr_patch, cv2.COLOR_BGR2RGB)
        data = image_rgb.astype(np.float32) / 255.0
        data = np.transpose(data, (2, 0, 1))[None, ...]
        return data

    def _output_to_probability(self, output: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
        arr = np.asarray(output)

        while arr.ndim > 3:
            arr = arr[0]
        if arr.ndim == 3:
            if arr.shape[0] in (1, 2, 3):
                arr = arr[0]
            elif arr.shape[-1] in (1, 2, 3):
                arr = arr[..., 0]
            else:
                arr = arr[0]

        arr = arr.astype(np.float32)
        if arr.max() > 1.0 or arr.min() < 0.0:
            arr = cv2.normalize(arr, None, 0.0, 1.0, cv2.NORM_MINMAX)

        h, w = target_hw
        return cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)

    def _model_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        if self.loaded_model is None:
            raise RuntimeError("Пользовательская модель не загружена")

        h, w = image_bgr.shape[:2]
        tile_size = 512
        if self.loaded_model.model_format == "onnx":
            shape = self.loaded_model.model.get_inputs()[0].shape
            if len(shape) >= 4:
                h_in = shape[2] if isinstance(shape[2], int) and shape[2] > 0 else None
                w_in = shape[3] if isinstance(shape[3], int) and shape[3] > 0 else None
                if h_in is not None and w_in is not None and h_in == w_in:
                    tile_size = int(h_in)

        tile_size = max(128, tile_size)
        overlap = max(16, tile_size // 4)
        stride = max(32, tile_size - overlap)

        prob_sum = np.zeros((h, w), dtype=np.float32)
        prob_count = np.zeros((h, w), dtype=np.float32)

        y_starts = list(range(0, max(1, h - tile_size + 1), stride))
        x_starts = list(range(0, max(1, w - tile_size + 1), stride))
        if not y_starts or y_starts[-1] != max(0, h - tile_size):
            y_starts.append(max(0, h - tile_size))
        if not x_starts or x_starts[-1] != max(0, w - tile_size):
            x_starts.append(max(0, w - tile_size))

        for y0 in y_starts:
            for x0 in x_starts:
                y1 = min(h, y0 + tile_size)
                x1 = min(w, x0 + tile_size)
                patch = image_bgr[y0:y1, x0:x1]
                ph, pw = patch.shape[:2]

                if ph != tile_size or pw != tile_size:
                    padded = np.zeros((tile_size, tile_size, 3), dtype=patch.dtype)
                    padded[:ph, :pw] = patch
                    patch = padded

                input_tensor = self._preprocess_patch(patch)

                if self.loaded_model.model_format == "onnx":
                    assert self.loaded_model.input_name is not None
                    assert self.loaded_model.output_name is not None
                    output = self.loaded_model.model.run(
                        [self.loaded_model.output_name],
                        {self.loaded_model.input_name: input_tensor},
                    )[0]
                else:
                    torch_module = _get_torch_module()
                    if torch_module is None:
                        raise RuntimeError("PyTorch runtime is not available")
                    model = self.loaded_model.model
                    with torch_module.no_grad():
                        tensor = torch_module.from_numpy(input_tensor)
                        output = model(tensor)
                        if isinstance(output, (tuple, list)):
                            output = output[0]
                        if hasattr(output, "detach"):
                            output = output.detach().cpu().numpy()

                prob_patch = self._output_to_probability(output, (tile_size, tile_size))
                prob_sum[y0:y1, x0:x1] += prob_patch[:ph, :pw]
                prob_count[y0:y1, x0:x1] += 1.0

        prob = prob_sum / np.clip(prob_count, 1e-6, None)
        binary = (prob > 0.5).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    def _default_pretrained_instances(
        self,
        image_bgr: np.ndarray,
        diagnostics: dict | None = None,
    ) -> list[dict]:
        model = self._get_stardist_model()
        if model is None:
            raise RuntimeError(
                "Предобученная нейросеть StarDist недоступна. "
                "Установите stardist/tensorflow-cpu или загрузите пользовательскую .pt/.onnx модель."
            )

        cfg = self.stardist_config
        n_channel_in = int(getattr(getattr(model, "config", None), "n_channel_in", 3))
        model_input = self._prepare_stardist_input(image_bgr, n_channel_in)
        apply_stardist_thresholds(model, cfg.prob_thresh, cfg.nms_thresh)
        if model_input.ndim == 2:
            n_tiles = (int(cfg.n_tiles_y), int(cfg.n_tiles_x))
        else:
            n_tiles = (int(cfg.n_tiles_y), int(cfg.n_tiles_x), 1)

        labels, details = model.predict_instances(
            model_input,
            prob_thresh=float(cfg.prob_thresh),
            nms_thresh=float(cfg.nms_thresh),
            scale=float(cfg.scale),
            n_tiles=n_tiles,
            show_tile_progress=False,
            verbose=False,
        )

        nuclei = _extract_nuclei_from_label_image(np.asarray(labels).astype(np.int32, copy=False))
        if diagnostics is not None:
            diagnostics["model_candidates_before_nms"] = _extract_model_candidate_count(details)
            diagnostics["after_nms_count"] = int(len(nuclei))
        return self._postprocess_nuclei(nuclei, image_bgr, diagnostics=diagnostics)

    def _cellpose_pretrained_instances(
        self,
        image_bgr: np.ndarray,
        diagnostics: dict | None = None,
    ) -> list[dict]:
        model = self._get_cellpose_model()
        if model is None:
            raise RuntimeError(
                "Предобученная нейросеть Cellpose Nuclei недоступна. "
                "Установите пакет cellpose или загрузите пользовательскую .pt/.onnx модель."
            )

        cfg = self.stardist_config
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        masks, *_ = model.eval(
            image_rgb,
            channels=[0, 0],
            diameter=float(cfg.cellpose_diameter_px),
            flow_threshold=float(cfg.cellpose_flow_threshold),
            cellprob_threshold=float(cfg.cellpose_cellprob_threshold),
            min_size=int(max(0, cfg.min_area_px)),
            normalize=True,
        )

        labels = masks
        if isinstance(labels, list):
            if not labels:
                return []
            labels = labels[0]
        labels = np.asarray(labels)
        labels = np.squeeze(labels)
        if labels.ndim != 2:
            raise RuntimeError("Cellpose вернул результат неожиданной размерности")

        nuclei = _extract_nuclei_from_label_image(labels.astype(np.int32, copy=False))
        if diagnostics is not None:
            diagnostics["model_candidates_before_nms"] = int(len(nuclei))
            diagnostics["after_nms_count"] = int(len(nuclei))
        return self._postprocess_nuclei(nuclei, image_bgr, diagnostics=diagnostics)

    def _postprocess_nuclei(
        self,
        nuclei: list[dict],
        image_bgr: np.ndarray,
        diagnostics: dict | None = None,
    ) -> list[dict]:
        cfg = self.stardist_config
        min_area = int(cfg.min_area_px)
        max_area = int(cfg.max_area_px)

        filtered: list[dict] = []
        for nucleus in nuclei:
            area = float(nucleus.get("area_px", 0.0))
            if min_area > 0 and area < min_area:
                if diagnostics is not None:
                    diagnostics["removed_by_min_area"] += 1
                continue
            if max_area > 0 and area > max_area:
                if diagnostics is not None:
                    diagnostics["removed_by_max_area"] += 1
                continue
            filtered.append(nucleus)
        if diagnostics is not None:
            diagnostics["after_area_filters_count"] = int(len(filtered))

        if cfg.purple_filter_enabled:
            filtered, purple_stats = _filter_nuclei_by_purple_stain(
                filtered,
                image_bgr,
                hue_min=int(cfg.purple_h_min),
                hue_max=int(cfg.purple_h_max),
                sat_min=int(cfg.purple_s_min),
                val_max=int(cfg.purple_v_max),
                min_ratio=float(cfg.min_purple_ratio),
                require_center=bool(cfg.require_center_purple),
                strict=bool(cfg.strict_purple_filter),
            )
            if diagnostics is not None:
                diagnostics["purple_filter_enabled"] = True
                diagnostics["strict_purple_filter"] = bool(cfg.strict_purple_filter)
                diagnostics["purple_checked"] = int(purple_stats.get("checked", 0))
                diagnostics["removed_by_purple_ratio"] = int(purple_stats.get("failed_ratio", 0))
                diagnostics["removed_by_purple_center"] = int(purple_stats.get("failed_center", 0))
                diagnostics["removed_by_purple_other"] = int(purple_stats.get("failed_other", 0))
                diagnostics["removed_by_purple_total"] = int(purple_stats.get("rejected", 0))
                diagnostics["weak_color_marked"] = int(purple_stats.get("weak_marked", 0))
        elif diagnostics is not None:
            diagnostics["purple_filter_enabled"] = False
            diagnostics["strict_purple_filter"] = bool(cfg.strict_purple_filter)

        if diagnostics is not None:
            diagnostics["after_purple_count"] = int(len(filtered))

        return filtered

    def _get_stardist_model(self):
        cfg = self.stardist_config

        if self._stardist_checked and self._stardist_model is not None and self._stardist_model_name == cfg.model_name:
            return self._stardist_model
        if self._stardist_checked and self._stardist_model is None and self._stardist_model_name == cfg.model_name:
            return None

        self._stardist_checked = True
        stardist_cls = _get_stardist_class()
        if stardist_cls is None:
            self._stardist_model = None
            self._stardist_model_name = cfg.model_name
            return None

        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self._stardist_model = stardist_cls.from_pretrained(cfg.model_name)
            self._stardist_model_name = cfg.model_name
        except Exception:
            self._stardist_model = None
            self._stardist_model_name = cfg.model_name
        return self._stardist_model

    def _get_cellpose_model(self):
        cfg = self.stardist_config

        if (
            self._cellpose_checked
            and self._cellpose_model is not None
            and self._cellpose_model_type == cfg.cellpose_model_type
        ):
            return self._cellpose_model
        if (
            self._cellpose_checked
            and self._cellpose_model is None
            and self._cellpose_model_type == cfg.cellpose_model_type
        ):
            return None

        self._cellpose_checked = True
        cellpose_cls = _get_cellpose_class()
        if cellpose_cls is None:
            self._cellpose_model = None
            self._cellpose_model_type = cfg.cellpose_model_type
            return None

        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self._cellpose_model = cellpose_cls(gpu=False, model_type=cfg.cellpose_model_type)
            self._cellpose_model_type = cfg.cellpose_model_type
        except Exception:
            self._cellpose_model = None
            self._cellpose_model_type = cfg.cellpose_model_type
        return self._cellpose_model

    def _prepare_stardist_input(self, image_bgr: np.ndarray, n_channel_in: int) -> np.ndarray:
        cfg = self.stardist_config
        work = image_bgr
        if bool(cfg.stain_norm_enabled):
            custom_mean, custom_std = get_custom_hes_reference()
            work = _reinhard_stain_normalize(work, ref_mean=custom_mean, ref_std=custom_std)
        rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = percentile_normalize_rgb(rgb, cfg.norm_p_low, cfg.norm_p_high)
        normalized = np.clip(normalized.astype(np.float32), 0.0, 1.0)

        if n_channel_in <= 1:
            gray = np.mean(normalized, axis=2)
            return np.clip(gray.astype(np.float32), 0.0, 1.0)

        if n_channel_in == normalized.shape[2]:
            return normalized
        if n_channel_in < normalized.shape[2]:
            return normalized[..., :n_channel_in]

        repeats = int(np.ceil(n_channel_in / normalized.shape[2]))
        expanded = np.tile(normalized, (1, 1, repeats))
        return expanded[..., :n_channel_in]


_detector = NucleiDetector()


@lru_cache(maxsize=1)
def _is_stardist_package_available() -> bool:
    return importlib.util.find_spec("stardist") is not None


@lru_cache(maxsize=1)
def _is_cellpose_package_available() -> bool:
    return importlib.util.find_spec("cellpose") is not None


def _get_stardist_class():
    global StarDist2D, _STARDIST_IMPORT_ATTEMPTED
    if StarDist2D is not None:
        return StarDist2D
    if _STARDIST_IMPORT_ATTEMPTED:
        return None

    _STARDIST_IMPORT_ATTEMPTED = True
    try:
        from stardist.models import StarDist2D as _StarDist2D
    except Exception:
        return None
    StarDist2D = _StarDist2D
    return StarDist2D


def _get_cellpose_class():
    global Cellpose, _CELLPOSE_IMPORT_ATTEMPTED
    if Cellpose is not None:
        return Cellpose
    if _CELLPOSE_IMPORT_ATTEMPTED:
        return None

    _CELLPOSE_IMPORT_ATTEMPTED = True
    try:
        from cellpose.models import Cellpose as _Cellpose
    except Exception:
        return None
    Cellpose = _Cellpose
    return Cellpose


def _get_onnxruntime_module():
    global ort, _ONNX_IMPORT_ATTEMPTED
    if ort is not None:
        return ort
    if _ONNX_IMPORT_ATTEMPTED:
        return None

    _ONNX_IMPORT_ATTEMPTED = True
    try:
        import onnxruntime as _ort
    except Exception:
        return None
    ort = _ort
    return ort


def _get_torch_module():
    global torch, _TORCH_IMPORT_ATTEMPTED
    if torch is not None:
        return torch
    if _TORCH_IMPORT_ATTEMPTED:
        return None

    _TORCH_IMPORT_ATTEMPTED = True
    try:
        import torch as _torch
    except Exception:
        return None
    torch = _torch
    return torch


def apply_stardist_thresholds(model, prob_thresh: float, nms_thresh: float) -> None:
    thresholds = getattr(model, "thresholds", None)
    if thresholds is None:
        return
    try:
        if isinstance(thresholds, dict):
            thresholds["prob"] = float(prob_thresh)
            thresholds["nms"] = float(nms_thresh)
            return
        if hasattr(thresholds, "prob"):
            setattr(thresholds, "prob", float(prob_thresh))
        if hasattr(thresholds, "nms"):
            setattr(thresholds, "nms", float(nms_thresh))
    except Exception:
        return


def _reinhard_stain_normalize(
    image_bgr: np.ndarray,
    ref_mean: np.ndarray | None = None,
    ref_std: np.ndarray | None = None,
) -> np.ndarray:
    if ref_mean is None:
        ref_mean = _HE_REF_MEAN_LAB
    if ref_std is None:
        ref_std = _HE_REF_STD_LAB
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    for c in range(3):
        src_mean = float(lab[:, :, c].mean())
        src_std = float(lab[:, :, c].std()) + 1e-6
        lab[:, :, c] = (
            (lab[:, :, c] - src_mean) * (ref_std[c] / src_std)
            + ref_mean[c]
        )
    lab = np.clip(lab, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _upscale_image(image_bgr: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 1e-4:
        return image_bgr
    h, w = image_bgr.shape[:2]
    out_h = int(round(h * factor))
    out_w = int(round(w * factor))
    return cv2.resize(image_bgr, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)


def _scale_nuclei_coords(nuclei: list[dict], factor: float) -> list[dict]:
    if abs(factor - 1.0) < 1e-4:
        return nuclei
    inv = 1.0 / factor
    inv2 = inv * inv
    out: list[dict] = []
    for nucleus in nuclei:
        item = dict(nucleus)
        center = nucleus.get("center")
        contour = nucleus.get("contour", [])
        if center is not None:
            item["center"] = (float(center[0]) * inv, float(center[1]) * inv)
        item["contour"] = [(float(x) * inv, float(y) * inv) for x, y in contour]
        item["area_px"] = float(nucleus.get("area_px", 0.0)) * inv2
        out.append(item)
    return out


def _read_image_unicode(path: str) -> np.ndarray:
    ext = Path(path).suffix.lower()
    if ext not in SUPPORTED_IMAGE_FORMATS:
        raise ValueError(
            "Неподдерживаемый формат изображения. Поддерживаются TIFF, PNG, JPEG."
        )

    raw = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")
    return image


@lru_cache(maxsize=64)
def _cached_image(path: str, mtime_ns: int) -> np.ndarray:
    return _read_image_unicode(path)


@lru_cache(maxsize=64)
def _cached_display_image(path: str, mtime_ns: int, max_side: int) -> tuple[np.ndarray, float]:
    image = _read_image_unicode(path)
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return image, 1.0

    scale = long_side / float(max_side)
    new_w = int(round(w / scale))
    new_h = int(round(h / scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def load_image(image_path: str) -> np.ndarray:
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Изображение не найдено: {image_path}")
    stat = path.stat()
    return _cached_image(str(path), stat.st_mtime_ns).copy()


def load_display_image(image_path: str, max_side: int = 2400) -> tuple[np.ndarray, float]:
    """Return resized image for fast rendering and scale factor (orig_px / display_px)."""
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Изображение не найдено: {image_path}")
    stat = path.stat()
    image, scale = _cached_display_image(str(path), stat.st_mtime_ns, int(max_side))
    return image.copy(), float(scale)


def load_custom_model(model_path: str) -> None:
    _detector.load_model(model_path)


def reset_custom_model() -> None:
    _detector.reset_model()


def get_loaded_model_info() -> dict | None:
    if _detector.loaded_model is None:
        return None
    return {
        "path": _detector.loaded_model.path,
        "format": _detector.loaded_model.model_format,
        "runtime": _detector.loaded_model.runtime,
    }


def get_default_detector_info() -> dict:
    cfg = _detector.get_detection_params()
    backend_name = str(cfg.get("detector_backend", "stardist")).strip().lower()
    if backend_name == "cellpose_nuclei":
        if _is_cellpose_package_available():
            return {
                "type": "pretrained",
                "backend": backend_name,
                "name": "Cellpose Nuclei",
                "runtime": "pytorch-cpu",
            }
        return {
            "type": "unavailable",
            "backend": backend_name,
            "name": "Cellpose Nuclei (недоступна)",
            "runtime": "pytorch-cpu",
        }

    if _is_stardist_package_available():
        return {
            "type": "pretrained",
            "backend": "stardist",
            "name": f"StarDist ({cfg['model_name']})",
            "runtime": "tensorflow-cpu",
        }
    return {
        "type": "unavailable",
        "backend": "stardist",
        "name": "StarDist (недоступна)",
        "runtime": "tensorflow-cpu",
    }


def warmup_pretrained_detector() -> None:
    if _detector.loaded_model is not None:
        return
    cfg = _detector.get_detection_params()
    backend_name = str(cfg.get("detector_backend", "stardist")).strip().lower()
    if backend_name == "cellpose_nuclei":
        model = _detector._get_cellpose_model()
        if model is None:
            raise RuntimeError(
                "Предобученная нейросеть Cellpose Nuclei недоступна. "
                "Установите cellpose или загрузите .pt/.onnx модель."
            )
        return

    model = _detector._get_stardist_model()
    if model is None:
        raise RuntimeError(
            "Предобученная нейросеть StarDist недоступна. "
            "Установите stardist/tensorflow-cpu или загрузите .pt/.onnx модель."
        )


def run_detection_job(payload: dict, event_queue) -> None:
    try:
        mode = str(payload.get("mode", "single"))
        enhancement_params = payload.get("enhancement_params")
        detection_params = payload.get("detection_params")
        custom_model_path = str(payload.get("custom_model_path", "") or "").strip()

        if isinstance(detection_params, dict):
            set_detection_params(detection_params)

        if custom_model_path:
            load_custom_model(custom_model_path)
        else:
            reset_custom_model()

        if mode == "single":
            image_path = str(payload.get("image_path", "")).strip()
            if not image_path:
                raise ValueError("Не передан путь к изображению для детекции")
            roi_rect = payload.get("roi_rect")
            roi_poly = payload.get("roi_poly")
            roi_before_infer = bool(payload.get("roi_before_infer", False))
            if roi_before_infer and (roi_rect or roi_poly):
                avg_diam = float(payload.get("average_cell_diameter_px", 14.0))
                if avg_diam <= 0.0:
                    avg_diam = 14.0
                nuclei, _ = detect_nuclei_tiled(
                    image_path=image_path,
                    average_cell_diameter_px=avg_diam,
                    enhancement_params=enhancement_params,
                    min_roi_cover_ratio=float(payload.get("min_roi_cover_ratio", 0.05)),
                    roi_rect=roi_rect,
                    roi_poly=roi_poly,
                )
            else:
                nuclei = detect_nuclei(image_path, enhancement_params)
            event_queue.put({"type": "result", "nuclei": nuclei})
            return

        if mode == "batch":
            image_paths = [str(x) for x in payload.get("image_paths", [])]
            rois_by_file = payload.get("rois_by_file", {})
            pixels_per_mm = float(payload.get("pixels_per_mm", 0.0))
            if pixels_per_mm <= 0.0:
                raise ValueError("Некорректный коэффициент масштаба для пакетной обработки")

            rows: list[dict] = []
            total = len(image_paths)
            for idx, image_path in enumerate(image_paths, start=1):
                file_name = Path(image_path).name
                event_queue.put(
                    {
                        "type": "progress",
                        "current": idx - 1,
                        "total": total,
                        "file_name": file_name,
                    }
                )

                nuclei = detect_nuclei(image_path, enhancement_params)
                file_rois = rois_by_file.get(image_path, [])
                for roi in file_rois:
                    from .roi import build_roi_metrics
                    metrics = build_roi_metrics(roi, nuclei, pixels_per_mm)
                    rows.append(
                        {
                            "Файл": file_name,
                            "ROI ID": metrics.get("ROI ID"),
                            "Название ROI": roi.get("name", ""),
                            "Тип": metrics.get("Тип"),
                            "Площадь (мм²)": metrics.get("Площадь (мм²)"),
                            "Количество ядер": metrics.get("Количество ядер"),
                            "Плотность (ядра/мм²)": metrics.get("Плотность (ядра/мм²)"),
                        }
                    )

                event_queue.put(
                    {
                        "type": "progress",
                        "current": idx,
                        "total": total,
                        "file_name": file_name,
                    }
                )

            event_queue.put({"type": "result", "rows": rows})
            return

        raise ValueError(f"Неизвестный режим задачи детекции: {mode}")
    except Exception as exc:
        tb = traceback.format_exc()
        event_queue.put({"type": "error", "message": f"{exc}\n{tb}"})


def get_pretrained_model_names() -> list[str]:
    return list(STARDIST_PRETRAINED_MODELS)


def get_detector_backends() -> list[str]:
    return list(DETECTOR_BACKENDS)


def get_preprocess_modes() -> list[str]:
    return list(STARDIST_PREPROCESS_MODES)


def get_detection_presets() -> list[str]:
    return list(DETECTION_PRESETS)


def get_runtime_detection_presets() -> list[str]:
    return list(DETECTION_RUNTIME_PRESETS)


def build_detection_params_for_preset(preset_name: str) -> dict:
    params = get_detection_params()
    params.update(get_runtime_preset_values(preset_name))
    return params


def get_detection_params() -> dict:
    return _detector.get_detection_params()


def set_detection_params(params: dict) -> None:
    _detector.set_detection_params(params)


def detect_nuclei_in_image(
    image_bgr: np.ndarray,
    enhancement_params: dict | None = None,
) -> list[dict]:
    enhanced = apply_image_enhancement(image_bgr, enhancement_params)
    return _detector.detect_nuclei_instances(enhanced)


def detect_nuclei_in_image_with_diagnostics(
    image_bgr: np.ndarray,
    enhancement_params: dict | None = None,
    area_small_px: float = 15.0,
    area_large_px: float = 2000.0,
) -> tuple[list[dict], dict]:
    enhanced = apply_image_enhancement(image_bgr, enhancement_params)
    return _detector.detect_nuclei_instances_with_diagnostics(
        enhanced,
        area_small_px=area_small_px,
        area_large_px=area_large_px,
    )


def detect_nuclei(
    image_path: str,
    enhancement_params: dict | None = None,
) -> list[dict]:
    image = load_image(image_path)
    return detect_nuclei_in_image(image, enhancement_params)


def detect_nuclei_with_diagnostics(
    image_path: str,
    enhancement_params: dict | None = None,
    area_small_px: float = 15.0,
    area_large_px: float = 2000.0,
) -> tuple[list[dict], dict]:
    image = load_image(image_path)
    return detect_nuclei_in_image_with_diagnostics(
        image,
        enhancement_params=enhancement_params,
        area_small_px=area_small_px,
        area_large_px=area_large_px,
    )


def estimate_average_cell_diameter_px(
    average_cell_diameter_px: float | None = None,
    average_cell_bbox: Sequence[float] | None = None,
) -> float:
    if average_cell_diameter_px is not None:
        diameter = float(average_cell_diameter_px)
        if diameter <= 0.0:
            raise ValueError("average_cell_diameter_px должен быть больше 0")
        return diameter

    if average_cell_bbox is None:
        raise ValueError(
            "Нужно задать average_cell_diameter_px или average_cell_bbox (x, y, w, h)"
        )

    if len(average_cell_bbox) != 4:
        raise ValueError("average_cell_bbox должен содержать 4 значения: x, y, w, h")
    width = abs(float(average_cell_bbox[2]))
    height = abs(float(average_cell_bbox[3]))
    diameter = 0.5 * (width + height)
    if diameter <= 0.0:
        raise ValueError("average_cell_bbox должен задавать положительные ширину и высоту")
    return float(diameter)


def _generate_tile_starts(start: int, end: int, tile_size: int, stride: int) -> list[int]:
    if end <= start:
        return [start]
    span = end - start
    if span <= tile_size:
        return [start]

    starts = list(range(start, end - tile_size + 1, stride))
    if not starts:
        starts = [start]

    last_start = end - tile_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _tile_bbox_intersects(a: Sequence[float], b: Sequence[float]) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def _polygon_tile_cover_ratio(
    polygon_points: Sequence[tuple[float, float]],
    tile_rect: Sequence[int],
) -> float:
    tx, ty, tw, th = [int(v) for v in tile_rect]
    if tw <= 0 or th <= 0:
        return 0.0

    tile_bbox = [float(tx), float(ty), float(tx + tw), float(ty + th)]
    poly_arr = np.asarray(polygon_points, dtype=np.float32)
    if poly_arr.size == 0:
        return 0.0
    poly_bbox = [
        float(np.min(poly_arr[:, 0])),
        float(np.min(poly_arr[:, 1])),
        float(np.max(poly_arr[:, 0])),
        float(np.max(poly_arr[:, 1])),
    ]
    if not _tile_bbox_intersects(poly_bbox, tile_bbox):
        return 0.0

    local_poly = np.round(poly_arr - np.asarray([[tx, ty]], dtype=np.float32)).astype(np.int32)
    mask = np.zeros((th, tw), dtype=np.uint8)
    cv2.fillPoly(mask, [local_poly.reshape((-1, 1, 2))], color=255)
    cover = float(np.count_nonzero(mask)) / float(max(1, tw * th))
    return cover


def _build_tiles_for_roi(
    roi_infer: dict,
    image_w_infer: int,
    image_h_infer: int,
    tile_size_infer: int,
    stride_infer: int,
    min_roi_cover_ratio: float,
) -> list[dict]:
    bbox = roi_infer.get("bbox", [0.0, 0.0, 0.0, 0.0])
    bx1 = int(max(0, np.floor(float(bbox[0]))))
    by1 = int(max(0, np.floor(float(bbox[1]))))
    bx2 = int(min(image_w_infer, np.ceil(float(bbox[2]))))
    by2 = int(min(image_h_infer, np.ceil(float(bbox[3]))))
    if bx2 <= bx1 or by2 <= by1:
        return []

    x_starts = _generate_tile_starts(bx1, bx2, tile_size_infer, stride_infer)
    y_starts = _generate_tile_starts(by1, by2, tile_size_infer, stride_infer)
    roi_bbox = [bx1, by1, bx2, by2]

    tiles: list[dict] = []
    for y0 in y_starts:
        for x0 in x_starts:
            x1 = min(image_w_infer, x0 + tile_size_infer)
            y1 = min(image_h_infer, y0 + tile_size_infer)
            tw = int(x1 - x0)
            th = int(y1 - y0)
            if tw <= 0 or th <= 0:
                continue

            cover_ratio = 1.0
            if roi_infer.get("kind") == "poly":
                cover_ratio = _polygon_tile_cover_ratio(roi_infer.get("points", []), [x0, y0, tw, th])
                if cover_ratio <= 0.0:
                    continue
                if min_roi_cover_ratio > 0.0 and cover_ratio < float(min_roi_cover_ratio):
                    continue

            touches_outer = (
                x0 <= roi_bbox[0]
                or y0 <= roi_bbox[1]
                or x1 >= roi_bbox[2]
                or y1 >= roi_bbox[3]
                or x0 <= 0
                or y0 <= 0
                or x1 >= image_w_infer
                or y1 >= image_h_infer
            )

            tiles.append(
                {
                    "x": int(x0),
                    "y": int(y0),
                    "w": int(tw),
                    "h": int(th),
                    "touches_outer": bool(touches_outer),
                    "roi_cover_ratio": float(cover_ratio),
                    "roi_id": roi_infer.get("id"),
                }
            )
    return tiles


def _shift_nucleus_to_global(nucleus: dict, shift_x: float, shift_y: float) -> dict:
    out = dict(nucleus)
    center = nucleus.get("center")
    if center is not None:
        out["center"] = (float(center[0]) + float(shift_x), float(center[1]) + float(shift_y))
    contour = nucleus.get("contour", [])
    shifted_contour = []
    for x, y in contour:
        shifted_contour.append((float(x) + float(shift_x), float(y) + float(shift_y)))
    out["contour"] = shifted_contour
    return out


def _center_in_tile_border_zone(center: tuple[float, float], tile_w: int, tile_h: int, trim_px: float) -> bool:
    if trim_px <= 0.0:
        return False
    cx = float(center[0])
    cy = float(center[1])
    return (
        cx < trim_px
        or cy < trim_px
        or cx > (float(tile_w) - trim_px)
        or cy > (float(tile_h) - trim_px)
    )


def _nucleus_bbox(nucleus: dict) -> tuple[float, float, float, float]:
    contour = nucleus.get("contour", [])
    if contour:
        arr = np.asarray(contour, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] >= 1:
            x1 = float(np.min(arr[:, 0]))
            y1 = float(np.min(arr[:, 1]))
            x2 = float(np.max(arr[:, 0]))
            y2 = float(np.max(arr[:, 1]))
            return x1, y1, x2, y2

    center = nucleus.get("center", (0.0, 0.0))
    area = max(1.0, float(nucleus.get("area_px", 1.0)))
    radius = float(np.sqrt(area / np.pi))
    cx = float(center[0])
    cy = float(center[1])
    return cx - radius, cy - radius, cx + radius, cy + radius


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _nucleus_proxy_score(nucleus: dict) -> float:
    for key in ("score", "prob", "confidence"):
        value = nucleus.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return float(nucleus.get("area_px", 0.0))


def deduplicate_nuclei_by_bbox_iou(nuclei: list[dict], merge_iou_thresh: float = 0.3) -> list[dict]:
    if not nuclei:
        return nuclei
    thresh = float(np.clip(float(merge_iou_thresh), 0.0, 1.0))
    ranked = sorted(nuclei, key=_nucleus_proxy_score, reverse=True)
    kept: list[dict] = []
    kept_boxes: list[tuple[float, float, float, float]] = []
    for nucleus in ranked:
        box = _nucleus_bbox(nucleus)
        conflict = False
        for ref_box in kept_boxes:
            if _bbox_iou(box, ref_box) > thresh:
                conflict = True
                break
        if conflict:
            continue
        kept.append(nucleus)
        kept_boxes.append(box)
    return kept


def detect_nuclei_tiled_in_image(
    image_bgr: np.ndarray,
    average_cell_diameter_px: float | None = None,
    average_cell_bbox: Sequence[float] | None = None,
    enhancement_params: dict | None = None,
    tile_factor: float = 64.0,
    overlap_ratio: float = 0.25,
    tile_min_px: int = 512,
    tile_max_px: int = 2048,
    min_roi_cover_ratio: float = 0.05,
    roi_rect: Sequence[Sequence[float]] | Sequence[float] | None = None,
    roi_poly: Sequence[Sequence[Sequence[float]]] | Sequence[Sequence[float]] | None = None,
    merge_iou_thresh: float = 0.30,
    border_trim_enabled: bool = True,
    border_trim_factor: float = 1.0,
) -> tuple[list[dict], dict]:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Пустое изображение для sliced inference")

    image_h, image_w = image_bgr.shape[:2]
    avg_diameter = estimate_average_cell_diameter_px(
        average_cell_diameter_px=average_cell_diameter_px,
        average_cell_bbox=average_cell_bbox,
    )

    rois = normalize_rois_for_image((image_h, image_w), roi_rect=roi_rect, roi_poly=roi_poly)
    scale = float(get_detection_params().get("scale", 1.0))
    scale = float(max(1e-6, scale))
    image_w_infer = int(max(1, round(float(image_w) * scale)))
    image_h_infer = int(max(1, round(float(image_h) * scale)))

    overlap = float(np.clip(float(overlap_ratio), 0.0, 0.95))
    tile_factor_value = float(max(1.0, float(tile_factor)))
    tile_min = int(max(32, int(tile_min_px)))
    tile_max = int(max(tile_min, int(tile_max_px)))
    avg_diameter_infer = float(avg_diameter * scale)
    tile_size_infer = int(np.clip(round(tile_factor_value * avg_diameter_infer), tile_min, tile_max))
    stride_infer = int(max(1, round(float(tile_size_infer) * (1.0 - overlap))))
    tile_size_orig = int(max(1, round(float(tile_size_infer) / scale)))
    stride_orig = int(max(1, round(float(stride_infer) / scale)))

    trim_px = 0
    if border_trim_enabled:
        trim_px = int(max(0, round(float(border_trim_factor) * float(avg_diameter))))

    rois_infer = [_roi_to_infer_space(roi, scale) for roi in rois]
    all_candidates: list[dict] = []
    total_tiles = 0
    used_tiles = 0
    roi_reports = []

    for roi, roi_infer in zip(rois, rois_infer):
        tiles = _build_tiles_for_roi(
            roi_infer=roi_infer,
            image_w_infer=image_w_infer,
            image_h_infer=image_h_infer,
            tile_size_infer=tile_size_infer,
            stride_infer=stride_infer,
            min_roi_cover_ratio=float(max(0.0, min_roi_cover_ratio)),
        )
        total_tiles += len(tiles)

        for tile in tiles:
            x0_inf = int(tile["x"])
            y0_inf = int(tile["y"])
            x1_inf = int(tile["x"] + tile["w"])
            y1_inf = int(tile["y"] + tile["h"])

            x0 = int(max(0, np.floor(float(x0_inf) / scale)))
            y0 = int(max(0, np.floor(float(y0_inf) / scale)))
            x1 = int(min(image_w, np.ceil(float(x1_inf) / scale)))
            y1 = int(min(image_h, np.ceil(float(y1_inf) / scale)))
            if x1 <= x0 or y1 <= y0:
                continue

            tile_image = image_bgr[y0:y1, x0:x1]
            if tile_image.size == 0:
                continue

            used_tiles += 1
            local_nuclei = detect_nuclei_in_image(tile_image, enhancement_params=enhancement_params)
            for nucleus in local_nuclei:
                center = nucleus.get("center")
                if center is None:
                    continue
                if trim_px > 0 and not bool(tile.get("touches_outer", False)):
                    if _center_in_tile_border_zone(center, tile_image.shape[1], tile_image.shape[0], float(trim_px)):
                        continue
                shifted = _shift_nucleus_to_global(nucleus, x0, y0)
                shifted["tile_roi_id"] = roi.get("id")
                all_candidates.append(shifted)

        roi_reports.append(
            {
                "roi_id": roi.get("id"),
                "roi_kind": roi.get("kind"),
                "roi_area_px": float(roi.get("area_px", 0.0)),
                "roi_area_in_cells": _roi_area_in_cells(float(roi.get("area_px", 0.0)), avg_diameter),
                "tiles_total": len(tiles),
            }
        )

    in_roi_candidates: list[dict] = []
    for nucleus in all_candidates:
        center = nucleus.get("center")
        if center is None:
            continue
        point = (float(center[0]), float(center[1]))
        if _point_in_any_roi(point, rois):
            in_roi_candidates.append(nucleus)

    merged = deduplicate_nuclei_by_bbox_iou(in_roi_candidates, merge_iou_thresh=merge_iou_thresh)

    report = {
        "average_cell_diameter_px": float(avg_diameter),
        "scale": float(scale),
        "tile_factor": float(tile_factor_value),
        "overlap_ratio": float(overlap),
        "tile_size_px": int(tile_size_infer),
        "stride_px": int(stride_infer),
        "tile_size_orig_px": int(tile_size_orig),
        "stride_orig_px": int(stride_orig),
        "min_roi_cover_ratio": float(max(0.0, min_roi_cover_ratio)),
        "merge_iou_thresh": float(np.clip(float(merge_iou_thresh), 0.0, 1.0)),
        "border_trim_enabled": bool(border_trim_enabled),
        "border_trim_px": int(trim_px),
        "roi_count": len(rois),
        "tiles_total": int(total_tiles),
        "tiles_used": int(used_tiles),
        "candidates_before_roi_filter": int(len(all_candidates)),
        "candidates_in_roi": int(len(in_roi_candidates)),
        "final_count": int(len(merged)),
        "roi_stats": roi_reports,
    }
    return merged, report


def detect_nuclei_tiled(
    image_path: str,
    average_cell_diameter_px: float | None = None,
    average_cell_bbox: Sequence[float] | None = None,
    enhancement_params: dict | None = None,
    tile_factor: float = 64.0,
    overlap_ratio: float = 0.25,
    tile_min_px: int = 512,
    tile_max_px: int = 2048,
    min_roi_cover_ratio: float = 0.05,
    roi_rect: Sequence[Sequence[float]] | Sequence[float] | None = None,
    roi_poly: Sequence[Sequence[Sequence[float]]] | Sequence[Sequence[float]] | None = None,
    merge_iou_thresh: float = 0.30,
    border_trim_enabled: bool = True,
    border_trim_factor: float = 1.0,
) -> tuple[list[dict], dict]:
    image = load_image(image_path)
    return detect_nuclei_tiled_in_image(
        image_bgr=image,
        average_cell_diameter_px=average_cell_diameter_px,
        average_cell_bbox=average_cell_bbox,
        enhancement_params=enhancement_params,
        tile_factor=tile_factor,
        overlap_ratio=overlap_ratio,
        tile_min_px=tile_min_px,
        tile_max_px=tile_max_px,
        min_roi_cover_ratio=min_roi_cover_ratio,
        roi_rect=roi_rect,
        roi_poly=roi_poly,
        merge_iou_thresh=merge_iou_thresh,
        border_trim_enabled=border_trim_enabled,
        border_trim_factor=border_trim_factor,
    )


def _build_detection_diagnostics_template(area_small_px: float, area_large_px: float) -> dict:
    small = float(max(0.0, area_small_px))
    large = float(max(small, area_large_px))
    return {
        "backend": "",
        "model_candidates_before_nms": None,
        "after_nms_count": 0,
        "removed_by_min_area": 0,
        "removed_by_max_area": 0,
        "after_area_filters_count": 0,
        "purple_filter_enabled": False,
        "strict_purple_filter": True,
        "purple_checked": 0,
        "removed_by_purple_ratio": 0,
        "removed_by_purple_center": 0,
        "removed_by_purple_other": 0,
        "removed_by_purple_total": 0,
        "weak_color_marked": 0,
        "after_purple_count": 0,
        "final_count": 0,
        "small_area_threshold_px": small,
        "large_area_threshold_px": large,
        "final_area_stats": {},
    }


def _extract_model_candidate_count(details) -> int | None:
    if not isinstance(details, dict):
        return None

    direct_keys = (
        "num_candidates_before_nms",
        "n_candidates_before_nms",
        "candidates_before_nms",
        "before_nms_count",
    )
    for key in direct_keys:
        value = details.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue

    array_keys = (
        "points_before_nms",
        "points_raw",
        "candidates_raw",
    )
    for key in array_keys:
        value = details.get(key)
        if value is None:
            continue
        try:
            return int(len(value))
        except Exception:
            continue
    return None


def _summarize_nuclei_areas(
    nuclei: list[dict],
    small_area_px: float,
    large_area_px: float,
) -> dict:
    if not nuclei:
        return {
            "count": 0,
            "min": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "max": 0.0,
            "pct_small": 0.0,
            "pct_large": 0.0,
        }

    areas = np.asarray([float(x.get("area_px", 0.0)) for x in nuclei], dtype=np.float32)
    if areas.size == 0:
        return {
            "count": 0,
            "min": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "max": 0.0,
            "pct_small": 0.0,
            "pct_large": 0.0,
        }

    small_thr = float(max(0.0, small_area_px))
    large_thr = float(max(small_thr, large_area_px))
    return {
        "count": int(areas.size),
        "min": float(np.min(areas)),
        "median": float(np.median(areas)),
        "mean": float(np.mean(areas)),
        "max": float(np.max(areas)),
        "pct_small": float(np.mean(areas < small_thr)),
        "pct_large": float(np.mean(areas > large_thr)),
    }


def _filter_nuclei_by_purple_stain(
    nuclei: list[dict],
    image_bgr: np.ndarray,
    hue_min: int,
    hue_max: int,
    sat_min: int,
    val_max: int,
    min_ratio: float,
    require_center: bool,
    strict: bool = True,
) -> tuple[list[dict], dict]:
    stats = {
        "checked": 0,
        "failed_ratio": 0,
        "failed_center": 0,
        "failed_other": 0,
        "rejected": 0,
        "weak_marked": 0,
    }
    if not nuclei:
        return nuclei, stats

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)
    img_h, img_w = image_bgr.shape[:2]
    strict = bool(strict)

    kept: list[dict] = []
    for nucleus in nuclei:
        stats["checked"] += 1
        failure_reason = ""
        contour = nucleus.get("contour", [])
        if len(contour) < 3:
            failure_reason = "other"
        else:
            contour_arr = np.asarray(contour, dtype=np.float32).reshape((-1, 1, 2))
            contour_int = np.round(contour_arr).astype(np.int32)
            contour_int[:, 0, 0] = np.clip(contour_int[:, 0, 0], 0, img_w - 1)
            contour_int[:, 0, 1] = np.clip(contour_int[:, 0, 1], 0, img_h - 1)

            x, y, w, h = cv2.boundingRect(contour_int)
            if w <= 0 or h <= 0:
                failure_reason = "other"
            else:
                local_mask = np.zeros((h, w), dtype=np.uint8)
                shifted = contour_int.copy()
                shifted[:, 0, 0] -= x
                shifted[:, 0, 1] -= y
                cv2.drawContours(local_mask, [shifted], -1, 255, thickness=cv2.FILLED)

                inside = local_mask > 0
                area = int(np.count_nonzero(inside))
                if area == 0:
                    failure_reason = "other"
                else:
                    local_h = h_channel[y : y + h, x : x + w]
                    local_s = s_channel[y : y + h, x : x + w]
                    local_v = v_channel[y : y + h, x : x + w]

                    purple_mask = (
                        _hue_in_range(local_h, hue_min, hue_max)
                        & (local_s >= sat_min)
                        & (local_v <= val_max)
                    )
                    purple_ratio = float(np.count_nonzero(purple_mask & inside)) / float(area)
                    if purple_ratio < min_ratio:
                        failure_reason = "ratio"
                    elif require_center:
                        center = nucleus.get("center")
                        if center is None:
                            failure_reason = "center"
                        else:
                            cx = int(round(float(center[0])))
                            cy = int(round(float(center[1])))
                            cx = int(np.clip(cx, 0, img_w - 1))
                            cy = int(np.clip(cy, 0, img_h - 1))

                            center_h = int(h_channel[cy, cx])
                            center_s = int(s_channel[cy, cx])
                            center_v = int(v_channel[cy, cx])
                            center_is_purple = (
                                bool(_hue_in_range(np.asarray([[center_h]]), hue_min, hue_max)[0, 0])
                                and center_s >= sat_min
                                and center_v <= val_max
                            )
                            if not center_is_purple:
                                failure_reason = "center"

        if not failure_reason:
            kept.append(nucleus)
            continue

        if failure_reason == "ratio":
            stats["failed_ratio"] += 1
        elif failure_reason == "center":
            stats["failed_center"] += 1
        else:
            stats["failed_other"] += 1

        if strict:
            stats["rejected"] += 1
            continue

        marked = dict(nucleus)
        marked["weak_color"] = True
        marked["purple_filter_passed"] = False
        kept.append(marked)
        stats["weak_marked"] += 1

    if not strict:
        for idx, nucleus in enumerate(kept):
            if bool(nucleus.get("weak_color", False)):
                continue
            clean = dict(nucleus)
            clean["weak_color"] = False
            clean["purple_filter_passed"] = True
            kept[idx] = clean

    return kept, stats


def _hue_in_range(h_channel: np.ndarray, hue_min: int, hue_max: int) -> np.ndarray:
    if hue_min <= hue_max:
        return (h_channel >= hue_min) & (h_channel <= hue_max)
    return (h_channel >= hue_min) | (h_channel <= hue_max)


def _extract_nuclei_from_binary_mask(mask: np.ndarray) -> list[dict]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    nuclei: list[dict] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 20.0:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
        points = [(float(p[0][0]), float(p[0][1])) for p in contour]

        nuclei.append(
            {
                "center": (cx, cy),
                "contour": points,
                "area_px": area,
            }
        )

    nuclei.sort(key=lambda x: x["area_px"], reverse=True)
    return nuclei


def _extract_nuclei_from_label_image(labels: np.ndarray) -> list[dict]:
    nuclei: list[dict] = []
    if labels.ndim != 2:
        return nuclei

    label_ids = np.unique(labels)
    for label_id in label_ids:
        if label_id <= 0:
            continue
        mask = (labels == label_id).astype(np.uint8)
        area_px = float(mask.sum())
        if area_px < 5:
            continue

        moments = cv2.moments(mask, binaryImage=True)
        if moments["m00"] == 0:
            continue

        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        points = [(float(p[0][0]), float(p[0][1])) for p in contour]
        if len(points) < 3:
            continue

        nuclei.append(
            {
                "center": (cx, cy),
                "contour": points,
                "area_px": area_px,
            }
        )

    nuclei.sort(key=lambda x: x["area_px"], reverse=True)
    return nuclei


def _nuclei_to_binary_mask(nuclei: list[dict], target_hw: tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    for nucleus in nuclei:
        contour = nucleus.get("contour", [])
        if len(contour) < 3:
            continue
        arr = np.asarray(contour, dtype=np.int32).reshape((-1, 1, 2))
        cv2.drawContours(mask, [arr], contourIdx=-1, color=255, thickness=cv2.FILLED)
    return mask
