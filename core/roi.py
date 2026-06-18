from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .calibration import point_in_polygon, polygon_area_px


def _normalize_rect_roi(rect: Sequence[float], image_w: int, image_h: int, roi_id: str) -> dict | None:
    if len(rect) != 4:
        return None
    x = float(rect[0])
    y = float(rect[1])
    w = float(rect[2])
    h = float(rect[3])
    if w <= 0.0 or h <= 0.0:
        return None

    x1 = float(np.clip(x, 0.0, float(image_w)))
    y1 = float(np.clip(y, 0.0, float(image_h)))
    x2 = float(np.clip(x + w, 0.0, float(image_w)))
    y2 = float(np.clip(y + h, 0.0, float(image_h)))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None

    return {
        "id": roi_id,
        "kind": "rect",
        "rect": [x1, y1, x2 - x1, y2 - y1],
        "bbox": [x1, y1, x2, y2],
        "area_px": float((x2 - x1) * (y2 - y1)),
    }


def _normalize_poly_roi(points: Sequence[Sequence[float]], image_w: int, image_h: int, roi_id: str) -> dict | None:
    poly: list[tuple[float, float]] = []
    for p in points:
        if len(p) < 2:
            continue
        px = float(np.clip(float(p[0]), 0.0, float(image_w)))
        py = float(np.clip(float(p[1]), 0.0, float(image_h)))
        poly.append((px, py))

    if len(poly) < 3:
        return None
    area_px = polygon_area_px(poly)
    if area_px <= 0.0:
        return None

    arr = np.asarray(poly, dtype=np.float32)
    x1 = float(np.min(arr[:, 0]))
    y1 = float(np.min(arr[:, 1]))
    x2 = float(np.max(arr[:, 0]))
    y2 = float(np.max(arr[:, 1]))

    return {
        "id": roi_id,
        "kind": "poly",
        "points": poly,
        "bbox": [x1, y1, x2, y2],
        "area_px": float(area_px),
    }


def normalize_rois_for_image(
    image_hw: tuple[int, int],
    roi_rect: Sequence[Sequence[float]] | Sequence[float] | None = None,
    roi_poly: Sequence[Sequence[Sequence[float]]] | Sequence[Sequence[float]] | None = None,
) -> list[dict]:
    image_h, image_w = image_hw
    rois: list[dict] = []

    rect_items: list[Sequence[float]] = []
    if roi_rect is not None:
        if len(roi_rect) == 4 and isinstance(roi_rect[0], (int, float)):
            rect_items = [roi_rect]
        else:
            rect_items = list(roi_rect)

    for idx, rect in enumerate(rect_items, start=1):
        roi = _normalize_rect_roi(rect, image_w, image_h, f"rect_{idx}")
        if roi is not None:
            rois.append(roi)

    poly_items: list[Sequence[Sequence[float]]] = []
    if roi_poly is not None:
        if roi_poly and isinstance(roi_poly[0], (list, tuple)) and len(roi_poly[0]) >= 2 and isinstance(roi_poly[0][0], (int, float)):
            poly_items = [roi_poly]
        else:
            poly_items = list(roi_poly)

    for idx, poly in enumerate(poly_items, start=1):
        roi = _normalize_poly_roi(poly, image_w, image_h, f"poly_{idx}")
        if roi is not None:
            rois.append(roi)

    if rois:
        return rois

    full_roi = _normalize_rect_roi([0.0, 0.0, float(image_w), float(image_h)], image_w, image_h, "full_frame")
    if full_roi is None:
        raise ValueError("Не удалось сформировать ROI полного кадра")
    return [full_roi]


def _roi_to_infer_space(roi: dict, scale: float) -> dict:
    out = dict(roi)
    if roi.get("kind") == "rect":
        x, y, w, h = roi["rect"]
        out["rect"] = [x * scale, y * scale, w * scale, h * scale]
    if roi.get("kind") == "poly":
        points = [(float(x) * scale, float(y) * scale) for x, y in roi.get("points", [])]
        out["points"] = points

    bbox = roi.get("bbox", [0.0, 0.0, 0.0, 0.0])
    out["bbox"] = [float(bbox[0]) * scale, float(bbox[1]) * scale, float(bbox[2]) * scale, float(bbox[3]) * scale]
    return out


def _point_in_roi(point: tuple[float, float], roi: dict) -> bool:
    if roi.get("kind") == "rect":
        x, y, w, h = roi.get("rect", [0.0, 0.0, 0.0, 0.0])
        return (x <= point[0] <= x + w) and (y <= point[1] <= y + h)
    if roi.get("kind") == "poly":
        return point_in_polygon(point, roi.get("points", []))
    return False


def _point_in_any_roi(point: tuple[float, float], rois: Sequence[dict]) -> bool:
    for roi in rois:
        if _point_in_roi(point, roi):
            return True
    return False


def _roi_area_in_cells(area_px: float, average_cell_diameter_px: float) -> float:
    if average_cell_diameter_px <= 0.0:
        return 0.0
    cell_area = np.pi * (average_cell_diameter_px * 0.5) ** 2
    if cell_area <= 0.0:
        return 0.0
    return float(area_px / cell_area)


def count_nuclei_in_roi(nuclei: Iterable[dict], roi_points: Sequence[tuple[float, float]]) -> int:
    total = 0
    for nucleus in nuclei:
        center = nucleus.get("center")
        if center is None:
            continue
        if point_in_polygon((float(center[0]), float(center[1])), roi_points):
            total += 1
    return total


def build_roi_metrics(
    roi: dict,
    nuclei: Iterable[dict],
    pixels_per_mm: float | None,
) -> dict:
    points = roi.get("points", [])
    area_px = polygon_area_px(points)
    nuclei_count = count_nuclei_in_roi(nuclei, points)

    area_mm2: float | None = None
    density: float | None = None

    if pixels_per_mm is not None:
        if pixels_per_mm <= 0:
            raise ValueError("Коэффициент калибровки должен быть больше 0")
        area_mm2 = area_px / float(pixels_per_mm**2)
        if area_mm2 > 0:
            density = nuclei_count / area_mm2

    return {
        "ROI ID": roi.get("id"),
        "Тип": roi.get("type", "unknown"),
        "Площадь (мм²)": area_mm2,
        "Количество ядер": nuclei_count,
        "Плотность (ядра/мм²)": density,
    }
