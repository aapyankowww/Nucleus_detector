from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import point_in_polygon


@dataclass
class SquareRoiConfig:
    side_length_mm: float = 1.0
    center_x_mm: float | None = None
    center_y_mm: float | None = None


def create_square_roi_points(
    center_px: tuple[float, float],
    side_length_mm: float,
    pixels_per_mm: float,
    image_width_px: int,
    image_height_px: int,
) -> tuple[list[tuple[float, float]], float]:
    cx, cy = center_px
    half_side_px = side_length_mm * pixels_per_mm * 0.5
    half_side_px = max(1.0, half_side_px)

    x1 = max(0.0, cx - half_side_px)
    y1 = max(0.0, cy - half_side_px)
    x2 = min(float(image_width_px), cx + half_side_px)
    y2 = min(float(image_height_px), cy + half_side_px)

    actual_side_px = min(x2 - x1, y2 - y1)
    actual_side_px = max(1.0, actual_side_px)

    points: list[tuple[float, float]] = [
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2),
    ]
    return points, actual_side_px


def square_roi_metrics_for_nuclei(
    nuclei: list[dict],
    square_points: list[tuple[float, float]],
    pixels_per_mm: float,
) -> dict:
    count = 0
    for nucleus in nuclei:
        center = nucleus.get("center")
        if center is None:
            continue
        if point_in_polygon((float(center[0]), float(center[1])), square_points):
            count += 1

    area_mm2: float | None = None
    density: float | None = None

    if pixels_per_mm > 0.0 and len(square_points) >= 3:
        arr = np.asarray(square_points, dtype=np.float64)
        x = arr[:, 0]
        y = arr[:, 1]
        area_px = float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        area_mm2 = area_px / (pixels_per_mm * pixels_per_mm)
        if area_mm2 > 0.0:
            density = float(count) / area_mm2

    return {
        "count": count,
        "area_mm2": area_mm2,
        "density": density,
    }
