from __future__ import annotations

from typing import Sequence

import numpy as np


def calibrate_scale(
    image_with_scalebar: np.ndarray,
    line_coords: tuple[tuple[float, float], tuple[float, float]],
    real_length_mm: float,
) -> float:
    """Return pixels per mm based on user-defined scale-bar line."""
    if image_with_scalebar is None or image_with_scalebar.size == 0:
        raise ValueError("Передано пустое изображение для калибровки")
    if real_length_mm <= 0:
        raise ValueError("Реальная длина должна быть больше 0 мм")

    (x1, y1), (x2, y2) = line_coords
    px_len = float(np.hypot(x2 - x1, y2 - y1))
    if px_len <= 0:
        raise ValueError("Длина линии калибровки должна быть больше 0 пикселей")

    return px_len / float(real_length_mm)


def polygon_area_px(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    arr = np.asarray(points, dtype=np.float64)
    x = arr[:, 0]
    y = arr[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def point_in_polygon(point: tuple[float, float], polygon: Sequence[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y)
        if intersects:
            denom = (yj - yi)
            if abs(denom) < 1e-12:
                j = i
                continue
            x_intersection = (xj - xi) * (y - yi) / denom + xi
            if x < x_intersection:
                inside = not inside
        j = i
    return inside
