from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass
class LayerConfig:
    n_layers: int = 3
    auto_detect: bool = True
    boundary_y_mm: list[float] | None = None


@dataclass
class LayerResult:
    name: str
    y_start_mm: float
    y_end_mm: float
    nuclei_count: int
    area_mm2: float | None
    density: float | None
    mean_diameter_um: float | None
    median_diameter_um: float | None
    min_diameter_um: float | None
    max_diameter_um: float | None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "y_start_mm": self.y_start_mm,
            "y_end_mm": self.y_end_mm,
            "nuclei_count": self.nuclei_count,
            "area_mm2": self.area_mm2,
            "density": self.density,
            "mean_diameter_um": self.mean_diameter_um,
            "median_diameter_um": self.median_diameter_um,
            "min_diameter_um": self.min_diameter_um,
            "max_diameter_um": self.max_diameter_um,
        }


def compute_nuclei_diameter_um(nucleus: dict, pixels_per_mm: float) -> float:
    area_px = float(nucleus.get("area_px", 0.0))
    if area_px <= 0.0 or pixels_per_mm <= 0.0:
        return 0.0
    radius_px = np.sqrt(area_px / np.pi)
    return radius_px * 2.0 / pixels_per_mm * 1000.0


def detect_layer_boundaries_auto(
    nuclei: list[dict],
    image_height_px: int,
    pixels_per_mm: float,
    n_layers: int = 3,
    n_bins: int = 100,
    smooth_sigma: float = 2.0,
) -> list[float]:
    if not nuclei or image_height_px <= 0 or pixels_per_mm <= 0:
        return []

    y_centers_mm = []
    for n in nuclei:
        center = n.get("center")
        if center is not None:
            y_centers_mm.append(float(center[1]) / pixels_per_mm)
    if not y_centers_mm:
        return []

    y_centers_mm = np.asarray(y_centers_mm, dtype=np.float64)
    height_mm = float(image_height_px) / pixels_per_mm

    hist, bin_edges = np.histogram(y_centers_mm, bins=n_bins, range=(0.0, height_mm))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    smoothed = gaussian_filter1d(hist.astype(np.float64), sigma=smooth_sigma, mode="reflect")

    valleys = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] < smoothed[i - 1] and smoothed[i] < smoothed[i + 1]:
            valleys.append((bin_centers[i], smoothed[i]))

    valleys.sort(key=lambda x: x[1])

    n_boundaries = n_layers - 1
    selected = valleys[:n_boundaries]
    selected.sort(key=lambda x: x[0])

    return [v[0] for v in selected]


def segment_into_layers(
    nuclei: list[dict],
    image_height_px: int,
    pixels_per_mm: float,
    layer_config: LayerConfig,
) -> list[LayerResult]:
    if pixels_per_mm <= 0.0 or image_height_px <= 0:
        return []

    if layer_config.auto_detect:
        boundaries = detect_layer_boundaries_auto(
            nuclei, image_height_px, pixels_per_mm, n_layers=layer_config.n_layers
        )
    else:
        boundaries = list(layer_config.boundary_y_mm or [])

    full_height_mm = float(image_height_px) / pixels_per_mm
    y_edges = [0.0] + boundaries + [full_height_mm]

    results: list[LayerResult] = []
    for i in range(len(y_edges) - 1):
        y_start = y_edges[i]
        y_end = y_edges[i + 1]
        name = f"Layer_{i + 1}"

        layer_nuclei = []
        for n in nuclei:
            center = n.get("center")
            if center is not None:
                cy_mm = float(center[1]) / pixels_per_mm
                if y_start <= cy_mm < y_end:
                    layer_nuclei.append(n)

        count = len(layer_nuclei)
        area_mm2 = None
        density = None
        diameters = []
        for n in layer_nuclei:
            d = compute_nuclei_diameter_um(n, pixels_per_mm)
            if d > 0.0:
                diameters.append(d)

        if diameters:
            mean_d = float(np.mean(diameters))
            median_d = float(np.median(diameters))
            min_d = float(np.min(diameters))
            max_d = float(np.max(diameters))
        else:
            mean_d = None
            median_d = None
            min_d = None
            max_d = None
            area_mm2 = 0.0
            density = 0.0

        results.append(
            LayerResult(
                name=name,
                y_start_mm=y_start,
                y_end_mm=y_end,
                nuclei_count=count,
                area_mm2=area_mm2,
                density=density,
                mean_diameter_um=mean_d,
                median_diameter_um=median_d,
                min_diameter_um=min_d,
                max_diameter_um=max_d,
            )
        )

    return results


def format_layer_report_for_csv(layers: list[LayerResult], total: LayerResult | None = None) -> list[dict]:
    rows: list[dict] = []
    for layer in layers:
        rows.append(layer.to_dict())
    if total is not None:
        row = total.to_dict()
        row["name"] = "Total"
        rows.append(row)
    return rows
