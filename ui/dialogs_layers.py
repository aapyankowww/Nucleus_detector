from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import backend
from PyQt5.QtCore import Qt, QPointF, QLineF, QRectF
from PyQt5.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QScrollArea,
)

from ui.scenes import CalibrationView

LAYER_COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"]
LAYER_NAMES_RU = ["Слой 1", "Слой 2", "Слой 3", "Слой 4", "Слой 5"]


class _LayerBoundaryLine(QGraphicsLineItem):
    def __init__(self, x1: float, y: float, x2: float, parent=None):
        super().__init__(x1, y, x2, y, parent)
        self._y = y
        pen = QPen(QColor("#FFD700"), 2, Qt.DashLine)
        self.setPen(pen)
        self.setZValue(60)

    def set_y(self, y: float) -> None:
        self._y = y
        line = self.line()
        self.setLine(line.x1(), y, line.x2(), y)

    def get_y(self) -> float:
        return self._y


class LayerSetupDialog(QDialog):
    def __init__(
        self,
        image_path: str,
        enhancement_params: dict | None = None,
        detection_params: dict | None = None,
        pixels_per_mm: float | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Настройка слоёв ткани")
        self.resize(1300, 900)
        self.setWindowState(self.windowState() | Qt.WindowMaximized)

        self.image_path = image_path
        self.enhancement_params = backend.normalize_enhancement_params(enhancement_params)
        self.detection_params = dict(detection_params or backend.get_detection_params())
        self.pixels_per_mm = pixels_per_mm
        self._display_scale = 1.0

        self._image_bgr: np.ndarray | None = None
        self._display_image: np.ndarray | None = None
        self._nuclei: list[dict] = []

        self._boundary_lines: list[_LayerBoundaryLine] = []
        self._layer_regions: list[dict] = []
        self._auto_mode = True
        self._manual_line_index = 0

        self._build_ui()
        self._load_and_detect()
        self._update_info()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        top_row = QHBoxLayout()
        self.path_label = QLabel(f"Файл: {Path(self.image_path).name}")
        self.btn_zoom_in = QToolButton()
        self.btn_zoom_in.setText("+")
        self.btn_zoom_out = QToolButton()
        self.btn_zoom_out.setText("-")
        self.btn_zoom_fit = QToolButton()
        self.btn_zoom_fit.setText("По размеру")
        top_row.addWidget(self.path_label, 1)
        top_row.addWidget(self.btn_zoom_in)
        top_row.addWidget(self.btn_zoom_out)
        top_row.addWidget(self.btn_zoom_fit)
        layout.addLayout(top_row)

        body = QHBoxLayout()
        layout.addLayout(body, 1)

        left_panel = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setFixedWidth(300)
        left_widget.setLayout(left_panel)

        mode_group = QGroupBox("Режим определения слоёв")
        mode_layout = QVBoxLayout(mode_group)
        self.radio_auto = QRadioButton("Автоматический")
        self.radio_manual = QRadioButton("Ручной (кликните для границ)")
        self.radio_auto.setChecked(True)
        mode_layout.addWidget(self.radio_auto)
        mode_layout.addWidget(self.radio_manual)
        left_panel.addWidget(mode_group)

        self.info_group = QGroupBox("Границы слоёв")
        self.info_layout = QFormLayout(self.info_group)
        left_panel.addWidget(self.info_group)

        self.btn_run_auto = QPushButton("Запустить авто-определение")
        self.btn_run_auto.setMinimumHeight(40)
        left_panel.addWidget(self.btn_run_auto)

        left_panel.addStretch(1)
        body.addWidget(left_widget)

        self._scene = QGraphicsScene(self)
        self._view = CalibrationView(self._scene, self)
        self._view.setCursor(Qt.CrossCursor)
        self._view.viewport().setCursor(Qt.CrossCursor)
        body.addWidget(self._view, 1)

        self._coord_overlay = QLabel("x —   y —", self._view.viewport())
        self._coord_overlay.setObjectName("coord_overlay")
        self._coord_overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._coord_overlay.adjustSize()
        self._coord_overlay.move(8, 8)
        self._coord_overlay.show()
        self._coord_overlay.raise_()

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(button_box)

        self.btn_zoom_in.clicked.connect(self._view.zoom_in)
        self.btn_zoom_out.clicked.connect(self._view.zoom_out)
        self.btn_zoom_fit.clicked.connect(self._view.zoom_fit)
        self.radio_auto.toggled.connect(self._on_mode_changed)
        self.btn_run_auto.clicked.connect(self.auto_detect_layers)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

    def _load_and_detect(self) -> None:
        try:
            self._image_bgr = backend.load_image(self.image_path)
            display_img, self._display_scale = backend.load_display_image(self.image_path, max_side=2200)
            self._display_image = backend.apply_image_enhancement(display_img, self.enhancement_params)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))
            return

        pixmap = self._bgr_to_pixmap(self._display_image)
        self._scene.clear()
        self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(self._scene.itemsBoundingRect())
        self._view.zoom_fit()

        try:
            model_info = backend.get_loaded_model_info()
            custom_model_path = str(model_info.get("path")) if model_info else None
            self._nuclei = backend.detect_nuclei_in_image(
                self.image_path,
                enhancement_params=self.enhancement_params,
                detection_params=self.detection_params,
                custom_model_path=custom_model_path,
            )
        except Exception:
            self._nuclei = []

        self._show_nuclei_overlay()

        if self._auto_mode and self._nuclei:
            self.auto_detect_layers()

    def _show_nuclei_overlay(self) -> None:
        if not self._nuclei:
            return

        layer_colors_qt = [QColor(c) for c in LAYER_COLORS]
        layer_assignments = self._assign_nuclei_to_layers()

        for nuc, layer_idx in zip(self._nuclei, layer_assignments):
            contour = nuc.get("contour", [])
            if len(contour) < 3:
                continue
            poly = QPolygonF([QPointF(float(x) / self._display_scale, float(y) / self._display_scale) for x, y in contour])
            color = layer_colors_qt[layer_idx % len(layer_colors_qt)]
            item = QGraphicsPolygonItem(poly)
            item.setPen(QPen(color, 1))
            item.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 60)))
            item.setZValue(10)
            self._scene.addItem(item)

    def _assign_nuclei_to_layers(self) -> list[int]:
        if not self._nuclei:
            return []

        boundaries_mm = self._get_boundary_positions_mm()
        if not boundaries_mm or self.pixels_per_mm is None or self.pixels_per_mm <= 0:
            return [0] * len(self._nuclei)

        boundaries_px = [b * self.pixels_per_mm for b in boundaries_mm]
        result = []
        for nuc in self._nuclei:
            center = nuc.get("center")
            if center is None:
                result.append(0)
                continue
            cy = float(center[1])
            layer = 0
            for i, b_px in enumerate(boundaries_px):
                if cy >= b_px:
                    layer = i + 1
            result.append(layer)
        return result

    def _get_boundary_positions_mm(self) -> list[float]:
        if self._boundary_lines:
            boundaries = []
            for line in self._boundary_lines:
                y_mm = line.get_y() * self._display_scale
                if self.pixels_per_mm is not None and self.pixels_per_mm > 0:
                    y_mm = y_mm / self.pixels_per_mm
                boundaries.append(y_mm)
            return sorted(boundaries)
        return []

    def auto_detect_layers(self) -> bool:
        if not self._nuclei:
            return False

        centers_y = []
        for nuc in self._nuclei:
            center = nuc.get("center")
            if center is not None:
                centers_y.append(float(center[1]))
        if len(centers_y) < 20:
            return False

        centers_y = np.array(centers_y)
        y_min, y_max = float(centers_y.min()), float(centers_y.max())
        if y_max - y_min < 10:
            return False

        hist, edges = np.histogram(centers_y, bins=20)
        hist_smooth = np.convolve(hist, [0.2, 0.6, 0.2], mode="same")

        valleys = []
        for i in range(1, len(hist_smooth) - 1):
            if hist_smooth[i] < hist_smooth[i - 1] and hist_smooth[i] < hist_smooth[i + 1]:
                valleys.append((edges[i] + edges[i + 1]) / 2.0)

        num_layers = min(len(valleys) + 1, 5)
        if num_layers < 2:
            mid = (y_min + y_max) / 2.0
            valleys = [mid]
            num_layers = 2

        valleys = sorted(valleys)
        valleys = [v for v in valleys if y_min < v < y_max]

        self._clear_boundary_lines()
        image_rect = self._scene.sceneRect()
        for v in valleys:
            y_display = v / self._display_scale
            line = _LayerBoundaryLine(image_rect.left(), y_display, image_rect.right())
            self._scene.addItem(line)
            self._boundary_lines.append(line)

        self._show_nuclei_overlay()
        self._update_info()
        return True

    def enable_manual_mode(self, enabled: bool) -> None:
        self._auto_mode = not enabled
        if enabled:
            self._manual_line_index = 0
        self._update_info()

    def _on_mode_changed(self) -> None:
        is_auto = self.radio_auto.isChecked()
        self._auto_mode = is_auto
        self.btn_run_auto.setEnabled(is_auto)
        if not is_auto:
            self._manual_line_index = 0
            QMessageBox.information(
                self,
                "Ручной режим",
                "Кликните на изображении, чтобы установить границы слоёв. "
                "Первый клик — верхняя граница, второй — нижняя.",
            )

    def _clear_boundary_lines(self) -> None:
        for line in self._boundary_lines:
            self._scene.removeItem(line)
        self._boundary_lines.clear()

    def _update_info(self) -> None:
        while self.info_layout.count():
            item = self.info_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        boundaries = self._get_boundary_positions_mm()
        if not boundaries:
            self.info_layout.addRow(QLabel("Границы не определены"))
            return

        for i, b_mm in enumerate(boundaries):
            name = LAYER_NAMES_RU[i] if i < len(LAYER_NAMES_RU) else f"Слой {i + 1}"
            color = LAYER_COLORS[i % len(LAYER_COLORS)]
            lbl = QLabel(f"{name}: {b_mm:.4f} мм")
            lbl.setStyleSheet(f"color: {color}; font-weight: bold;")
            self.info_layout.addRow(lbl)

        n_layers = len(boundaries) + 1
        total_lbl = QLabel(f"Всего слоёв: {n_layers}")
        total_lbl.setStyleSheet("font-weight: bold;")
        self.info_layout.addRow(total_lbl)

    def get_layer_config(self) -> dict:
        boundaries_mm = self._get_boundary_positions_mm()
        return {
            "boundaries_mm": boundaries_mm,
            "num_layers": len(boundaries_mm) + 1,
            "pixels_per_mm": self.pixels_per_mm,
            "nuclei_per_layer": self._count_nuclei_per_layer(boundaries_mm),
        }

    def _count_nuclei_per_layer(self, boundaries_mm: list[float]) -> list[int]:
        if not self._nuclei or not boundaries_mm:
            return [len(self._nuclei)]

        boundaries_px = [b * self.pixels_per_mm for b in boundaries_mm] if self.pixels_per_mm else []
        counts = [0] * (len(boundaries_px) + 1)
        for nuc in self._nuclei:
            center = nuc.get("center")
            if center is None:
                counts[0] += 1
                continue
            cy = float(center[1])
            layer = 0
            for i, b_px in enumerate(boundaries_px):
                if cy >= b_px:
                    layer = i + 1
            if layer < len(counts):
                counts[layer] += 1
        return counts

    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)

    def _bgr_to_pixmap(self, image_bgr) -> QPixmap:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    def accept(self) -> None:
        super().accept()
