from __future__ import annotations

import copy
import os
from pathlib import Path

import cv2
import numpy as np
import backend
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsScene,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QSpinBox,
    QGroupBox,
)

from ui.scenes import (
    CalibrationScene,
    CalibrationView,
    CellSelectionScene,
    ImageScene,
    ToolMode,
)
from ui.workers import (
    enhancement_to_slider_values,
    slider_values_to_enhancement,
)

PREPROCESS_MODE_LABELS_RU = {
    "rgb": "Исходные цвета (RGB)",
}
PREPROCESS_MODE_KEYS_BY_LABEL_RU = {v: k for k, v in PREPROCESS_MODE_LABELS_RU.items()}

DETECTOR_BACKEND_LABELS_RU = {
    "stardist": "StarDist (классический вариант)",
}
DETECTOR_BACKEND_KEYS_BY_LABEL_RU = {v: k for k, v in DETECTOR_BACKEND_LABELS_RU.items()}


BATCH_MODE_FULL = "full_frame"
BATCH_MODE_RECT = "rectangles"
BATCH_MODE_POLY = "polygons"

BATCH_MODE_LABELS_RU = {
    BATCH_MODE_FULL: "По всему кадру",
    BATCH_MODE_RECT: "По прямоугольникам",
    BATCH_MODE_POLY: "По полигонам",
}


def preprocess_mode_to_russian(mode_key: str) -> str:
    return PREPROCESS_MODE_LABELS_RU.get(mode_key, mode_key)


class CellSelectionDialog(QDialog):
    def __init__(
        self,
        image_path: str,
        enhancement_params: dict | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Выбор средней клетки")
        self.resize(1100, 820)
        self.setWindowState(self.windowState() | Qt.WindowMaximized)

        self.image_path = image_path
        self.enhancement_params = backend.normalize_enhancement_params(enhancement_params)
        self.image_bgr = None
        self.selected_radius_px: float | None = None
        self.selected_diameter_px: float | None = None
        self.selected_center_xy: tuple[float, float] | None = None
        self.selected_preset: str = "точный"

        layout = QVBoxLayout(self)

        top_row = QHBoxLayout()
        self.btn_clear_circle = QPushButton("Сбросить круг")
        self.btn_zoom_in = QToolButton()
        self.btn_zoom_in.setText("+")
        self.btn_zoom_out = QToolButton()
        self.btn_zoom_out.setText("-")
        self.btn_zoom_fit = QToolButton()
        self.btn_zoom_fit.setText("По размеру")
        self.path_label = QLabel("Файл: -")
        top_row.addWidget(self.btn_clear_circle)
        top_row.addWidget(self.btn_zoom_in)
        top_row.addWidget(self.btn_zoom_out)
        top_row.addWidget(self.btn_zoom_fit)
        top_row.addWidget(self.path_label, 1)
        layout.addLayout(top_row)

        self.scene_cell = CellSelectionScene(self)
        self.view_cell = CalibrationView(self.scene_cell, self)
        layout.addWidget(self.view_cell, 1)

        form = QFormLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["точный", "чувствительный"])
        self.diameter_label = QLabel("0.0 px")
        self.area_label = QLabel("0.0 px²")
        form.addRow("Режим подбора:", self.preset_combo)
        form.addRow("Диаметр выбранной клетки:", self.diameter_label)
        form.addRow("Площадь выбранной клетки:", self.area_label)
        layout.addLayout(form)

        hint = QLabel(
            "Выделите среднюю по размеру клетку (ядро): "
            "нажмите на один край ядра и протяните до противоположного края. "
            "Колесо мыши, + и - изменяют масштаб для точного выделения."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(self.button_box)

        self.btn_clear_circle.clicked.connect(self.scene_cell.clear_circle)
        self.btn_zoom_in.clicked.connect(self.view_cell.zoom_in)
        self.btn_zoom_out.clicked.connect(self.view_cell.zoom_out)
        self.btn_zoom_fit.clicked.connect(self.view_cell.zoom_fit)
        self.scene_cell.circle_updated.connect(self._on_circle_updated)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self._load_image(image_path)

    def _load_image(self, image_path: str) -> None:
        try:
            self.image_bgr = backend.load_image(image_path)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))
            return
        self.path_label.setText(f"Файл: {Path(image_path).name}")
        preview = backend.apply_image_enhancement(self.image_bgr, self.enhancement_params)
        self.scene_cell.set_image_pixmap(self._bgr_to_pixmap(preview))
        self.view_cell.zoom_100()

    def _bgr_to_pixmap(self, image_bgr) -> QPixmap:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    def _on_circle_updated(self, diameter_px: float, area_px: float) -> None:
        self.diameter_label.setText(f"{diameter_px:.1f} px")
        self.area_label.setText(f"{area_px:.1f} px²")

    def accept(self) -> None:
        center_xy = self.scene_cell.selected_center_xy()
        radius = self.scene_cell.selected_radius_px()
        if center_xy is None or radius is None or radius <= 0.0:
            QMessageBox.warning(self, "Нет выделения", "Сначала выделите среднюю клетку кругом")
            return
        self.selected_center_xy = (float(center_xy[0]), float(center_xy[1]))
        self.selected_radius_px = float(radius)
        self.selected_diameter_px = float(radius * 2.0)
        self.selected_preset = str(self.preset_combo.currentText()).strip().lower()
        super().accept()


class ScaleCalibrationDialog(QDialog):
    def __init__(self, parent: QWidget | None = None, initial_path: str | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Калибровка масштаба")
        self.resize(1200, 820)
        self.setWindowState(self.windowState() | Qt.WindowMaximized)

        self.image_path: str | None = None
        self.image_bgr = None
        self.pixels_per_mm: float | None = None
        self.line_coords_result: tuple[tuple[float, float], tuple[float, float]] | None = None
        self.real_length_mm: float | None = None

        layout = QVBoxLayout(self)

        top_row = QHBoxLayout()
        self.btn_open_scale_image = QPushButton("Открыть фото линейки")
        self.btn_clear_line = QPushButton("Сбросить линию")
        self.btn_zoom_in = QToolButton()
        self.btn_zoom_in.setText("+")
        self.btn_zoom_out = QToolButton()
        self.btn_zoom_out.setText("-")
        self.btn_zoom_fit = QToolButton()
        self.btn_zoom_fit.setText("По размеру")
        self.path_label = QLabel("Файл: не выбран")
        top_row.addWidget(self.btn_open_scale_image)
        top_row.addWidget(self.btn_clear_line)
        top_row.addWidget(self.btn_zoom_in)
        top_row.addWidget(self.btn_zoom_out)
        top_row.addWidget(self.btn_zoom_fit)
        top_row.addWidget(self.path_label, 1)
        layout.addLayout(top_row)

        self.scene_cal = CalibrationScene(self)
        self.view_cal = CalibrationView(self.scene_cal, self)
        layout.addWidget(self.view_cal, 1)

        form_row = QFormLayout()
        self.mm_spin = QDoubleSpinBox()
        self.mm_spin.setDecimals(4)
        self.mm_spin.setRange(0.0001, 1_000_000.0)
        self.mm_spin.setValue(1.0)
        self.px_len_label = QLabel("0.00 px")
        self.ppm_preview_label = QLabel("-")
        form_row.addRow("Реальная длина (мм):", self.mm_spin)
        form_row.addRow("Длина линии (px):", self.px_len_label)
        form_row.addRow("Коэффициент (px/mm):", self.ppm_preview_label)
        layout.addLayout(form_row)

        help_label = QLabel(
            "ЛКМ: задать отрезок. Колесо мыши / + / -: масштаб. "
            "Линия и прицел инвертируются относительно фона."
        )
        layout.addWidget(help_label)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(self.button_box)

        self.btn_open_scale_image.clicked.connect(self.open_scale_image)
        self.btn_clear_line.clicked.connect(self.scene_cal.clear_line)
        self.btn_zoom_in.clicked.connect(self.view_cal.zoom_in)
        self.btn_zoom_out.clicked.connect(self.view_cal.zoom_out)
        self.btn_zoom_fit.clicked.connect(self.view_cal.zoom_fit)
        self.scene_cal.line_updated.connect(self._on_line_updated)
        self.mm_spin.valueChanged.connect(self._update_preview)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        if initial_path and os.path.exists(initial_path):
            self.load_scale_image(initial_path)

    def open_scale_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Открыть фото линейки",
            "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg)",
        )
        if not path:
            return
        self.load_scale_image(path)

    def load_scale_image(self, path: str) -> None:
        try:
            image = backend.load_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))
            return

        self.image_path = path
        self.image_bgr = image
        self.path_label.setText(f"Файл: {Path(path).name}")
        self.scene_cal.set_image_pixmap(self._bgr_to_pixmap(image))
        self.view_cal.zoom_100()
        self._on_line_updated(0.0)

    def _bgr_to_pixmap(self, image_bgr) -> QPixmap:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    def _on_line_updated(self, px_length: float) -> None:
        self.px_len_label.setText(f"{px_length:.2f} px")
        self._update_preview()

    def _update_preview(self) -> None:
        if self.image_bgr is None:
            self.ppm_preview_label.setText("-")
            return
        line_coords = self.scene_cal.line_coords()
        if line_coords is None:
            self.ppm_preview_label.setText("-")
            return
        try:
            ppm = backend.calibrate_scale(self.image_bgr, line_coords, float(self.mm_spin.value()))
        except Exception:
            self.ppm_preview_label.setText("-")
            return
        self.ppm_preview_label.setText(f"{ppm:.4f}")

    def accept(self) -> None:
        if self.image_bgr is None:
            QMessageBox.warning(self, "Нет изображения", "Сначала откройте фото линейки")
            return
        line_coords = self.scene_cal.line_coords()
        if line_coords is None:
            QMessageBox.warning(self, "Нет линии", "Выберите отрезок по масштабной линейке")
            return

        real_mm = float(self.mm_spin.value())
        try:
            ppm = backend.calibrate_scale(self.image_bgr, line_coords, real_mm)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка калибровки", str(exc))
            return

        self.line_coords_result = line_coords
        self.real_length_mm = real_mm
        self.pixels_per_mm = ppm
        super().accept()


class DetectionParamsDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Параметры детекции ядер")
        self.resize(560, 640)

        self._base_params = backend.get_detection_params()
        self.params_result: dict | None = None
        detector_backends = backend.get_detector_backends()
        model_names = backend.get_pretrained_model_names()
        preprocess_modes = backend.get_preprocess_modes()

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.detector_backend_combo = QComboBox()
        for backend_name in detector_backends:
            self.detector_backend_combo.addItem(
                DETECTOR_BACKEND_LABELS_RU.get(backend_name, backend_name),
                backend_name,
            )
        current_backend = str(self._base_params.get("detector_backend", "stardist"))
        backend_index = self.detector_backend_combo.findData(current_backend)
        if backend_index < 0:
            self.detector_backend_combo.addItem(
                DETECTOR_BACKEND_LABELS_RU.get(current_backend, current_backend),
                current_backend,
            )
            backend_index = self.detector_backend_combo.count() - 1
        self.detector_backend_combo.setCurrentIndex(backend_index)
        form.addRow("Встроенная модель:", self.detector_backend_combo)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        for model_name in model_names:
            self.model_combo.addItem(model_name)
        current_model = str(self._base_params.get("model_name", "2D_versatile_he"))
        if self.model_combo.findText(current_model) < 0:
            self.model_combo.addItem(current_model)
        self.model_combo.setCurrentText(current_model)
        form.addRow("Модель StarDist:", self.model_combo)

        self.prob_spin = QDoubleSpinBox()
        self.prob_spin.setDecimals(3)
        self.prob_spin.setRange(0.0, 1.0)
        self.prob_spin.setSingleStep(0.01)
        self.prob_spin.setValue(float(self._base_params.get("prob_thresh", 0.15)))
        form.addRow("Порог уверенности:", self.prob_spin)

        self.nms_spin = QDoubleSpinBox()
        self.nms_spin.setDecimals(3)
        self.nms_spin.setRange(0.0, 1.0)
        self.nms_spin.setSingleStep(0.01)
        self.nms_spin.setValue(float(self._base_params.get("nms_thresh", 0.55)))
        form.addRow("Порог разделения соседних ядер:", self.nms_spin)

        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setDecimals(3)
        self.scale_spin.setRange(0.2, 3.0)
        self.scale_spin.setSingleStep(0.05)
        self.scale_spin.setValue(float(self._base_params.get("scale", 1.0)))
        form.addRow("Масштаб для сети:", self.scale_spin)

        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(0, 100000)
        self.min_area_spin.setValue(int(self._base_params.get("min_area_px", 18)))
        form.addRow("Мин. площадь ядра (пикс):", self.min_area_spin)

        self.max_area_spin = QSpinBox()
        self.max_area_spin.setRange(0, 500000)
        self.max_area_spin.setSpecialValueText("без ограничения")
        self.max_area_spin.setValue(int(self._base_params.get("max_area_px", 0)))
        form.addRow("Макс. площадь ядра (пикс):", self.max_area_spin)

        self.tiles_x_spin = QSpinBox()
        self.tiles_x_spin.setRange(1, 12)
        self.tiles_x_spin.setValue(int(self._base_params.get("n_tiles_x", 3)))
        form.addRow("Тайлы по горизонтали:", self.tiles_x_spin)

        self.tiles_y_spin = QSpinBox()
        self.tiles_y_spin.setRange(1, 12)
        self.tiles_y_spin.setValue(int(self._base_params.get("n_tiles_y", 3)))
        form.addRow("Тайлы по вертикали:", self.tiles_y_spin)

        self.preprocess_combo = QComboBox()
        for mode in preprocess_modes:
            self.preprocess_combo.addItem(preprocess_mode_to_russian(mode), mode)
        current_mode_key = str(self._base_params.get("preprocess_mode", "rgb"))
        current_mode_index = self.preprocess_combo.findData(current_mode_key)
        if current_mode_index < 0:
            self.preprocess_combo.addItem(
                preprocess_mode_to_russian(current_mode_key),
                current_mode_key,
            )
            current_mode_index = self.preprocess_combo.count() - 1
        self.preprocess_combo.setCurrentIndex(current_mode_index)
        form.addRow("Подготовка изображения:", self.preprocess_combo)

        self.upscale_spin = QDoubleSpinBox()
        self.upscale_spin.setDecimals(2)
        self.upscale_spin.setRange(1.0, 4.0)
        self.upscale_spin.setSingleStep(0.10)
        self.upscale_spin.setValue(float(self._base_params.get("upscale_factor", 1.5)))
        form.addRow("Upscale перед инференсом:", self.upscale_spin)

        self.stain_norm_check = QCheckBox("Нормализация окраски (Reinhard)")
        self.stain_norm_check.setChecked(bool(self._base_params.get("stain_norm_enabled", True)))
        form.addRow(self.stain_norm_check)

        self.norm_low_spin = QDoubleSpinBox()
        self.norm_low_spin.setDecimals(2)
        self.norm_low_spin.setRange(0.0, 99.0)
        self.norm_low_spin.setSingleStep(0.1)
        self.norm_low_spin.setValue(float(self._base_params.get("norm_p_low", 1.0)))
        form.addRow("Нижний перцентиль нормализации:", self.norm_low_spin)

        self.norm_high_spin = QDoubleSpinBox()
        self.norm_high_spin.setDecimals(2)
        self.norm_high_spin.setRange(1.0, 100.0)
        self.norm_high_spin.setSingleStep(0.1)
        self.norm_high_spin.setValue(float(self._base_params.get("norm_p_high", 99.8)))
        form.addRow("Верхний перцентиль нормализации:", self.norm_high_spin)

        self.purple_filter_check = QCheckBox(
            "Фильтровать объекты по окраске гематоксилином (фиолетовые ядра)"
        )
        self.purple_filter_check.setChecked(bool(self._base_params.get("purple_filter_enabled", True)))
        form.addRow(self.purple_filter_check)

        self.center_purple_check = QCheckBox("Требовать окраску в центре ядра")
        self.center_purple_check.setChecked(bool(self._base_params.get("require_center_purple", True)))
        form.addRow(self.center_purple_check)

        self.hue_min_spin = QSpinBox()
        self.hue_min_spin.setRange(0, 179)
        self.hue_min_spin.setValue(int(self._base_params.get("purple_h_min", 120)))
        form.addRow("Мин. оттенок фиолетового:", self.hue_min_spin)

        self.hue_max_spin = QSpinBox()
        self.hue_max_spin.setRange(0, 179)
        self.hue_max_spin.setValue(int(self._base_params.get("purple_h_max", 170)))
        form.addRow("Макс. оттенок фиолетового:", self.hue_max_spin)

        self.sat_min_spin = QSpinBox()
        self.sat_min_spin.setRange(0, 255)
        self.sat_min_spin.setValue(int(self._base_params.get("purple_s_min", 45)))
        form.addRow("Мин. насыщенность:", self.sat_min_spin)

        self.val_max_spin = QSpinBox()
        self.val_max_spin.setRange(0, 255)
        self.val_max_spin.setValue(int(self._base_params.get("purple_v_max", 200)))
        form.addRow("Макс. яркость:", self.val_max_spin)

        self.min_ratio_spin = QDoubleSpinBox()
        self.min_ratio_spin.setDecimals(3)
        self.min_ratio_spin.setRange(0.0, 1.0)
        self.min_ratio_spin.setSingleStep(0.01)
        self.min_ratio_spin.setValue(float(self._base_params.get("min_purple_ratio", 0.20)))
        form.addRow("Мин. доля окрашенных пикселей в ядре:", self.min_ratio_spin)

        layout.addLayout(form)

        preset_row = QHBoxLayout()
        self.btn_preset_precise = QPushButton("Точный")
        self.btn_preset_sensitive = QPushButton("Чувствительный")
        preset_row.addWidget(QLabel("Пресет:"))
        preset_row.addWidget(self.btn_preset_precise)
        preset_row.addWidget(self.btn_preset_sensitive)
        preset_row.addStretch(1)
        layout.addLayout(preset_row)

        note = QLabel(
            "Точный — меньше ложных срабатываний. "
            "Чувствительный — находит больше ядер, но может добавить лишние объекты."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.purple_filter_check.toggled.connect(self._toggle_purple_controls)
        self.detector_backend_combo.currentIndexChanged.connect(self._toggle_detector_controls)
        self.btn_preset_precise.clicked.connect(self._apply_precision_preset)
        self.btn_preset_sensitive.clicked.connect(self._apply_sensitive_preset)
        self._toggle_purple_controls(self.purple_filter_check.isChecked())
        self._toggle_detector_controls()

    def _toggle_purple_controls(self, enabled: bool) -> None:
        for widget in [
            self.center_purple_check,
            self.hue_min_spin,
            self.hue_max_spin,
            self.sat_min_spin,
            self.val_max_spin,
            self.min_ratio_spin,
        ]:
            widget.setEnabled(enabled)

    def _toggle_detector_controls(self, *_args) -> None:
        backend_name = self.detector_backend_combo.currentData()
        if backend_name is None:
            backend_name = DETECTOR_BACKEND_KEYS_BY_LABEL_RU.get(
                self.detector_backend_combo.currentText().strip(),
                "stardist",
            )
        use_stardist = str(backend_name).strip().lower() == "stardist"

        for widget in [
            self.model_combo,
            self.prob_spin,
            self.nms_spin,
            self.scale_spin,
            self.tiles_x_spin,
            self.tiles_y_spin,
            self.preprocess_combo,
            self.norm_low_spin,
            self.norm_high_spin,
        ]:
            widget.setEnabled(use_stardist)

        self.upscale_spin.setEnabled(use_stardist)
        self.stain_norm_check.setEnabled(use_stardist)

    def _apply_precision_preset(self) -> None:
        self.model_combo.setCurrentText("2D_versatile_he")
        self.prob_spin.setValue(0.15)
        self.nms_spin.setValue(0.55)
        self.scale_spin.setValue(1.0)
        self.min_area_spin.setValue(18)
        self.max_area_spin.setValue(0)
        self.tiles_x_spin.setValue(3)
        self.tiles_y_spin.setValue(3)
        self.preprocess_combo.setCurrentIndex(
            max(0, self.preprocess_combo.findData("rgb"))
        )
        self.norm_low_spin.setValue(1.0)
        self.norm_high_spin.setValue(99.8)
        self.purple_filter_check.setChecked(True)
        self.center_purple_check.setChecked(True)
        self.hue_min_spin.setValue(120)
        self.hue_max_spin.setValue(170)
        self.sat_min_spin.setValue(42)
        self.val_max_spin.setValue(200)
        self.min_ratio_spin.setValue(0.20)
        self.upscale_spin.setValue(1.5)
        self.stain_norm_check.setChecked(True)

    def _apply_sensitive_preset(self) -> None:
        self.model_combo.setCurrentText("2D_versatile_he")
        self.prob_spin.setValue(0.15)
        self.nms_spin.setValue(0.29)
        self.scale_spin.setValue(1.0)
        self.min_area_spin.setValue(10)
        self.max_area_spin.setValue(0)
        self.tiles_x_spin.setValue(3)
        self.tiles_y_spin.setValue(3)
        self.preprocess_combo.setCurrentIndex(
            max(0, self.preprocess_combo.findData("rgb"))
        )
        self.norm_low_spin.setValue(1.0)
        self.norm_high_spin.setValue(99.8)
        self.purple_filter_check.setChecked(True)
        self.center_purple_check.setChecked(True)
        self.hue_min_spin.setValue(118)
        self.hue_max_spin.setValue(172)
        self.sat_min_spin.setValue(30)
        self.val_max_spin.setValue(215)
        self.min_ratio_spin.setValue(0.12)
        self.upscale_spin.setValue(2.05)
        self.stain_norm_check.setChecked(True)

    def _collect_params(self) -> dict:
        detector_backend = self.detector_backend_combo.currentData()
        if detector_backend is None:
            detector_backend = DETECTOR_BACKEND_KEYS_BY_LABEL_RU.get(
                self.detector_backend_combo.currentText().strip(),
                "stardist",
            )
        preprocess_mode = self.preprocess_combo.currentData()
        if preprocess_mode is None:
            preprocess_mode = PREPROCESS_MODE_KEYS_BY_LABEL_RU.get(
                self.preprocess_combo.currentText().strip(),
                "rgb",
            )
        return {
            "detector_backend": str(detector_backend).strip(),
            "model_name": str(self.model_combo.currentText()).strip() or "2D_versatile_he",
            "prob_thresh": float(self.prob_spin.value()),
            "nms_thresh": float(self.nms_spin.value()),
            "scale": float(self.scale_spin.value()),
            "min_area_px": int(self.min_area_spin.value()),
            "max_area_px": int(self.max_area_spin.value()),
            "n_tiles_x": int(self.tiles_x_spin.value()),
            "n_tiles_y": int(self.tiles_y_spin.value()),
            "preprocess_mode": str(preprocess_mode).strip(),
            "norm_p_low": float(self.norm_low_spin.value()),
            "norm_p_high": float(self.norm_high_spin.value()),
            "purple_filter_enabled": bool(self.purple_filter_check.isChecked()),
            "purple_h_min": int(self.hue_min_spin.value()),
            "purple_h_max": int(self.hue_max_spin.value()),
            "purple_s_min": int(self.sat_min_spin.value()),
            "purple_v_max": int(self.val_max_spin.value()),
            "min_purple_ratio": float(self.min_ratio_spin.value()),
            "require_center_purple": bool(self.center_purple_check.isChecked()),
            "upscale_factor": float(self.upscale_spin.value()),
            "stain_norm_enabled": bool(self.stain_norm_check.isChecked()),
        }

    def accept(self) -> None:
        params = self._collect_params()
        try:
            backend.set_detection_params(params)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка параметров", str(exc))
            return
        self.params_result = params
        super().accept()


class ColorTuningDialog(QDialog):
    def __init__(
        self,
        image_path: str,
        initial_params: dict | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Настройка цветокоррекции")
        self.resize(1180, 860)

        self.image_path = image_path
        self.result_params: dict | None = None
        self._base_display_image: np.ndarray | None = None
        self._current_params = backend.normalize_enhancement_params(initial_params)
        self._preview_initialized = False

        layout = QVBoxLayout(self)

        top_row = QHBoxLayout()
        self.path_label = QLabel("Файл: -")
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

        self.scene_preview = QGraphicsScene(self)
        self.view_preview = CalibrationView(self.scene_preview, self)
        layout.addWidget(self.view_preview, 1)

        sliders_group = QGroupBox("Параметры цветокоррекции")
        sliders_layout = QFormLayout(sliders_group)

        self.slider_saturation = QSlider(Qt.Horizontal)
        self.slider_saturation.setRange(0, 300)
        self.slider_brightness = QSlider(Qt.Horizontal)
        self.slider_brightness.setRange(-100, 100)
        self.slider_contrast = QSlider(Qt.Horizontal)
        self.slider_contrast.setRange(20, 300)
        self.slider_sharpness = QSlider(Qt.Horizontal)
        self.slider_sharpness.setRange(0, 300)

        sat, bri, con, sha = enhancement_to_slider_values(self._current_params)
        self.slider_saturation.setValue(sat)
        self.slider_brightness.setValue(bri)
        self.slider_contrast.setValue(con)
        self.slider_sharpness.setValue(sha)

        self.lbl_saturation = QLabel()
        self.lbl_brightness = QLabel()
        self.lbl_contrast = QLabel()
        self.lbl_sharpness = QLabel()
        self._update_slider_labels()

        sat_row = QHBoxLayout()
        sat_row.addWidget(self.slider_saturation, 1)
        sat_row.addWidget(self.lbl_saturation)
        sliders_layout.addRow("Цветность:", sat_row)

        bri_row = QHBoxLayout()
        bri_row.addWidget(self.slider_brightness, 1)
        bri_row.addWidget(self.lbl_brightness)
        sliders_layout.addRow("Яркость:", bri_row)

        con_row = QHBoxLayout()
        con_row.addWidget(self.slider_contrast, 1)
        con_row.addWidget(self.lbl_contrast)
        sliders_layout.addRow("Контрастность:", con_row)

        sha_row = QHBoxLayout()
        sha_row.addWidget(self.slider_sharpness, 1)
        sha_row.addWidget(self.lbl_sharpness)
        sliders_layout.addRow("Резкость:", sha_row)

        layout.addWidget(sliders_group)

        actions_row = QHBoxLayout()
        self.btn_reset = QPushButton("Сбросить")
        actions_row.addWidget(self.btn_reset)
        actions_row.addStretch(1)
        layout.addLayout(actions_row)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(self.button_box)

        self.slider_saturation.valueChanged.connect(self._on_sliders_changed)
        self.slider_brightness.valueChanged.connect(self._on_sliders_changed)
        self.slider_contrast.valueChanged.connect(self._on_sliders_changed)
        self.slider_sharpness.valueChanged.connect(self._on_sliders_changed)
        self.btn_reset.clicked.connect(self._reset_values)
        self.btn_zoom_in.clicked.connect(self.view_preview.zoom_in)
        self.btn_zoom_out.clicked.connect(self.view_preview.zoom_out)
        self.btn_zoom_fit.clicked.connect(self.view_preview.zoom_fit)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self._load_image()

    def _load_image(self) -> None:
        display_img, _ = backend.load_display_image(self.image_path, max_side=2200)
        self._base_display_image = display_img
        self.path_label.setText(f"Файл: {Path(self.image_path).name}")
        self._update_preview()

    def _current_from_sliders(self) -> dict:
        return slider_values_to_enhancement(
            int(self.slider_saturation.value()),
            int(self.slider_brightness.value()),
            int(self.slider_contrast.value()),
            int(self.slider_sharpness.value()),
        )

    def _update_slider_labels(self) -> None:
        params = self._current_from_sliders()
        self.lbl_saturation.setText(f"{params['saturation']:.2f}")
        self.lbl_brightness.setText(f"{params['brightness']:.0f}")
        self.lbl_contrast.setText(f"{params['contrast']:.2f}")
        self.lbl_sharpness.setText(f"{params['sharpness']:.2f}")

    def _on_sliders_changed(self) -> None:
        self._update_slider_labels()
        self._update_preview()

    def _update_preview(self) -> None:
        if self._base_display_image is None:
            return
        params = self._current_from_sliders()
        enhanced = backend.apply_image_enhancement(self._base_display_image, params)
        pixmap = self._bgr_to_pixmap(enhanced)
        self.scene_preview.clear()
        self.scene_preview.addPixmap(pixmap)
        self.scene_preview.setSceneRect(self.scene_preview.itemsBoundingRect())
        if not self._preview_initialized:
            self.view_preview.zoom_fit()
            self._preview_initialized = True

    def _reset_values(self) -> None:
        defaults = backend.get_default_enhancement_params()
        sat, bri, con, sha = enhancement_to_slider_values(defaults)
        self.slider_saturation.setValue(sat)
        self.slider_brightness.setValue(bri)
        self.slider_contrast.setValue(con)
        self.slider_sharpness.setValue(sha)

    def _bgr_to_pixmap(self, image_bgr) -> QPixmap:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    def accept(self) -> None:
        self.result_params = self._current_from_sliders()
        super().accept()


class RoiNameDialog(QDialog):
    def __init__(
        self,
        suggestions: list[str],
        default_name: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Название ROI")
        self.resize(420, 120)
        self.result_name: str | None = None

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.combo_name = QComboBox()
        self.combo_name.setEditable(True)
        seen: set[str] = set()
        for value in suggestions:
            text = str(value).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            self.combo_name.addItem(text)
        if default_name and self.combo_name.findText(default_name) < 0:
            self.combo_name.addItem(default_name)
        self.combo_name.setCurrentText(default_name)
        form.addRow("Название:", self.combo_name)
        layout.addLayout(form)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(self.button_box)

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

    def accept(self) -> None:
        text = self.combo_name.currentText().strip()
        if not text:
            QMessageBox.warning(self, "Пустое название", "Введите название ROI")
            return
        self.result_name = text
        super().accept()


class RoiAnnotationDialog(QDialog):
    def __init__(
        self,
        image_path: str,
        mode_key: str,
        enhancement_params: dict | None,
        name_suggestions: list[str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Разметка областей интереса")
        self.resize(1500, 900)

        self.image_path = image_path
        self.mode_key = mode_key
        self.enhancement_params = backend.normalize_enhancement_params(enhancement_params)
        self.name_suggestions = [str(x).strip() for x in name_suggestions if str(x).strip()]
        self.new_names: list[str] = []

        self.display_scale = 1.0
        self.rois: list[dict] = []
        self.next_roi_id = 1

        layout = QVBoxLayout(self)

        top_row = QHBoxLayout()
        self.path_label = QLabel(f"Файл: {Path(image_path).name}")
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

        center = QHBoxLayout()
        layout.addLayout(center, 1)

        controls = QVBoxLayout()
        self.btn_rect_mode = QPushButton("Прямоугольник")
        self.btn_poly_mode = QPushButton("Полигон")
        self.btn_delete_roi = QPushButton("Удалить выбранные")
        self.btn_clear_roi = QPushButton("Удалить все ROI")
        self.btn_rect_mode.setCheckable(True)
        self.btn_poly_mode.setCheckable(True)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.btn_rect_mode)
        self.mode_group.addButton(self.btn_poly_mode)
        controls.addWidget(self.btn_rect_mode)
        controls.addWidget(self.btn_poly_mode)
        controls.addWidget(self.btn_delete_roi)
        controls.addWidget(self.btn_clear_roi)
        controls.addStretch(1)
        controls_widget = QWidget(self)
        controls_widget.setLayout(controls)
        controls_widget.setFixedWidth(220)
        center.addWidget(controls_widget)

        self.scene = ImageScene(self)
        self.view = CalibrationView(self.scene, self)
        center.addWidget(self.view, 1)

        right = QVBoxLayout()
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["ID", "Название", "Тип"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        right.addWidget(self.table, 1)
        right_widget = QWidget(self)
        right_widget.setLayout(right)
        right_widget.setFixedWidth(320)
        center.addWidget(right_widget)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(self.button_box)

        self.btn_zoom_in.clicked.connect(self.view.zoom_in)
        self.btn_zoom_out.clicked.connect(self.view.zoom_out)
        self.btn_zoom_fit.clicked.connect(self.view.zoom_fit)
        self.btn_delete_roi.clicked.connect(self._delete_selected_rois)
        self.btn_clear_roi.clicked.connect(self._clear_rois)
        self.scene.roi_created.connect(self._on_roi_created)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.btn_rect_mode.clicked.connect(lambda: self._set_mode(ToolMode.RECTANGLE))
        self.btn_poly_mode.clicked.connect(lambda: self._set_mode(ToolMode.POLYGON))

        self._load_image()
        self._configure_mode()

    def _configure_mode(self) -> None:
        if self.mode_key == BATCH_MODE_RECT:
            self.btn_rect_mode.setChecked(True)
            self.btn_poly_mode.setEnabled(False)
            self._set_mode(ToolMode.RECTANGLE)
        else:
            self.btn_poly_mode.setChecked(True)
            self.btn_rect_mode.setEnabled(False)
            self._set_mode(ToolMode.POLYGON)

    def _set_mode(self, mode: ToolMode) -> None:
        self.scene.set_tool_mode(mode)

    def _load_image(self) -> None:
        display_img, self.display_scale = backend.load_display_image(self.image_path, max_side=2200)
        enhanced_display = backend.apply_image_enhancement(display_img, self.enhancement_params)
        pixmap = self._bgr_to_pixmap(enhanced_display)
        self.scene.set_image_pixmap(pixmap)
        self.view.zoom_fit()

    def _bgr_to_pixmap(self, image_bgr) -> QPixmap:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    def _on_roi_created(self, payload: dict) -> None:
        points_display = payload.get("points", [])
        roi_type = payload.get("type", "polygon")
        if len(points_display) < 3:
            return

        name_dialog = RoiNameDialog(
            suggestions=self.name_suggestions,
            default_name=f"Область {self.next_roi_id}",
            parent=self,
        )
        if name_dialog.exec_() != QDialog.Accepted or not name_dialog.result_name:
            return

        roi_name = name_dialog.result_name
        points_orig = [
            (float(x) * self.display_scale, float(y) * self.display_scale)
            for x, y in points_display
        ]
        roi = {
            "id": self.next_roi_id,
            "name": roi_name,
            "type": roi_type,
            "points": points_orig,
        }
        self.rois.append(roi)
        self.scene.add_roi_item(self.next_roi_id, roi_type, points_display)
        self.next_roi_id += 1

        if roi_name not in self.name_suggestions:
            self.name_suggestions.append(roi_name)
        if roi_name not in self.new_names:
            self.new_names.append(roi_name)

        self._refresh_table()

    def _refresh_table(self) -> None:
        self.table.setRowCount(len(self.rois))
        for row_idx, roi in enumerate(self.rois):
            self.table.setItem(row_idx, 0, QTableWidgetItem(str(roi["id"])))
            self.table.setItem(row_idx, 1, QTableWidgetItem(str(roi.get("name", ""))))
            self.table.setItem(row_idx, 2, QTableWidgetItem(str(roi.get("type", ""))))

    def _delete_selected_rois(self) -> None:
        ids_from_scene = set(self.scene.selected_roi_ids())
        ids_from_table: set[int] = set()
        for idx in self.table.selectionModel().selectedRows():
            roi_item = self.table.item(idx.row(), 0)
            if roi_item is None:
                continue
            try:
                ids_from_table.add(int(roi_item.text()))
            except Exception:
                continue
        selected_ids = ids_from_scene | ids_from_table
        if not selected_ids:
            return

        self.rois = [roi for roi in self.rois if int(roi["id"]) not in selected_ids]
        for roi_id in selected_ids:
            self.scene.remove_roi_item(roi_id)
        self._refresh_table()

    def _clear_rois(self) -> None:
        self.rois.clear()
        self.scene.clear_roi_items()
        self._refresh_table()

    def accept(self) -> None:
        if not self.rois:
            QMessageBox.warning(
                self,
                "Нет ROI",
                "Добавьте хотя бы одну область интереса на изображении",
            )
            return
        super().accept()


class BatchSetupDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Пакетная обработка")
        self.resize(620, 220)

        self.folder_path: str | None = None
        self.mode_key: str = BATCH_MODE_FULL

        layout = QVBoxLayout(self)
        form = QFormLayout()

        folder_row = QHBoxLayout()
        self.input_folder = QLineEdit()
        self.input_folder.setPlaceholderText("Выберите папку с изображениями")
        self.btn_browse = QPushButton("Обзор")
        folder_row.addWidget(self.input_folder, 1)
        folder_row.addWidget(self.btn_browse)
        form.addRow("Папка:", folder_row)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem(BATCH_MODE_LABELS_RU[BATCH_MODE_FULL], BATCH_MODE_FULL)
        self.mode_combo.addItem(BATCH_MODE_LABELS_RU[BATCH_MODE_RECT], BATCH_MODE_RECT)
        self.mode_combo.addItem(BATCH_MODE_LABELS_RU[BATCH_MODE_POLY], BATCH_MODE_POLY)
        form.addRow("Режим:", self.mode_combo)

        layout.addLayout(form)

        self.hint = QLabel(
            "Режим по всему кадру не требует ручной разметки ROI. "
            "В режимах по прямоугольникам и полигонам разметка будет выполнена для каждого фото."
        )
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(self.button_box)

        self.btn_browse.clicked.connect(self._browse_folder)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями")
        if folder:
            self.input_folder.setText(folder)

    def accept(self) -> None:
        folder = self.input_folder.text().strip()
        if not folder:
            QMessageBox.warning(self, "Нет папки", "Укажите папку с изображениями")
            return

        path = Path(folder)
        if not path.exists() or not path.is_dir():
            QMessageBox.warning(self, "Ошибка пути", "Указанная папка не существует")
            return

        images = [
            p for p in sorted(path.iterdir())
            if p.is_file() and p.suffix.lower() in backend.SUPPORTED_IMAGE_FORMATS
        ]
        if not images:
            QMessageBox.warning(
                self,
                "Нет изображений",
                "В папке не найдено поддерживаемых изображений (TIFF/PNG/JPEG)",
            )
            return

        self.folder_path = str(path)
        self.mode_key = str(self.mode_combo.currentData())
        super().accept()
