from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import backend
from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QBrush, QColor, QImage, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGraphicsItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QLabel,
    QMessageBox,
    QToolButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)

from ui.scenes import CalibrationView


class SquareRoiDialog(QDialog):
    def __init__(
        self,
        image_path: str,
        enhancement_params: dict | None = None,
        pixels_per_mm: float | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Размещение квадратной ROI")
        self.resize(1200, 900)
        self.setWindowState(self.windowState() | Qt.WindowMaximized)

        self.image_path = image_path
        self.enhancement_params = backend.normalize_enhancement_params(enhancement_params)
        self.pixels_per_mm = pixels_per_mm
        self._display_scale = 1.0
        self._square_points: list[tuple[float, float]] | None = None
        self._placed = False

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

        form = QFormLayout()
        self.side_mm_spin = QDoubleSpinBox()
        self.side_mm_spin.setDecimals(3)
        self.side_mm_spin.setRange(0.001, 1000.0)
        self.side_mm_spin.setValue(1.0)
        self.side_mm_spin.setSingleStep(0.1)
        self.side_px_label = QLabel("-")
        form.addRow("Сторона квадрата (мм):", self.side_mm_spin)
        form.addRow("Сторона (пикс):", self.side_px_label)
        layout.addLayout(form)

        hint = QLabel(
            "Кликните на изображении, чтобы разместить центр квадрата. "
            "Правая кнопка / Esc — отменить."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._scene = QGraphicsScene(self)
        self._view = CalibrationView(self._scene, self)
        self._view.setCursor(Qt.CrossCursor)
        self._view.viewport().setCursor(Qt.CrossCursor)
        layout.addWidget(self._view, 1)

        self._coord_overlay = QLabel("x —   y —", self._view.viewport())
        self._coord_overlay.setObjectName("coord_overlay")
        self._coord_overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._coord_overlay.adjustSize()
        self._coord_overlay.move(8, 8)
        self._coord_overlay.show()
        self._coord_overlay.raise_()

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(self.button_box)

        self._square_item: QGraphicsPolygonItem | None = None

        self.btn_zoom_in.clicked.connect(self._view.zoom_in)
        self.btn_zoom_out.clicked.connect(self._view.zoom_out)
        self.btn_zoom_fit.clicked.connect(self._view.zoom_fit)
        self.side_mm_spin.valueChanged.connect(self._update_side_px)
        self._scene.installEventFilter(self)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self._load_image()

    def _load_image(self) -> None:
        try:
            display_img, self._display_scale = backend.load_display_image(self.image_path, max_side=2200)
            enhanced = backend.apply_image_enhancement(display_img, self.enhancement_params)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))
            return

        pixmap = self._bgr_to_pixmap(enhanced)
        self._scene.clear()
        self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(self._scene.itemsBoundingRect())
        self._view.zoom_fit()
        self._update_side_px()

    def _bgr_to_pixmap(self, image_bgr) -> QPixmap:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    def _update_side_px(self) -> None:
        mm = self.side_mm_spin.value()
        if self.pixels_per_mm is not None and self.pixels_per_mm > 0:
            px = mm * self.pixels_per_mm
            self.side_px_label.setText(f"{px:.2f}")
        else:
            self.side_px_label.setText("— (нет калибровки)")

    def _draw_square_at(self, center_display: QPointF) -> None:
        if self._square_item is not None:
            self._scene.removeItem(self._square_item)
            self._square_item = None

        mm = self.side_mm_spin.value()
        if self.pixels_per_mm is not None and self.pixels_per_mm > 0:
            side_px = mm * self.pixels_per_mm
        else:
            side_px = 100.0

        side_display = side_px / self._display_scale
        half = side_display / 2.0

        cx, cy = center_display.x(), center_display.y()
        pts = [
            (cx - half, cy - half),
            (cx + half, cy - half),
            (cx + half, cy + half),
            (cx - half, cy + half),
        ]
        poly = QPolygonF([QPointF(x, y) for x, y in pts])
        self._square_item = QGraphicsPolygonItem(poly)
        self._square_item.setPen(QPen(QColor("#FF8C00"), 2))
        self._square_item.setBrush(QBrush(QColor(255, 140, 0, 40)))
        self._square_item.setZValue(50)
        self._scene.addItem(self._square_item)

        self._square_points = [
            (float(x) * self._display_scale, float(y) * self._display_scale)
            for x, y in pts
        ]
        self._placed = True

    def get_square_points(self) -> list[tuple[float, float]] | None:
        return self._square_points

    def eventFilter(self, obj, event) -> bool:
        if obj is self._scene:
            etype = event.type()
            if etype == event.GraphicsSceneMousePress:
                if event.button() == Qt.LeftButton:
                    pos = event.scenePos()
                    self._draw_square_at(pos)
                    event.accept()
                    return True
                if event.button() == Qt.RightButton:
                    if self._square_item is not None:
                        self._scene.removeItem(self._square_item)
                        self._square_item = None
                        self._square_points = None
                        self._placed = False
                    event.accept()
                    return True
            elif etype == event.GraphicsSceneMouseMove:
                pos = event.scenePos()
                ox = pos.x() * self._display_scale
                oy = pos.y() * self._display_scale
                self._coord_overlay.setText(f"x {ox:.0f}   y {oy:.0f}")
                self._coord_overlay.adjustSize()
                return False
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            if self._square_item is not None:
                self._scene.removeItem(self._square_item)
                self._square_item = None
                self._square_points = None
                self._placed = False
            event.accept()
            return
        super().keyPressEvent(event)

    def accept(self) -> None:
        if not self._placed or self._square_points is None:
            QMessageBox.warning(self, "Нет квадрата", "Кликните на изображении, чтобы разместить квадрат")
            return
        super().accept()
