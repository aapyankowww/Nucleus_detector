from __future__ import annotations

from enum import Enum

import cv2
import numpy as np
from PyQt5.QtCore import QLineF, QObject, QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)


class ToolMode(str, Enum):
    NONE = "none"
    RECTANGLE = "rectangle"
    POLYGON = "polygon"
    LINE = "line"


class ImageScene(QGraphicsScene):
    roi_created = pyqtSignal(dict)
    line_created = pyqtSignal(tuple)
    cursor_moved = pyqtSignal(float, float)
    point_picked = pyqtSignal(float, float)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.tool_mode: ToolMode = ToolMode.NONE
        self._picking_point: bool = False

        self._image_item = None

        self._start_point: QPointF | None = None
        self._temp_rect_item: QGraphicsRectItem | None = None
        self._temp_line_item: QGraphicsLineItem | None = None

        self._polygon_points: list[QPointF] = []
        self._polygon_item: QGraphicsPolygonItem | None = None
        self._poly_preview_line: QGraphicsLineItem | None = None

        self._roi_items: dict[int, QGraphicsPolygonItem] = {}
        self._item_to_roi: dict[QGraphicsItem, int] = {}
        self._nuclei_items: list[QGraphicsItem] = []

        self.setBackgroundBrush(QBrush(QColor("#151515")))

    def set_tool_mode(self, mode: ToolMode) -> None:
        self.tool_mode = mode
        if mode != ToolMode.POLYGON:
            self._clear_polygon_in_progress()

    def set_image_pixmap(self, pixmap: QPixmap | None) -> None:
        self.clear()
        self._reset_temp_state()
        self._roi_items.clear()
        self._item_to_roi.clear()
        self._nuclei_items.clear()

        if pixmap is None or pixmap.isNull():
            self._image_item = None
            self.setSceneRect(QRectF())
            return

        self._image_item = self.addPixmap(pixmap)
        self._image_item.setZValue(0)
        self.setSceneRect(self._image_item.boundingRect())

    def clear_roi_items(self) -> None:
        for item in list(self._roi_items.values()):
            self.removeItem(item)
        self._roi_items.clear()
        self._item_to_roi.clear()

    def clear_nuclei_items(self) -> None:
        for item in self._nuclei_items:
            self.removeItem(item)
        self._nuclei_items.clear()

    def add_roi_item(self, roi_id: int, roi_type: str, points: list[tuple[float, float]]) -> None:
        if len(points) < 3:
            return

        polygon = QPolygonF([QPointF(x, y) for x, y in points])
        item = QGraphicsPolygonItem(polygon)

        if roi_type == "rectangle":
            pen_color = QColor("#ffad33")
            fill_color = QColor(255, 173, 51, 30)
        else:
            pen_color = QColor("#5ec4ff")
            fill_color = QColor(94, 196, 255, 30)

        item.setPen(QPen(pen_color, 2))
        item.setBrush(QBrush(fill_color))
        item.setFlag(QGraphicsItem.ItemIsSelectable, True)
        item.setZValue(20)

        self.addItem(item)
        self._roi_items[roi_id] = item
        self._item_to_roi[item] = roi_id

    def remove_roi_item(self, roi_id: int) -> None:
        item = self._roi_items.pop(roi_id, None)
        if item is None:
            return
        self._item_to_roi.pop(item, None)
        self.removeItem(item)

    def selected_roi_ids(self) -> list[int]:
        result: list[int] = []
        for item in self.selectedItems():
            roi_id = self._item_to_roi.get(item)
            if roi_id is not None:
                result.append(roi_id)
        return result

    def set_nuclei_items(self, nuclei: list[dict]) -> None:
        self.clear_nuclei_items()

        contour_pen = QPen(QColor(50, 230, 120, 180), 1)
        center_pen = QPen(QColor(30, 255, 110, 230), 1)
        center_brush = QBrush(QColor(30, 255, 110, 150))

        for nuc in nuclei:
            contour = nuc.get("contour", [])
            if len(contour) >= 3:
                poly = QPolygonF([QPointF(float(x), float(y)) for x, y in contour])
                contour_item = QGraphicsPolygonItem(poly)
                contour_item.setPen(contour_pen)
                contour_item.setBrush(QBrush(Qt.NoBrush))
                contour_item.setZValue(30)
                self.addItem(contour_item)
                self._nuclei_items.append(contour_item)

            center = nuc.get("center")
            if center is not None:
                cx, cy = float(center[0]), float(center[1])
                radius = 2.0
                center_item = QGraphicsEllipseItem(cx - radius, cy - radius, radius * 2, radius * 2)
                center_item.setPen(center_pen)
                center_item.setBrush(center_brush)
                center_item.setZValue(31)
                self.addItem(center_item)
                self._nuclei_items.append(center_item)

    def start_pick_point(self) -> None:
        self._picking_point = True

    def cancel_pick_point(self) -> None:
        self._picking_point = False

    def cancel_current_drawing(self) -> None:
        self._reset_temp_state()

    def mousePressEvent(self, event) -> None:
        if self._image_item is None:
            super().mousePressEvent(event)
            return

        pos = self._clamp_to_image(event.scenePos())
        button = event.button()

        if self._picking_point and button == Qt.LeftButton:
            self._picking_point = False
            self.point_picked.emit(float(pos.x()), float(pos.y()))
            event.accept()
            return

        if self.tool_mode == ToolMode.POLYGON and button == Qt.RightButton and self._polygon_points:
            self._finalize_polygon()
            event.accept()
            return

        if button != Qt.LeftButton:
            super().mousePressEvent(event)
            return

        if self.tool_mode == ToolMode.RECTANGLE:
            self._start_point = pos
            self._temp_rect_item = QGraphicsRectItem(QRectF(pos, pos))
            self._temp_rect_item.setPen(QPen(QColor("#ffad33"), 2, Qt.DashLine))
            self._temp_rect_item.setZValue(50)
            self.addItem(self._temp_rect_item)
            event.accept()
            return

        if self.tool_mode == ToolMode.LINE:
            self._start_point = pos
            self._temp_line_item = QGraphicsLineItem(pos.x(), pos.y(), pos.x(), pos.y())
            self._temp_line_item.setPen(QPen(QColor("#ffd966"), 2))
            self._temp_line_item.setZValue(50)
            self.addItem(self._temp_line_item)
            event.accept()
            return

        if self.tool_mode == ToolMode.POLYGON:
            self._polygon_points.append(pos)
            self._update_polygon_preview(pos)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._image_item is None:
            super().mouseMoveEvent(event)
            return

        pos = self._clamp_to_image(event.scenePos())
        self.cursor_moved.emit(pos.x(), pos.y())

        if self.tool_mode == ToolMode.RECTANGLE and self._temp_rect_item and self._start_point:
            rect = QRectF(self._start_point, pos).normalized()
            self._temp_rect_item.setRect(rect)
            event.accept()
            return

        if self.tool_mode == ToolMode.LINE and self._temp_line_item and self._start_point:
            self._temp_line_item.setLine(self._start_point.x(), self._start_point.y(), pos.x(), pos.y())
            event.accept()
            return

        if self.tool_mode == ToolMode.POLYGON and self._polygon_points:
            self._update_polygon_preview(pos)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._image_item is None:
            super().mouseReleaseEvent(event)
            return

        pos = self._clamp_to_image(event.scenePos())

        if self.tool_mode == ToolMode.RECTANGLE and event.button() == Qt.LeftButton:
            if self._start_point is not None:
                rect = QRectF(self._start_point, pos).normalized()
                self._finalize_rectangle(rect)
                event.accept()
                return

        if self.tool_mode == ToolMode.LINE and event.button() == Qt.LeftButton:
            if self._start_point is not None:
                p1 = (self._start_point.x(), self._start_point.y())
                p2 = (pos.x(), pos.y())
                self._finalize_line(p1, p2)
                event.accept()
                return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if self.tool_mode == ToolMode.POLYGON and event.button() == Qt.LeftButton:
            self._finalize_polygon()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def _finalize_rectangle(self, rect: QRectF) -> None:
        if self._temp_rect_item is not None:
            self.removeItem(self._temp_rect_item)
            self._temp_rect_item = None

        self._start_point = None

        if rect.width() < 3 or rect.height() < 3:
            return

        points = [
            (rect.left(), rect.top()),
            (rect.right(), rect.top()),
            (rect.right(), rect.bottom()),
            (rect.left(), rect.bottom()),
        ]
        self.roi_created.emit({"type": "rectangle", "points": points})

    def _finalize_line(self, p1: tuple[float, float], p2: tuple[float, float]) -> None:
        if self._temp_line_item is not None:
            self.removeItem(self._temp_line_item)
            self._temp_line_item = None

        self._start_point = None
        if abs(p1[0] - p2[0]) < 1e-6 and abs(p1[1] - p2[1]) < 1e-6:
            return

        self.line_created.emit((p1, p2))

    def _update_polygon_preview(self, current_pos: QPointF | None = None) -> None:
        if self._polygon_item is not None:
            self.removeItem(self._polygon_item)
            self._polygon_item = None
        if self._poly_preview_line is not None:
            self.removeItem(self._poly_preview_line)
            self._poly_preview_line = None

        if not self._polygon_points:
            return

        poly = QPolygonF(self._polygon_points)
        self._polygon_item = QGraphicsPolygonItem(poly)
        self._polygon_item.setPen(QPen(QColor("#5ec4ff"), 2, Qt.DashLine))
        self._polygon_item.setBrush(QBrush(QColor(94, 196, 255, 18)))
        self._polygon_item.setZValue(50)
        self.addItem(self._polygon_item)

        if current_pos is not None:
            start = self._polygon_points[-1]
            self._poly_preview_line = QGraphicsLineItem(start.x(), start.y(), current_pos.x(), current_pos.y())
            self._poly_preview_line.setPen(QPen(QColor("#5ec4ff"), 1, Qt.DotLine))
            self._poly_preview_line.setZValue(51)
            self.addItem(self._poly_preview_line)

    def _finalize_polygon(self) -> None:
        points = [(p.x(), p.y()) for p in self._polygon_points]
        self._clear_polygon_in_progress()

        if len(points) < 3:
            return

        self.roi_created.emit({"type": "polygon", "points": points})

    def _clear_polygon_in_progress(self) -> None:
        self._polygon_points.clear()
        if self._polygon_item is not None:
            self.removeItem(self._polygon_item)
            self._polygon_item = None
        if self._poly_preview_line is not None:
            self.removeItem(self._poly_preview_line)
            self._poly_preview_line = None

    def _reset_temp_state(self) -> None:
        self._start_point = None
        if self._temp_rect_item is not None:
            self.removeItem(self._temp_rect_item)
            self._temp_rect_item = None
        if self._temp_line_item is not None:
            self.removeItem(self._temp_line_item)
            self._temp_line_item = None
        self._clear_polygon_in_progress()

    def _clamp_to_image(self, point: QPointF) -> QPointF:
        if self._image_item is None:
            return point
        rect = self._image_item.boundingRect()
        x = min(max(point.x(), rect.left()), rect.right())
        y = min(max(point.y(), rect.top()), rect.bottom())
        return QPointF(x, y)


class InvertedLineItem(QGraphicsLineItem):
    def paint(self, painter, option, widget=None) -> None:
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setCompositionMode(QPainter.CompositionMode_Difference)
        pen = QPen(self.pen())
        pen.setCosmetic(True)
        if pen.widthF() < 1.0:
            pen.setWidthF(1.0)
        painter.setPen(pen)
        painter.drawLine(self.line())
        painter.restore()


class CalibrationScene(QGraphicsScene):
    line_updated = pyqtSignal(float)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._image_item = None
        self._image_rect = QRectF()
        self._start_point: QPointF | None = None
        self._line_coords: tuple[tuple[float, float], tuple[float, float]] | None = None

        self._line_item = InvertedLineItem()
        self._line_item.setPen(QPen(QColor(255, 255, 255), 1))
        self._line_item.setZValue(50)
        self.addItem(self._line_item)

        self._cross_h = InvertedLineItem()
        self._cross_h.setPen(QPen(QColor(255, 255, 255), 1, Qt.DotLine))
        self._cross_h.setZValue(40)
        self.addItem(self._cross_h)

        self._cross_v = InvertedLineItem()
        self._cross_v.setPen(QPen(QColor(255, 255, 255), 1, Qt.DotLine))
        self._cross_v.setZValue(40)
        self.addItem(self._cross_v)

        self.setBackgroundBrush(QBrush(QColor("#141414")))

    def set_image_pixmap(self, pixmap: QPixmap | None) -> None:
        self.clear()
        self._line_coords = None
        self._start_point = None

        self._image_item = None
        self._image_rect = QRectF()
        if pixmap is None or pixmap.isNull():
            self.setSceneRect(QRectF())
            return

        self._image_item = self.addPixmap(pixmap)
        self._image_item.setZValue(0)
        self._image_rect = self._image_item.boundingRect()
        self.setSceneRect(self._image_rect)

        self._line_item = InvertedLineItem()
        self._line_item.setPen(QPen(QColor(255, 255, 255), 1))
        self._line_item.setZValue(50)
        self.addItem(self._line_item)

        self._cross_h = InvertedLineItem()
        self._cross_h.setPen(QPen(QColor(255, 255, 255), 1, Qt.DotLine))
        self._cross_h.setZValue(40)
        self.addItem(self._cross_h)

        self._cross_v = InvertedLineItem()
        self._cross_v.setPen(QPen(QColor(255, 255, 255), 1, Qt.DotLine))
        self._cross_v.setZValue(40)
        self.addItem(self._cross_v)

    def clear_line(self) -> None:
        self._line_coords = None
        self._start_point = None
        self._line_item.setLine(QLineF())
        self.line_updated.emit(0.0)

    def line_coords(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        return self._line_coords

    def mousePressEvent(self, event) -> None:
        if self._image_item is None or event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return

        pos = self._clamp_to_image(event.scenePos())
        self._start_point = pos
        self._line_coords = None
        self._line_item.setLine(QLineF(pos, pos))
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._image_item is None:
            super().mouseMoveEvent(event)
            return

        pos = self._clamp_to_image(event.scenePos())
        self._update_crosshair(pos)

        if self._start_point is not None:
            self._line_item.setLine(QLineF(self._start_point, pos))
            self.line_updated.emit(self._line_item.line().length())
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._image_item is None or event.button() != Qt.LeftButton:
            super().mouseReleaseEvent(event)
            return

        if self._start_point is None:
            super().mouseReleaseEvent(event)
            return

        end_pos = self._clamp_to_image(event.scenePos())
        line = QLineF(self._start_point, end_pos)
        self._line_item.setLine(line)
        self._start_point = None

        if line.length() <= 0.0:
            self._line_coords = None
            self.line_updated.emit(0.0)
        else:
            self._line_coords = (
                (line.x1(), line.y1()),
                (line.x2(), line.y2()),
            )
            self.line_updated.emit(line.length())
        event.accept()

    def _update_crosshair(self, pos: QPointF) -> None:
        if self._image_item is None:
            return
        rect = self._image_rect
        self._cross_h.setLine(QLineF(rect.left(), pos.y(), rect.right(), pos.y()))
        self._cross_v.setLine(QLineF(pos.x(), rect.top(), pos.x(), rect.bottom()))

    def _clamp_to_image(self, point: QPointF) -> QPointF:
        rect = self._image_rect
        x = min(max(point.x(), rect.left()), rect.right())
        y = min(max(point.y(), rect.top()), rect.bottom())
        return QPointF(x, y)


class CalibrationView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene, parent: QWidget | None = None) -> None:
        super().__init__(scene, parent)
        self.setRenderHints(self.renderHints() | self.renderHints())
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self.viewport().setCursor(Qt.CrossCursor)
        self._zoom = 1.0

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        self._zoom *= factor
        self.scale(factor, factor)
        event.accept()

    def zoom_in(self) -> None:
        self._zoom *= 1.2
        self.scale(1.2, 1.2)

    def zoom_out(self) -> None:
        self._zoom /= 1.2
        self.scale(1.0 / 1.2, 1.0 / 1.2)

    def zoom_fit(self) -> None:
        rect = self.scene().sceneRect()
        if rect.isNull():
            return
        self.resetTransform()
        self._zoom = 1.0
        self.fitInView(rect, Qt.KeepAspectRatio)

    def zoom_100(self) -> None:
        self.resetTransform()
        self._zoom = 1.0


class InvertedEllipseItem(QGraphicsEllipseItem):
    def paint(self, painter, option, widget=None) -> None:
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setCompositionMode(QPainter.CompositionMode_Difference)
        pen = QPen(self.pen())
        pen.setCosmetic(True)
        if pen.widthF() < 1.0:
            pen.setWidthF(1.0)
        painter.setPen(pen)
        painter.setBrush(self.brush())
        painter.drawEllipse(self.rect())
        painter.restore()


class CellSelectionScene(QGraphicsScene):
    circle_updated = pyqtSignal(float, float)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._image_item = None
        self._image_rect = QRectF()
        self._drag_start: QPointF | None = None
        self._selected_center: QPointF | None = None
        self._radius_px: float | None = None

        self._circle_item = InvertedEllipseItem()
        self._circle_item.setPen(QPen(QColor(255, 255, 255), 1))
        self._circle_item.setBrush(QBrush(Qt.NoBrush))
        self._circle_item.setZValue(40)
        self.addItem(self._circle_item)

        self.setBackgroundBrush(QBrush(QColor("#141414")))

    def set_image_pixmap(self, pixmap: QPixmap | None) -> None:
        self.clear()
        self._drag_start = None
        self._selected_center = None
        self._radius_px = None
        self._image_item = None
        self._image_rect = QRectF()

        if pixmap is None or pixmap.isNull():
            self.setSceneRect(QRectF())
            self.circle_updated.emit(0.0, 0.0)
            return

        self._image_item = self.addPixmap(pixmap)
        self._image_item.setZValue(0)
        self._image_rect = self._image_item.boundingRect()
        self.setSceneRect(self._image_rect)

        self._circle_item = InvertedEllipseItem()
        self._circle_item.setPen(QPen(QColor(255, 255, 255), 1))
        self._circle_item.setBrush(QBrush(Qt.NoBrush))
        self._circle_item.setZValue(40)
        self.addItem(self._circle_item)
        self.circle_updated.emit(0.0, 0.0)

    def clear_circle(self) -> None:
        self._drag_start = None
        self._selected_center = None
        self._radius_px = None
        self._circle_item.setRect(QRectF())
        self.circle_updated.emit(0.0, 0.0)

    def selected_radius_px(self) -> float | None:
        return self._radius_px

    def selected_center_xy(self) -> tuple[float, float] | None:
        if self._selected_center is None:
            return None
        return (float(self._selected_center.x()), float(self._selected_center.y()))

    def mousePressEvent(self, event) -> None:
        if self._image_item is None or event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return

        self._drag_start = self._clamp_to_image(event.scenePos())
        self._selected_center = None
        self._radius_px = None
        self._set_circle_rect(self._drag_start, 0.0)
        self.circle_updated.emit(0.0, 0.0)
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._image_item is None or self._drag_start is None:
            super().mouseMoveEvent(event)
            return
        pos = self._clamp_to_image(event.scenePos())
        center, radius = self._circle_from_diameter_points(self._drag_start, pos)
        self._set_circle_rect(center, radius)
        area = float(np.pi * radius * radius)
        self.circle_updated.emit(radius * 2.0, area)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self._image_item is None or event.button() != Qt.LeftButton:
            super().mouseReleaseEvent(event)
            return
        if self._drag_start is None:
            super().mouseReleaseEvent(event)
            return

        pos = self._clamp_to_image(event.scenePos())
        center, radius = self._circle_from_diameter_points(self._drag_start, pos)
        if radius < 1.0:
            self.clear_circle()
            event.accept()
            return

        self._drag_start = None
        self._selected_center = center
        self._radius_px = radius
        self._set_circle_rect(center, radius)
        area = float(np.pi * radius * radius)
        self.circle_updated.emit(radius * 2.0, area)
        event.accept()

    def _circle_from_diameter_points(self, p1: QPointF, p2: QPointF) -> tuple[QPointF, float]:
        center = QPointF((p1.x() + p2.x()) * 0.5, (p1.y() + p2.y()) * 0.5)
        radius = float(QLineF(p1, p2).length() * 0.5)
        return center, radius

    def _set_circle_rect(self, center: QPointF, radius: float) -> None:
        rect = QRectF(
            center.x() - radius,
            center.y() - radius,
            radius * 2.0,
            radius * 2.0,
        )
        self._circle_item.setRect(rect)

    def _clamp_to_image(self, point: QPointF) -> QPointF:
        rect = self._image_rect
        x = min(max(point.x(), rect.left()), rect.right())
        y = min(max(point.y(), rect.top()), rect.bottom())
        return QPointF(x, y)
