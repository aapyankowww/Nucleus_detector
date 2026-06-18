from __future__ import annotations

import copy
import os
from pathlib import Path

import cv2
import numpy as np
import backend
from PyQt5.QtCore import QRectF, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QKeySequence, QPixmap, QBrush
from PyQt5.QtWidgets import QAction
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QActionGroup,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGraphicsScene,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QInputDialog,
)

from ui.theme import apply_light_theme
from ui.scenes import ImageScene, CalibrationView, ToolMode
from ui.workers import (
    DetectionWorker,
    BatchDetectionWorker,
    enhancement_to_slider_values,
    slider_values_to_enhancement,
)
from ui.dialogs import (
    ScaleCalibrationDialog,
    DetectionParamsDialog,
    ColorTuningDialog,
    CellSelectionDialog,
    BatchSetupDialog,
    RoiAnnotationDialog,
    preprocess_mode_to_russian,
    BATCH_MODE_FULL,
    BATCH_MODE_RECT,
    BATCH_MODE_POLY,
    BATCH_MODE_LABELS_RU,
)
from ui.dialogs_square import SquareRoiDialog
from ui.dialogs_layers import LayerSetupDialog


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Анализ ядер и ROI")
        self.resize(1600, 900)

        self.image_path: str | None = None
        self.display_scale: float = 1.0
        self.pixels_per_mm: float | None = None

        self.rois: list[dict] = []
        self.nuclei: list[dict] = []
        self.next_roi_id = 1
        self.metrics_rows: list[dict] = []
        self.enhancement_params = backend.get_default_enhancement_params()
        self.roi_name_history: list[str] = []
        self._initial_scale_completed = False

        self.undo_stack: list[dict] = []
        self.max_undo = 100
        self._restoring_state = False

        self._det_thread: QThread | None = None
        self._det_worker: DetectionWorker | None = None
        self._batch_thread: QThread | None = None
        self._batch_worker: BatchDetectionWorker | None = None
        self._batch_progress: QProgressDialog | None = None
        self._pending_batch_folder_name: str = ""
        self._pending_batch_total_files: int = 0
        self._batch_folder_path: str | None = None
        self._white_ref_bgr: list[float] | None = None
        self._picking_for_wb: bool = False
        self._picking_for_hes: bool = False

        self._scene: ImageScene | None = None
        self._view: CalibrationView | None = None

        self._current_tab = 0

        self._build_ui()
        self._connect_signals()
        apply_light_theme(self)
        self._update_detector_status()
        self._update_scale_status()

    def _build_ui(self) -> None:
        self._create_menu_bar()
        self._create_mode_toolbar()

        central = QWidget(self)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        main_layout.addLayout(body, 1)

        left_scroll = QScrollArea()
        left_scroll.setObjectName("sidebar_scroll")
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFixedWidth(340)

        self.left_stack = QStackedWidget()
        self.left_stack.setObjectName("sidebar_content")

        self._build_tab1_left()
        self._build_tab2_left()
        self._build_tab3_left()
        left_scroll.setWidget(self.left_stack)

        body.addWidget(left_scroll)

        center_w = QWidget()
        center_layout = QVBoxLayout(center_w)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        canvas_toolbar = QWidget()
        canvas_toolbar.setObjectName("canvas_toolbar")
        canvas_toolbar.setFixedHeight(38)
        ct_layout = QHBoxLayout(canvas_toolbar)
        ct_layout.setContentsMargins(8, 4, 8, 4)
        ct_layout.setSpacing(4)

        lbl_roi_t = QLabel("ROI:")
        lbl_roi_t.setObjectName("canvas_label")
        ct_layout.addWidget(lbl_roi_t)

        self.btn_rect = QPushButton("Прямоугольник")
        self.btn_rect.setCheckable(True)
        self.btn_rect.setFixedHeight(28)
        self.btn_poly = QPushButton("Полигон")
        self.btn_poly.setCheckable(True)
        self.btn_poly.setFixedHeight(28)
        self.btn_line = QPushButton("Линия")
        self.btn_line.setCheckable(True)
        self.btn_line.setFixedHeight(28)
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        self.tool_group.addButton(self.btn_rect)
        self.tool_group.addButton(self.btn_poly)
        self.tool_group.addButton(self.btn_line)
        ct_layout.addWidget(self.btn_rect)
        ct_layout.addWidget(self.btn_poly)
        ct_layout.addWidget(self.btn_line)

        vsep1 = QFrame()
        vsep1.setFrameShape(QFrame.VLine)
        vsep1.setObjectName("vsep")
        ct_layout.addWidget(vsep1)

        self.btn_zoom_in_main = QPushButton("+")
        self.btn_zoom_in_main.setFixedSize(28, 28)
        self.btn_zoom_out_main = QPushButton("\u2212")
        self.btn_zoom_out_main.setFixedSize(28, 28)
        self.btn_zoom_100_main = QPushButton("1:1")
        self.btn_zoom_100_main.setFixedHeight(28)
        self.btn_zoom_fit_main = QPushButton("По размеру")
        self.btn_zoom_fit_main.setFixedHeight(28)
        ct_layout.addWidget(self.btn_zoom_in_main)
        ct_layout.addWidget(self.btn_zoom_out_main)
        ct_layout.addWidget(self.btn_zoom_100_main)
        ct_layout.addWidget(self.btn_zoom_fit_main)

        self.proc_label = QLabel("Готово к работе")
        self.proc_label.setObjectName("status_text")
        ct_layout.addStretch(1)
        ct_layout.addWidget(self.proc_label)

        center_layout.addWidget(canvas_toolbar)

        self._scene = ImageScene(self)
        self._view = CalibrationView(self._scene, self)
        center_layout.addWidget(self._view, 1)

        body.addWidget(center_w, 1)

        right = QWidget()
        right.setObjectName("right_panel")
        right.setFixedWidth(380)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(6)

        lbl_count_hdr = QLabel("НАЙДЕНО ЯДЕР")
        lbl_count_hdr.setObjectName("right_header")
        self.lbl_nuclei_count_big = QLabel("\u2014")
        self.lbl_nuclei_count_big.setObjectName("nuclei_count_big")
        right_layout.addWidget(lbl_count_hdr)
        right_layout.addWidget(self.lbl_nuclei_count_big)

        hsep1 = QFrame()
        hsep1.setFrameShape(QFrame.HLine)
        hsep1.setObjectName("hsep")
        right_layout.addWidget(hsep1)

        lbl_roi_hdr = QLabel("ОБЛАСТИ ИНТЕРЕСА")
        lbl_roi_hdr.setObjectName("right_header")
        right_layout.addWidget(lbl_roi_hdr)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["ROI ID", "Тип", "Площадь (мм\u00b2)", "Ядра", "Плотность"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        right_layout.addWidget(self.table, 1)

        self.btn_export = QPushButton("Экспортировать результаты")
        self.btn_export.setMinimumHeight(42)
        self.btn_export.setObjectName("btn_primary")
        right_layout.addWidget(self.btn_export)

        body.addWidget(right)
        self.setCentralWidget(central)

        self.coord_overlay = QLabel("x \u2014   y \u2014", self._view.viewport())
        self.coord_overlay.setObjectName("coord_overlay")
        self.coord_overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.coord_overlay.adjustSize()
        self.coord_overlay.move(8, 8)
        self.coord_overlay.show()
        self.coord_overlay.raise_()

        self.coord_label = QLabel("x \u2014   y \u2014")
        self.statusBar().addPermanentWidget(self.coord_label)

        self.shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.shortcut_delete = QShortcut(QKeySequence("Delete"), self)
        self.shortcut_escape = QShortcut(QKeySequence("Esc"), self)

        self.detector_label = QLabel("")
        self.statusBar().addPermanentWidget(self.detector_label)

    def _create_menu_bar(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("Файл")
        self.act_open = QAction("Открыть изображение...", self)
        self.act_open.setShortcut("Ctrl+O")
        file_menu.addAction(self.act_open)
        self.act_calibrate = QAction("Калибровать масштаб...", self)
        file_menu.addAction(self.act_calibrate)
        file_menu.addSeparator()
        self.act_export = QAction("Экспортировать результаты...", self)
        self.act_export.setShortcut("Ctrl+E")
        file_menu.addAction(self.act_export)
        file_menu.addSeparator()
        self.act_exit = QAction("Выход", self)
        self.act_exit.setShortcut("Ctrl+Q")
        file_menu.addAction(self.act_exit)

        mode_menu = menubar.addMenu("Режим")
        self.mode_group = QActionGroup(self)
        self.mode_group.setExclusive(True)
        self.act_mode1 = QAction("Одно изображение", self, checkable=True, checked=True)
        self.act_mode1.setShortcut("Ctrl+1")
        self.act_mode2 = QAction("Слои ткани", self, checkable=True)
        self.act_mode2.setShortcut("Ctrl+2")
        self.act_mode3 = QAction("Пакет по папке", self, checkable=True)
        self.act_mode3.setShortcut("Ctrl+3")
        self.mode_group.addAction(self.act_mode1)
        self.mode_group.addAction(self.act_mode2)
        self.mode_group.addAction(self.act_mode3)
        mode_menu.addAction(self.act_mode1)
        mode_menu.addAction(self.act_mode2)
        mode_menu.addAction(self.act_mode3)

        view_menu = menubar.addMenu("Вид")
        self.act_zoom_in = QAction("Приблизить", self)
        self.act_zoom_in.setShortcut("Ctrl++")
        view_menu.addAction(self.act_zoom_in)
        self.act_zoom_out = QAction("Отдалить", self)
        self.act_zoom_out.setShortcut("Ctrl+-")
        view_menu.addAction(self.act_zoom_out)
        self.act_zoom_100 = QAction("100%", self)
        self.act_zoom_100.setShortcut("Ctrl+0")
        view_menu.addAction(self.act_zoom_100)
        self.act_zoom_fit = QAction("По размеру", self)
        view_menu.addAction(self.act_zoom_fit)

        help_menu = menubar.addMenu("Справка")
        self.act_about = QAction("О программе", self)
        help_menu.addAction(self.act_about)

    def _create_mode_toolbar(self) -> None:
        toolbar = QToolBar("Режимы")
        toolbar.setObjectName("mode_toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QRectF(0, 0, 0, 0).size().toSize())
        toolbar.addAction(self.act_mode1)
        toolbar.addAction(self.act_mode2)
        toolbar.addAction(self.act_mode3)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

    def _add_card(self, layout: QVBoxLayout, title: str) -> QVBoxLayout:
        lbl = QLabel(title)
        lbl.setObjectName("group_title")
        layout.addWidget(lbl)
        body = QWidget()
        body.setObjectName("group_body")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(8, 8, 8, 8)
        body_layout.setSpacing(6)
        layout.addWidget(body)
        return body_layout

    def _build_tab1_left(self) -> None:
        tab1 = QWidget()
        tab1.setObjectName("sidebar")
        layout = QVBoxLayout(tab1)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(0)

        c1 = self._add_card(layout, "Масштаб")
        self.btn_calibrate = QPushButton("Калибровать масштаб")
        self.btn_calibrate.setObjectName("btn_primary")
        self.scale_label = QLabel("Масштаб не задан")
        self.scale_label.setObjectName("status_text")
        c1.addWidget(self.btn_calibrate)
        c1.addWidget(self.scale_label)

        c2 = self._add_card(layout, "Изображение")
        self.btn_open = QPushButton("Открыть изображение")
        self.btn_open.setObjectName("btn_primary")
        c2.addWidget(self.btn_open)

        c3 = self._add_card(layout, "Область интереса")
        self.roi_mode_combo = QComboBox()
        self.roi_mode_combo.addItem("Весь кадр", "full")
        self.roi_mode_combo.addItem("Квадрат N×N мм", "square")
        self.roi_mode_combo.addItem("Свободная ROI", "free")
        c3.addWidget(QLabel("Режим ROI:"))
        c3.addWidget(self.roi_mode_combo)

        self.square_size_group = QWidget()
        sq_layout = QHBoxLayout(self.square_size_group)
        sq_layout.setContentsMargins(0, 0, 0, 0)
        sq_layout.addWidget(QLabel("Сторона (мм):"))
        self.square_size_spin = QDoubleSpinBox()
        self.square_size_spin.setDecimals(3)
        self.square_size_spin.setRange(0.001, 1000.0)
        self.square_size_spin.setValue(1.0)
        sq_layout.addWidget(self.square_size_spin)
        self.btn_place_square = QPushButton("Разместить")
        self.btn_place_square.setObjectName("btn_secondary")
        sq_layout.addWidget(self.btn_place_square)
        self.square_size_group.setVisible(False)
        c3.addWidget(self.square_size_group)

        self.btn_clear_rois = QPushButton("Удалить все ROI")
        self.btn_clear_rois.setObjectName("btn_danger")
        c3.addWidget(self.btn_clear_rois)

        c4 = self._add_card(layout, "Цветокоррекция")
        self.enhance_collapse_btn = QPushButton("▾ Настройки цветокоррекции")
        self.enhance_collapse_btn.setObjectName("btn_secondary")
        self.enhance_collapse_btn.setStyleSheet("text-align: left; font-weight: normal;")
        c4.addWidget(self.enhance_collapse_btn)

        self.btn_pick_white = QPushButton("Выбрать белую точку")
        self.btn_pick_white.setObjectName("btn_secondary")
        c4.addWidget(self.btn_pick_white)

        wb_row = QHBoxLayout()
        wb_row.setContentsMargins(0, 0, 0, 0)
        wb_row.addWidget(QLabel("Сила ББ:"))
        self.slider_wb = QSlider(Qt.Horizontal)
        self.slider_wb.setRange(0, 100)
        self.slider_wb.setValue(50)
        wb_row.addWidget(self.slider_wb, 1)
        self.lbl_wb = QLabel("0.50")
        wb_row.addWidget(self.lbl_wb)
        c4.addLayout(wb_row)

        self.slider_saturation = QSlider(Qt.Horizontal)
        self.slider_saturation.setRange(0, 300)
        self.slider_brightness = QSlider(Qt.Horizontal)
        self.slider_brightness.setRange(-100, 100)
        self.slider_contrast = QSlider(Qt.Horizontal)
        self.slider_contrast.setRange(20, 300)
        self.slider_sharpness = QSlider(Qt.Horizontal)
        self.slider_sharpness.setRange(0, 300)

        sat, bri, con, sha, wb_enabled, wb_val = enhancement_to_slider_values(self.enhancement_params)
        self.slider_saturation.setValue(sat)
        self.slider_brightness.setValue(bri)
        self.slider_contrast.setValue(con)
        self.slider_sharpness.setValue(sha)
        self.slider_wb.setValue(wb_val)

        self.lbl_saturation = QLabel()
        self.lbl_brightness = QLabel()
        self.lbl_contrast = QLabel()
        self.lbl_sharpness = QLabel()
        self._update_enhancement_labels()

        enh_form = QFormLayout()
        enh_form.setSpacing(4)
        enh_form.setContentsMargins(0, 0, 0, 0)
        for slider, lbl, name in [
            (self.slider_saturation, self.lbl_saturation, "Насыщенность:"),
            (self.slider_brightness, self.lbl_brightness, "Яркость:"),
            (self.slider_contrast, self.lbl_contrast, "Контраст:"),
            (self.slider_sharpness, self.lbl_sharpness, "Резкость:"),
        ]:
            row_w = QHBoxLayout()
            row_w.addWidget(slider, 1)
            row_w.addWidget(lbl)
            enh_form.addRow(name, row_w)
        c4.addLayout(enh_form)
        self.btn_enhance_reset = QPushButton("Сбросить всё")
        self.btn_enhance_reset.setObjectName("btn_secondary")
        c4.addWidget(self.btn_enhance_reset)
        enh_form.parentWidget().setVisible(True)
        self._enhance_visible = True

        self.lbl_hes_status = QLabel("")
        self.lbl_hes_status.setObjectName("status_text")
        self.lbl_hes_status.setWordWrap(True)
        c4.addWidget(self.lbl_hes_status)

        self.btn_pick_nucleus = QPushButton("Выбрать ядро для H\u0026E")
        self.btn_pick_nucleus.setObjectName("btn_secondary")
        c4.addWidget(self.btn_pick_nucleus)
        self.btn_hes_reset = QPushButton("Сбросить H\u0026E")
        self.btn_hes_reset.setObjectName("btn_danger")
        c4.addWidget(self.btn_hes_reset)

        c5 = self._add_card(layout, "Детекция")
        self.detection_preset_combo = QComboBox()
        self.detection_preset_combo.addItem("Стандартный")
        self.detection_preset_combo.addItem("Чувствительный")
        self.detection_preset_combo.addItem("Точный")
        c5.addWidget(QLabel("Пресет:"))
        c5.addWidget(self.detection_preset_combo)

        self.btn_detect = QPushButton("Запустить детекцию")
        self.btn_detect.setMinimumHeight(44)
        self.btn_detect.setObjectName("btn_primary")
        c5.addWidget(self.btn_detect)

        self.btn_detection_params = QPushButton("Параметры нейросети...")
        self.btn_detection_params.setObjectName("btn_secondary")
        c5.addWidget(self.btn_detection_params)

        self.lbl_nuclei_count = QLabel("")
        self.lbl_nuclei_count.setObjectName("status_text")
        c5.addWidget(self.lbl_nuclei_count)

        self.btn_batch = QPushButton("Пакетная обработка")
        self.btn_batch.setObjectName("btn_secondary")
        layout.addSpacing(4)
        layout.addWidget(self.btn_batch)
        layout.addStretch(1)

        self.btn_load_model = QPushButton()
        self.btn_reset_model = QPushButton()

        self.left_stack.addWidget(tab1)

    def _build_tab2_left(self) -> None:
        tab2 = QWidget()
        tab2.setObjectName("sidebar")
        layout = QVBoxLayout(tab2)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(0)

        c1 = self._add_card(layout, "Масштаб")
        self.btn_calibrate_t2 = QPushButton("Калибровать масштаб")
        self.btn_calibrate_t2.setObjectName("btn_primary")
        self.scale_label_t2 = QLabel("Масштаб не задан")
        self.scale_label_t2.setObjectName("status_text")
        c1.addWidget(self.btn_calibrate_t2)
        c1.addWidget(self.scale_label_t2)

        c2 = self._add_card(layout, "Изображение")
        self.btn_open_t2 = QPushButton("Открыть изображение")
        self.btn_open_t2.setObjectName("btn_primary")
        c2.addWidget(self.btn_open_t2)

        c3 = self._add_card(layout, "Слои ткани")
        self.btn_layer_setup = QPushButton("Настройка слоёв")
        self.btn_layer_setup.setObjectName("btn_primary")
        c3.addWidget(self.btn_layer_setup)
        self.lbl_layer_info = QLabel("Слои не настроены")
        self.lbl_layer_info.setObjectName("status_text")
        self.lbl_layer_info.setWordWrap(True)
        c3.addWidget(self.lbl_layer_info)

        c4 = self._add_card(layout, "Детекция")
        self.btn_detect_t2 = QPushButton("Запустить детекцию + слои")
        self.btn_detect_t2.setMinimumHeight(44)
        self.btn_detect_t2.setObjectName("btn_primary")
        c4.addWidget(self.btn_detect_t2)

        layout.addStretch(1)
        self.left_stack.addWidget(tab2)

    def _build_tab3_left(self) -> None:
        tab3 = QWidget()
        tab3.setObjectName("sidebar")
        layout = QVBoxLayout(tab3)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(0)

        c1 = self._add_card(layout, "Масштаб")
        self.btn_calibrate_t3 = QPushButton("Калибровать масштаб")
        self.btn_calibrate_t3.setObjectName("btn_primary")
        self.scale_label_t3 = QLabel("Масштаб не задан")
        self.scale_label_t3.setObjectName("status_text")
        c1.addWidget(self.btn_calibrate_t3)
        c1.addWidget(self.scale_label_t3)

        c2 = self._add_card(layout, "Папка")
        self.btn_select_folder = QPushButton("Выбрать папку с изображениями")
        self.btn_select_folder.setObjectName("btn_primary")
        self.lbl_folder = QLabel("Папка не выбрана")
        self.lbl_folder.setObjectName("status_text")
        c2.addWidget(self.btn_select_folder)
        c2.addWidget(self.lbl_folder)

        c3 = self._add_card(layout, "Режим обработки")
        self.batch_mode_combo = QComboBox()
        self.batch_mode_combo.addItem(BATCH_MODE_LABELS_RU[BATCH_MODE_FULL], BATCH_MODE_FULL)
        self.batch_mode_combo.addItem(BATCH_MODE_LABELS_RU[BATCH_MODE_RECT], BATCH_MODE_RECT)
        self.batch_mode_combo.addItem(BATCH_MODE_LABELS_RU[BATCH_MODE_POLY], BATCH_MODE_POLY)
        c3.addWidget(QLabel("Режим:"))
        c3.addWidget(self.batch_mode_combo)

        self.batch_square_group = QWidget()
        bs_layout = QHBoxLayout(self.batch_square_group)
        bs_layout.setContentsMargins(0, 0, 0, 0)
        bs_layout.addWidget(QLabel("Сторона (мм):"))
        self.batch_square_spin = QDoubleSpinBox()
        self.batch_square_spin.setDecimals(3)
        self.batch_square_spin.setRange(0.001, 1000.0)
        self.batch_square_spin.setValue(1.0)
        bs_layout.addWidget(self.batch_square_spin)
        self.batch_square_group.setVisible(False)
        c3.addWidget(self.batch_square_group)

        c4 = self._add_card(layout, "Запуск")
        self.btn_batch_run = QPushButton("Запустить пакетную обработку")
        self.btn_batch_run.setMinimumHeight(44)
        self.btn_batch_run.setObjectName("btn_primary")
        c4.addWidget(self.btn_batch_run)

        layout.addStretch(1)
        self.left_stack.addWidget(tab3)

    def _connect_signals(self) -> None:
        self.mode_group.triggered.connect(self._on_mode_action)
        self.act_open.triggered.connect(self.open_image)
        self.act_calibrate.triggered.connect(self.activate_calibration)
        self.act_export.triggered.connect(self.export_results)
        self.act_exit.triggered.connect(self.close)
        self.act_zoom_in.triggered.connect(self._view.zoom_in)
        self.act_zoom_out.triggered.connect(self._view.zoom_out)
        self.act_zoom_100.triggered.connect(self._view.zoom_100)
        self.act_zoom_fit.triggered.connect(self._view.zoom_fit)
        self.act_about.triggered.connect(self._show_about)

        self.btn_calibrate.clicked.connect(self.activate_calibration)
        self.btn_calibrate_t2.clicked.connect(self.activate_calibration)
        self.btn_calibrate_t3.clicked.connect(self.activate_calibration)

        self.btn_open.clicked.connect(self.open_image)
        self.btn_open_t2.clicked.connect(self.open_image)

        self.btn_layer_setup.clicked.connect(self._open_layer_setup)
        self.btn_detect_t2.clicked.connect(self._run_layer_detection)

        self.btn_select_folder.clicked.connect(self._select_batch_folder)
        self.btn_batch_run.clicked.connect(self.run_batch_processing)

        self.btn_detect.clicked.connect(self.detect_nuclei)
        self.btn_detection_params.clicked.connect(self.open_detection_params)
        self.btn_batch.clicked.connect(self.run_batch_processing)
        self.btn_export.clicked.connect(self.export_results)

        self.roi_mode_combo.currentIndexChanged.connect(self._on_roi_mode_changed)
        self.btn_place_square.clicked.connect(self._open_square_roi_dialog)
        self.btn_clear_rois.clicked.connect(self._clear_rois)

        self.btn_rect.clicked.connect(lambda: self.set_tool(ToolMode.RECTANGLE))
        self.btn_poly.clicked.connect(lambda: self.set_tool(ToolMode.POLYGON))
        self.btn_line.clicked.connect(lambda: self.set_tool(ToolMode.LINE))
        self.btn_zoom_in_main.clicked.connect(self._view.zoom_in)
        self.btn_zoom_out_main.clicked.connect(self._view.zoom_out)
        self.btn_zoom_100_main.clicked.connect(self._view.zoom_100)
        self.btn_zoom_fit_main.clicked.connect(self._view.zoom_fit)

        self._scene.roi_created.connect(self._on_roi_created)
        self._scene.line_created.connect(self._on_line_created)
        self._scene.cursor_moved.connect(self._on_cursor_moved)

        self.shortcut_undo.activated.connect(self.undo)
        self.shortcut_delete.activated.connect(self.delete_selected_rois)
        self.shortcut_escape.activated.connect(self.cancel_drawing)

        self.slider_saturation.valueChanged.connect(self._on_enhancement_changed)
        self.slider_brightness.valueChanged.connect(self._on_enhancement_changed)
        self.slider_contrast.valueChanged.connect(self._on_enhancement_changed)
        self.slider_sharpness.valueChanged.connect(self._on_enhancement_changed)
        self.btn_enhance_reset.clicked.connect(self.reset_enhancement_settings)

        self.btn_enhance_reset.clicked.connect(self.reset_enhancement_settings)

        self.enhance_collapse_btn.clicked.connect(self._toggle_enhance_collapse)
        self.slider_wb.valueChanged.connect(self._on_enhancement_changed)
        self.btn_pick_white.clicked.connect(self._start_pick_white)
        self.btn_pick_nucleus.clicked.connect(self._start_pick_nucleus)
        self.btn_hes_reset.clicked.connect(self._reset_hes_reference)

        self._scene.point_picked.connect(self._on_point_picked)

    def _on_mode_action(self, action: QAction) -> None:
        index = self.mode_group.actions().index(action)
        self._current_tab = index
        self.left_stack.setCurrentIndex(index)

        is_mode1 = index == 0
        is_mode2 = index == 1

        self.btn_rect.setVisible(is_mode1)
        self.btn_poly.setVisible(is_mode1)
        self.btn_line.setVisible(is_mode1)
        self.tool_group.setExclusive(not is_mode1)

        if is_mode2:
            self.btn_layer_setup.setEnabled(self.image_path is not None)

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "О программе",
            "Анализ клеточных ядер\nВерсия 2.0\n\n"
            "Программа для детекции и анализа ядер на гистологических срезах.\n"
            "Поддерживает StarDist и CellPose.",
        )

    def _on_roi_mode_changed(self, index: int) -> None:
        mode = self.roi_mode_combo.currentData()
        self.square_size_group.setVisible(mode == "square")
        if mode == "full":
            self._build_full_frame_roi_auto()

    def _open_square_roi_dialog(self) -> None:
        if not self.image_path:
            QMessageBox.warning(self, "Нет изображения", "Сначала откройте изображение")
            return

        dialog = SquareRoiDialog(
            self.image_path,
            enhancement_params=self.enhancement_params,
            pixels_per_mm=self.pixels_per_mm,
            parent=self,
        )
        if dialog.exec_() == QDialog.Accepted:
            points = dialog.get_square_points()
            if points and len(points) >= 3:
                self._push_undo_state()
                points_disp = [
                    (float(x) / self.display_scale, float(y) / self.display_scale)
                    for x, y in points
                ]
                roi = {
                    "id": self.next_roi_id,
                    "type": "rectangle",
                    "points": points,
                }
                self.rois.append(roi)
                self._scene.add_roi_item(self.next_roi_id, "rectangle", points_disp)
                self.next_roi_id += 1
                self._refresh_table()
                self.proc_label.setText("Статус: квадратная ROI добавлена")

    def _build_full_frame_roi_auto(self) -> None:
        if not self.image_path:
            return
        try:
            image = backend.load_image(self.image_path)
        except Exception:
            return
        h, w = image.shape[:2]
        points = [
            (0.0, 0.0),
            (float(w - 1), 0.0),
            (float(w - 1), float(h - 1)),
            (0.0, float(h - 1)),
        ]
        self._push_undo_state()
        self.rois.clear()
        self.nuclei.clear()
        self._scene.clear_roi_items()
        self._scene.clear_nuclei_items()
        roi = {
            "id": 1,
            "type": "full_frame",
            "points": points,
        }
        self.rois.append(roi)
        self.next_roi_id = 2
        self._refresh_table()
        self.proc_label.setText("Статус: ROI по всему кадру")

    def _open_layer_setup(self) -> None:
        if not self.image_path:
            QMessageBox.warning(self, "Нет изображения", "Сначала откройте изображение")
            return
        detection_params = backend.get_detection_params()
        dialog = LayerSetupDialog(
            self.image_path,
            enhancement_params=self.enhancement_params,
            detection_params=detection_params,
            pixels_per_mm=self.pixels_per_mm,
            parent=self,
        )
        if dialog.exec_() == QDialog.Accepted:
            config = dialog.get_layer_config()
            num_layers = config.get("num_layers", 1)
            self.lbl_layer_info.setText(f"Слоёв: {num_layers}")

    def _run_layer_detection(self) -> None:
        if not self.image_path:
            QMessageBox.warning(self, "Нет изображения", "Сначала откройте изображение")
            return
        self.detect_nuclei()

    def _select_batch_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями")
        if folder:
            self._batch_folder_path = folder
            self.lbl_folder.setText(f"Папка: {Path(folder).name}")

    def _toggle_enhance_collapse(self) -> None:
        self._enhance_visible = not self._enhance_visible
        parent = self.enhance_collapse_btn.parent()
        for child in parent.findChildren(QWidget, None):
            if child is not self.enhance_collapse_btn and child.parent() is parent:
                child.setVisible(self._enhance_visible)
        self.enhance_collapse_btn.setText(
            "▾ Настройки цветокоррекции" if self._enhance_visible else "▸ Настройки цветокоррекции"
        )

    def _on_enhancement_changed(self) -> None:
        self.enhancement_params = self._collect_enhancement_from_sliders()
        self._update_enhancement_labels()
        self._refresh_image_preview_with_enhancement()

    def _collect_enhancement_from_sliders(self) -> dict:
        wb_active = self._white_ref_bgr is not None
        params = slider_values_to_enhancement(
            int(self.slider_saturation.value()),
            int(self.slider_brightness.value()),
            int(self.slider_contrast.value()),
            int(self.slider_sharpness.value()),
            white_balance=wb_active,
            white_balance_strength=int(self.slider_wb.value()),
        )
        if self._white_ref_bgr is not None:
            old = backend.normalize_enhancement_params(params)
            old["white_balance_ref_bgr"] = list(self._white_ref_bgr)
            return old
        return params

    def _update_enhancement_labels(self) -> None:
        params = self._collect_enhancement_from_sliders()
        self.lbl_saturation.setText(f"{params['saturation']:.2f}")
        self.lbl_brightness.setText(f"{params['brightness']:.0f}")
        self.lbl_contrast.setText(f"{params['contrast']:.2f}")
        self.lbl_sharpness.setText(f"{params['sharpness']:.2f}")
        self.lbl_wb.setText(f"{params['white_balance_strength']:.2f}")

    def reset_enhancement_settings(self) -> None:
        defaults = backend.get_default_enhancement_params()
        sat, bri, con, sha, _, wb_val = enhancement_to_slider_values(defaults)
        self.slider_saturation.setValue(sat)
        self.slider_brightness.setValue(bri)
        self.slider_contrast.setValue(con)
        self.slider_sharpness.setValue(sha)
        self.slider_wb.setValue(wb_val)
        self._white_ref_bgr = None
        self._on_enhancement_changed()

    def _set_enhancement_params(self, params: dict, refresh: bool = True) -> None:
        self.enhancement_params = backend.normalize_enhancement_params(params)
        sat, bri, con, sha, _, wb_val = enhancement_to_slider_values(self.enhancement_params)
        for slider in [self.slider_saturation, self.slider_brightness, self.slider_contrast, self.slider_sharpness]:
            slider.blockSignals(True)
        self.slider_saturation.setValue(sat)
        self.slider_brightness.setValue(bri)
        self.slider_contrast.setValue(con)
        self.slider_sharpness.setValue(sha)
        for slider in [self.slider_saturation, self.slider_brightness, self.slider_contrast, self.slider_sharpness]:
            slider.blockSignals(False)
        self.slider_wb.setValue(wb_val)
        self._update_enhancement_labels()
        if refresh:
            self._refresh_image_preview_with_enhancement()

    def _start_pick_white(self) -> None:
        if not self.image_path:
            QMessageBox.warning(self, "Нет изображения", "Сначала откройте изображение")
            return
        self._picking_for_wb = True
        self._picking_for_hes = False
        self._scene.start_pick_point()
        self._view.setCursor(Qt.CrossCursor)
        self.proc_label.setText("Статус: кликните на белую область фона")

    def _start_pick_nucleus(self) -> None:
        if not self.image_path:
            QMessageBox.warning(self, "Нет изображения", "Сначала откройте изображение")
            return
        self._picking_for_wb = False
        self._picking_for_hes = True
        self._scene.start_pick_point()
        self._view.setCursor(Qt.CrossCursor)
        self.proc_label.setText("Статус: кликните на центр ядра")

    def _on_point_picked(self, x: float, y: float) -> None:
        self._view.setCursor(Qt.ArrowCursor)
        if not self.image_path:
            return
        try:
            image = backend.load_image(self.image_path)
            h, w = image.shape[:2]
            px = int(round(x))
            py = int(round(y))
            px = max(0, min(px, w - 1))
            py = max(0, min(py, h - 1))

            if self._picking_for_wb:
                self._picking_for_wb = False
                ref_bgr = [float(image[py, px, c]) for c in range(3)]
                self._white_ref_bgr = ref_bgr
                self._on_enhancement_changed()
                self.btn_pick_white.setText(
                    f"Белая точка: ({ref_bgr[2]:.0f},{ref_bgr[1]:.0f},{ref_bgr[0]:.0f})"
                )
                self.proc_label.setText("Статус: баланс белого установлен")
            elif self._picking_for_hes:
                self._picking_for_hes = False
                half = 15
                y1 = max(0, py - half)
                y2 = min(h, py + half)
                x1 = max(0, px - half)
                x2 = min(w, px + half)
                patch = image[y1:y2, x1:x2]
                lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
                pixels = lab.reshape(-1, 3)
                mean_lab = pixels.mean(axis=0)
                std_lab = pixels.std(axis=0) + 1e-6
                backend.set_custom_hes_reference(mean_lab, std_lab)
                self.lbl_hes_status.setText(
                    f"H&E: ядро ({px},{py})  L={mean_lab[0]:.0f} "
                    f"A={mean_lab[1]:.0f} B={mean_lab[2]:.0f}"
                )
                self.proc_label.setText("Статус: H&E эталон обновлён по ядру")
        except Exception as exc:
            self._picking_for_wb = False
            self._picking_for_hes = False
            QMessageBox.critical(self, "Ошибка", str(exc))

    def _reset_hes_reference(self) -> None:
        backend.clear_custom_hes_reference()
        self.lbl_hes_status.setText("")
        self.proc_label.setText("Статус: H&E эталон сброшен к стандартному")

    def _push_undo_state(self) -> None:
        if self._restoring_state:
            return
        snapshot = {
            "image_path": self.image_path,
            "display_scale": self.display_scale,
            "pixels_per_mm": self.pixels_per_mm,
            "enhancement_params": copy.deepcopy(self.enhancement_params),
            "rois": copy.deepcopy(self.rois),
            "nuclei": copy.deepcopy(self.nuclei),
            "next_roi_id": self.next_roi_id,
        }
        self.undo_stack.append(snapshot)
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)

    def undo(self) -> None:
        if not self.undo_stack:
            return

        snapshot = self.undo_stack.pop()
        self._restoring_state = True
        try:
            self.image_path = snapshot["image_path"]
            self.display_scale = snapshot["display_scale"]
            self.pixels_per_mm = snapshot["pixels_per_mm"]
            self.enhancement_params = copy.deepcopy(
                snapshot.get("enhancement_params", backend.get_default_enhancement_params())
            )
            self.rois = copy.deepcopy(snapshot["rois"])
            self.nuclei = copy.deepcopy(snapshot["nuclei"])
            self.next_roi_id = snapshot["next_roi_id"]

            sat, bri, con, sha = enhancement_to_slider_values(self.enhancement_params)
            self.slider_saturation.blockSignals(True)
            self.slider_brightness.blockSignals(True)
            self.slider_contrast.blockSignals(True)
            self.slider_sharpness.blockSignals(True)
            self.slider_saturation.setValue(sat)
            self.slider_brightness.setValue(bri)
            self.slider_contrast.setValue(con)
            self.slider_sharpness.setValue(sha)
            self.slider_saturation.blockSignals(False)
            self.slider_brightness.blockSignals(False)
            self.slider_contrast.blockSignals(False)
            self.slider_sharpness.blockSignals(False)
            self._update_enhancement_labels()

            if self.image_path and os.path.exists(self.image_path):
                img, self.display_scale = backend.load_display_image(self.image_path)
                img_enhanced = backend.apply_image_enhancement(img, self.enhancement_params)
                self._scene.set_image_pixmap(self._bgr_to_pixmap(img_enhanced))
                self._view.resetTransform()
            else:
                self._scene.set_image_pixmap(None)
                self.image_path = None

            self._redraw_roi_items()
            self._redraw_nuclei_items()
            self._refresh_table()
            self._update_scale_status()
            self.proc_label.setText("Статус: отменено последнее действие")
        finally:
            self._restoring_state = False

    def set_tool(self, mode: ToolMode) -> None:
        self._scene.set_tool_mode(mode)
        self._view.setCursor(Qt.CrossCursor)
        self._view.viewport().setCursor(Qt.CrossCursor)

    def cancel_drawing(self) -> None:
        self._scene.cancel_current_drawing()
        self._scene.cancel_pick_point()
        self._picking_for_wb = False
        self._picking_for_hes = False
        self._view.setCursor(Qt.ArrowCursor)
        self.tool_group.setExclusive(False)
        self.btn_rect.setChecked(False)
        self.btn_poly.setChecked(False)
        self.btn_line.setChecked(False)
        self.tool_group.setExclusive(True)
        self.set_tool(ToolMode.NONE)

    def _ensure_scale_ready(self) -> bool:
        if self.pixels_per_mm is not None and self.pixels_per_mm > 0:
            return True
        self.activate_calibration()
        return self.pixels_per_mm is not None and self.pixels_per_mm > 0

    def _enforce_initial_scale(self) -> None:
        if self._initial_scale_completed:
            return
        ok = self._ensure_scale_ready()
        if not ok:
            QMessageBox.warning(
                self,
                "Требуется масштаб",
                "Без калибровки масштаба работа приложения невозможна.",
            )
            self.close()
            return
        self._initial_scale_completed = True

    def activate_calibration(self) -> None:
        candidate = "Фотография линейки.jpg"
        initial_path = None
        if os.path.exists(candidate):
            initial_path = candidate
        elif self.image_path and os.path.exists(self.image_path):
            initial_path = self.image_path

        dialog = ScaleCalibrationDialog(self, initial_path=initial_path)
        if dialog.exec_() != QDialog.Accepted:
            return
        if dialog.pixels_per_mm is None:
            return

        self._push_undo_state()
        self.pixels_per_mm = dialog.pixels_per_mm
        self._initial_scale_completed = True
        self._update_scale_status()
        self._refresh_table()
        self.proc_label.setText("Статус: масштаб задан")

    def open_detection_params(self) -> None:
        dialog = DetectionParamsDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return
        self._update_detector_status()
        self.proc_label.setText("Статус: параметры детекции обновлены")

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Открыть изображение",
            "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg)",
        )
        if not path:
            return

        try:
            display_image, scale = backend.load_display_image(path)
            enhanced_display = backend.apply_image_enhancement(display_image, self.enhancement_params)
            pixmap = self._bgr_to_pixmap(enhanced_display)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))
            return

        self._push_undo_state()

        self.image_path = path
        self.display_scale = scale
        self.rois.clear()
        self.nuclei.clear()
        self.metrics_rows.clear()
        self.next_roi_id = 1

        self._scene.set_image_pixmap(pixmap)
        self._view.resetTransform()
        self.table.setRowCount(0)

        self._update_scale_status()
        self.proc_label.setText("Статус: изображение загружено")

        if self._current_tab == 0 and self.roi_mode_combo.currentData() == "full":
            self._build_full_frame_roi_auto()

        self.btn_layer_setup.setEnabled(True)

    def _refresh_image_preview_with_enhancement(self) -> None:
        if not self.image_path or not os.path.exists(self.image_path):
            return

        display_image, scale = backend.load_display_image(self.image_path)
        enhanced_display = backend.apply_image_enhancement(display_image, self.enhancement_params)
        self.display_scale = scale
        self._scene.set_image_pixmap(self._bgr_to_pixmap(enhanced_display))
        self._view.resetTransform()
        self._redraw_roi_items()
        self._redraw_nuclei_items()

    def run_batch_processing(self) -> None:
        if self._det_thread is not None:
            QMessageBox.information(
                self,
                "Пакетная обработка",
                "Сначала дождитесь завершения текущей детекции",
            )
            return

        folder = self._batch_folder_path
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "Нет папки", "Выберите папку с изображениями")
            return

        image_paths = self._list_images_in_folder(folder)
        if not image_paths:
            QMessageBox.warning(self, "Нет изображений", "В выбранной папке нет изображений")
            return

        mode_key = str(self.batch_mode_combo.currentData())

        color_dialog = ColorTuningDialog(
            image_path=str(image_paths[0]),
            initial_params=self.enhancement_params,
            parent=self,
        )
        if color_dialog.exec_() != QDialog.Accepted or color_dialog.result_params is None:
            return
        batch_enhancement = backend.normalize_enhancement_params(color_dialog.result_params)
        self._set_enhancement_params(batch_enhancement, refresh=True)

        if backend.get_loaded_model_info() is None:
            default_info = backend.get_default_detector_info()
            if default_info.get("type") == "unavailable":
                QMessageBox.warning(
                    self,
                    "Нейросеть недоступна",
                    "Предобученная модель недоступна. "
                    "Установите необходимые зависимости или загрузите .pt/.onnx модель.",
                )
                return

        rois_by_file = self._collect_rois_for_batch(
            image_paths=image_paths,
            mode_key=mode_key,
            enhancement_params=batch_enhancement,
        )
        if rois_by_file is None:
            self.proc_label.setText("Статус: пакетная обработка отменена")
            return

        if self._batch_thread is not None:
            QMessageBox.information(self, "Пакетная обработка", "Пакетная обработка уже выполняется")
            return

        self._pending_batch_folder_name = Path(folder).name
        self._pending_batch_total_files = len(image_paths)

        self._batch_progress = QProgressDialog(
            "Выполняется пакетная детекция...",
            "Отмена",
            0,
            len(image_paths),
            self,
        )
        self._batch_progress.setWindowTitle("Пакетная обработка")
        self._batch_progress.setMinimumDuration(0)
        self._batch_progress.setValue(0)
        self._batch_progress.setAutoClose(False)
        self._batch_progress.setAutoReset(False)

        self._batch_thread = QThread(self)
        detection_params = backend.get_detection_params()
        model_info = backend.get_loaded_model_info()
        custom_model_path = str(model_info.get("path")) if model_info else None
        self._batch_worker = BatchDetectionWorker(
            image_paths=[str(p) for p in image_paths],
            rois_by_file=rois_by_file,
            pixels_per_mm=float(self.pixels_per_mm or 0),
            enhancement_params=batch_enhancement,
            detection_params=detection_params,
            custom_model_path=custom_model_path,
        )
        self._batch_worker.moveToThread(self._batch_thread)

        self._batch_thread.started.connect(self._batch_worker.run)
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.finished.connect(self._on_batch_finished)
        self._batch_worker.failed.connect(self._on_batch_failed)
        self._batch_worker.canceled.connect(self._on_batch_canceled)
        self._batch_progress.canceled.connect(self._batch_worker.request_cancel)

        self._batch_worker.finished.connect(self._batch_thread.quit)
        self._batch_worker.failed.connect(self._batch_thread.quit)
        self._batch_worker.canceled.connect(self._batch_thread.quit)
        self._batch_thread.finished.connect(self._cleanup_batch_thread)

        self.btn_batch_run.setEnabled(False)
        self.btn_detect.setEnabled(False)
        self.proc_label.setText("Статус: запущена пакетная детекция")
        self._batch_thread.start()

    def _list_images_in_folder(self, folder_path: str) -> list[Path]:
        folder = Path(folder_path)
        return [
            p for p in sorted(folder.iterdir())
            if p.is_file() and p.suffix.lower() in backend.SUPPORTED_IMAGE_FORMATS
        ]

    def _build_full_frame_roi(self, image_path: str) -> list[dict]:
        image = backend.load_image(image_path)
        h, w = image.shape[:2]
        points = [
            (0.0, 0.0),
            (float(w - 1), 0.0),
            (float(w - 1), float(h - 1)),
            (0.0, float(h - 1)),
        ]
        return [
            {
                "id": 1,
                "name": "Весь кадр",
                "type": "full_frame",
                "points": points,
            }
        ]

    def _collect_rois_for_batch(
        self,
        image_paths: list[Path],
        mode_key: str,
        enhancement_params: dict,
    ) -> dict[str, list[dict]] | None:
        rois_by_file: dict[str, list[dict]] = {}
        if mode_key == BATCH_MODE_FULL:
            for image_path in image_paths:
                rois_by_file[str(image_path)] = self._build_full_frame_roi(str(image_path))
            return rois_by_file

        for index, image_path in enumerate(image_paths, start=1):
            self.proc_label.setText(
                f"Статус: разметка ROI {index}/{len(image_paths)}"
            )
            dialog = RoiAnnotationDialog(
                image_path=str(image_path),
                mode_key=mode_key,
                enhancement_params=enhancement_params,
                name_suggestions=self.roi_name_history,
                parent=self,
            )
            if dialog.exec_() != QDialog.Accepted:
                return None

            rois_by_file[str(image_path)] = copy.deepcopy(dialog.rois)
            for roi_name in dialog.new_names:
                if roi_name not in self.roi_name_history:
                    self.roi_name_history.append(roi_name)
        return rois_by_file

    def _on_batch_progress(self, current: int, total: int, file_name: str) -> None:
        if self._batch_progress is not None:
            self._batch_progress.setMaximum(total)
            self._batch_progress.setValue(current)
            self._batch_progress.setLabelText(
                f"Выполняется пакетная детекция...\n{current}/{total}: {file_name}"
            )

    def _on_batch_finished(self, batch_rows: list[dict]) -> None:
        if not batch_rows:
            QMessageBox.warning(self, "Нет результатов", "После обработки не получено данных для экспорта")
            return

        default_name = f"пакет_{self._pending_batch_folder_name}.csv"
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить пакетный отчёт",
            default_name,
            "CSV (*.csv);;Excel (*.xlsx)",
        )
        if not output_path:
            return

        try:
            backend.export_batch_results(batch_rows, output_path)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка экспорта", str(exc))
            return

        self.proc_label.setText(f"Статус: пакетная обработка завершена")

    def _on_batch_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Ошибка пакетной детекции", message)

    def _on_batch_canceled(self) -> None:
        self.proc_label.setText("Статус: пакетная обработка отменена")

    def _cleanup_batch_thread(self) -> None:
        if self._batch_progress is not None:
            self._batch_progress.close()
            self._batch_progress.deleteLater()
            self._batch_progress = None

        if self._batch_worker is not None:
            self._batch_worker.deleteLater()
            self._batch_worker = None

        if self._batch_thread is not None:
            self._batch_thread.deleteLater()
            self._batch_thread = None

        self.btn_batch_run.setEnabled(True)
        self.btn_detect.setEnabled(True)

    def _on_cursor_moved(self, x: float, y: float) -> None:
        ox = x * self.display_scale
        oy = y * self.display_scale
        text = f"x {ox:.0f}   y {oy:.0f}"
        self.coord_label.setText(text)
        if hasattr(self, "coord_overlay"):
            self.coord_overlay.setText(text)
            self.coord_overlay.adjustSize()

    def _on_roi_created(self, payload: dict) -> None:
        if not self.image_path:
            return

        points_display = payload.get("points", [])
        roi_type = payload.get("type", "polygon")
        if len(points_display) < 3:
            return

        self._push_undo_state()

        points_orig = [
            (float(x) * self.display_scale, float(y) * self.display_scale)
            for x, y in points_display
        ]
        roi = {
            "id": self.next_roi_id,
            "type": roi_type,
            "points": points_orig,
        }

        self.rois.append(roi)
        self._scene.add_roi_item(self.next_roi_id, roi_type, points_display)
        self.next_roi_id += 1

        self._refresh_table()
        self.proc_label.setText("Статус: область добавлена")

    def _on_line_created(self, line: tuple) -> None:
        if not self.image_path:
            return

        real_mm, ok = QInputDialog.getDouble(
            self,
            "Калибровка",
            "Реальная длина линии (мм):",
            1.0,
            0.0001,
            1_000_000.0,
            4,
        )
        if not ok:
            return

        (x1, y1), (x2, y2) = line
        line_orig = (
            (x1 * self.display_scale, y1 * self.display_scale),
            (x2 * self.display_scale, y2 * self.display_scale),
        )

        try:
            image = backend.load_image(self.image_path)
            ppm = backend.calibrate_scale(image, line_orig, real_mm)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка калибровки", str(exc))
            return

        self._push_undo_state()
        self.pixels_per_mm = ppm
        self._update_scale_status()
        self._refresh_table()
        self.proc_label.setText("Статус: масштаб задан")

    def detect_nuclei(self) -> None:
        if not self.image_path:
            QMessageBox.warning(self, "Нет изображения", "Сначала откройте изображение")
            return
        if self._batch_thread is not None:
            QMessageBox.information(self, "Детекция", "Пакетная обработка уже запущена")
            return

        if backend.get_loaded_model_info() is None:
            default_info = backend.get_default_detector_info()
            if default_info.get("type") == "unavailable":
                QMessageBox.warning(self, "Нейросеть недоступна", "Модель недоступна")
                return

        if self._det_thread is not None:
            return

        self.btn_detect.setEnabled(False)
        self.proc_label.setText("Статус: выполняется детекция ядер...")

        self._det_thread = QThread(self)
        detection_params = backend.get_detection_params()
        model_info = backend.get_loaded_model_info()
        custom_model_path = str(model_info.get("path")) if model_info else None
        self._det_worker = DetectionWorker(
            self.image_path,
            self.enhancement_params,
            detection_params=detection_params,
            custom_model_path=custom_model_path,
        )
        self._det_worker.moveToThread(self._det_thread)

        self._det_thread.started.connect(self._det_worker.run)
        self._det_worker.finished.connect(self._on_detect_finished)
        self._det_worker.failed.connect(self._on_detect_failed)

        self._det_worker.finished.connect(self._det_thread.quit)
        self._det_worker.failed.connect(self._det_thread.quit)
        self._det_thread.finished.connect(self._cleanup_detection_thread)

        self._det_thread.start()

    def _cleanup_detection_thread(self) -> None:
        if self._det_worker is not None:
            self._det_worker.deleteLater()
            self._det_worker = None
        if self._det_thread is not None:
            self._det_thread.deleteLater()
            self._det_thread = None
        self.btn_detect.setEnabled(True)

    def _on_detect_finished(self, nuclei: list[dict]) -> None:
        self._push_undo_state()
        self.nuclei = nuclei
        self._redraw_nuclei_items()
        self._refresh_table()
        self.proc_label.setText(f"Статус: найдено ядер: {len(nuclei)}")

    def _on_detect_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Ошибка детекции", message)

    def _redraw_nuclei_items(self) -> None:
        if not self.nuclei:
            self._scene.clear_nuclei_items()
            return

        nuclei_display: list[dict] = []
        for nuc in self.nuclei:
            center = nuc.get("center")
            contour = nuc.get("contour", [])
            if center is None:
                continue
            nuclei_display.append(
                {
                    "center": (
                        float(center[0]) / self.display_scale,
                        float(center[1]) / self.display_scale,
                    ),
                    "contour": [
                        (float(x) / self.display_scale, float(y) / self.display_scale)
                        for x, y in contour
                    ],
                }
            )
        self._scene.set_nuclei_items(nuclei_display)

    def _redraw_roi_items(self) -> None:
        self._scene.clear_roi_items()
        for roi in self.rois:
            points_disp = [
                (float(x) / self.display_scale, float(y) / self.display_scale)
                for x, y in roi.get("points", [])
            ]
            self._scene.add_roi_item(int(roi["id"]), str(roi.get("type", "polygon")), points_disp)

    def _update_scale_status(self) -> None:
        text = f"Масштаб: {self.pixels_per_mm:.4f} px/mm" if self.pixels_per_mm else "Масштаб: не задан"
        self.scale_label.setText(text)
        self.scale_label_t2.setText(text)
        self.scale_label_t3.setText(text)

    def _refresh_table(self) -> None:
        self.metrics_rows = []
        for roi in self.rois:
            row = backend.build_roi_metrics(roi, self.nuclei, self.pixels_per_mm)
            self.metrics_rows.append(row)

        self.table.setRowCount(len(self.metrics_rows))
        for r, row in enumerate(self.metrics_rows):
            values = [
                row.get("ROI ID"),
                row.get("Тип"),
                row.get("Площадь (мм\u00b2)"),
                row.get("Количество ядер"),
                row.get("Плотность (ядра/мм\u00b2)"),
            ]
            for c, value in enumerate(values):
                if value is None:
                    text = "-"
                elif isinstance(value, float):
                    text = f"{value:.4f}"
                else:
                    text = str(value)
                self.table.setItem(r, c, QTableWidgetItem(text))

        count = len(self.nuclei)
        count_text = str(count) if count else "\u2014"
        self.lbl_nuclei_count_big.setText(count_text)
        self.lbl_nuclei_count.setText(f"Найдено: {count}" if count else "")

    def export_results(self) -> None:
        if not self.metrics_rows:
            QMessageBox.warning(self, "Нет данных", "Нет ROI для экспорта")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Экспорт результатов",
            "results.csv",
            "CSV (*.csv);;Excel (*.xlsx)",
        )
        if not path:
            return

        try:
            backend.export_results(self.metrics_rows, path)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка экспорта", str(exc))
            return

        self.proc_label.setText("Статус: экспорт завершён")

    def _clear_rois(self) -> None:
        if not self.rois:
            return
        self._push_undo_state()
        self.rois.clear()
        self._scene.clear_roi_items()
        self._refresh_table()
        self.proc_label.setText("Статус: все ROI удалены")

    def delete_selected_rois(self) -> None:
        selected_ids = self._scene.selected_roi_ids()
        if not selected_ids:
            return

        self._push_undo_state()
        selected_set = set(selected_ids)
        self.rois = [roi for roi in self.rois if int(roi["id"]) not in selected_set]
        for roi_id in selected_set:
            self._scene.remove_roi_item(roi_id)

        self._refresh_table()
        self.proc_label.setText("Статус: область удалена")

    def _update_detector_status(self) -> None:
        model_info = backend.get_loaded_model_info()
        if model_info is not None:
            file_name = Path(model_info["path"]).name
            self.detector_label.setText(f"Детектор: {file_name} ({model_info['runtime']})")
            return

        default_info = backend.get_default_detector_info()
        params = backend.get_detection_params()
        mode_ru = preprocess_mode_to_russian(str(params.get("preprocess_mode", "")))
        upscale = float(params.get("upscale_factor", 1.5))
        stain_norm = bool(params.get("stain_norm_enabled", True))
        norm_mark = "норм. вкл." if stain_norm else "норм. выкл."
        self.detector_label.setText(
            f"Детектор: {default_info['name']} | увер. {params['prob_thresh']:.2f} "
            f"| раздел. {params['nms_thresh']:.2f} | upscale x{upscale:.2f} "
            f"| {norm_mark} | режим: {mode_ru}"
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_coord_overlay()

    def _position_coord_overlay(self) -> None:
        if not hasattr(self, "coord_overlay") or not hasattr(self, "_view"):
            return
        vp = self._view.viewport()
        self.coord_overlay.adjustSize()
        self.coord_overlay.move(8, vp.height() - self.coord_overlay.height() - 8)
        self.coord_overlay.raise_()

    def _bgr_to_pixmap(self, image_bgr) -> QPixmap:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)
