from __future__ import annotations


LIGHT_THEME_QSS = """
/* === Base === */
QMainWindow, QDialog, QWidget {
    background-color: #F0F0F0;
    color: #1E1E1E;
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}

/* === Menu bar === */
QMenuBar {
    background-color: #FFFFFF;
    border-bottom: 1px solid #D0D0D0;
    padding: 2px 0;
    font-size: 13px;
}
QMenuBar::item {
    padding: 4px 12px;
    background: transparent;
}
QMenuBar::item:selected {
    background-color: #E0E0E0;
    border-radius: 3px;
}
QMenu {
    background-color: #FFFFFF;
    border: 1px solid #C0C0C0;
    padding: 4px 0;
}
QMenu::item {
    padding: 6px 28px 6px 16px;
    font-size: 13px;
}
QMenu::item:selected {
    background-color: #0078D4;
    color: white;
}
QMenu::separator {
    height: 1px;
    background: #E0E0E0;
    margin: 4px 8px;
}

/* === Toolbar === */
QToolBar {
    background-color: #FFFFFF;
    border-bottom: 1px solid #D0D0D0;
    spacing: 0;
    padding: 2px 4px;
}
QToolBar QToolButton {
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 6px 16px;
    margin: 1px;
    font-size: 13px;
    color: #333333;
}
QToolBar QToolButton:hover {
    background-color: #E0E0E0;
    border-color: #C0C0C0;
}
QToolBar QToolButton:checked {
    background-color: #0078D4;
    color: white;
    border-color: #0078D4;
}

/* === Status bar === */
QStatusBar {
    background-color: #FFFFFF;
    border-top: 1px solid #D0D0D0;
    font-size: 12px;
    color: #555555;
    padding: 2px 8px;
}

/* === Left panel (sidebar) === */
QWidget#sidebar {
    background-color: #FAFAFA;
    border-right: 1px solid #D0D0D0;
}
QScrollArea#sidebar_scroll {
    border: none;
    background-color: #FAFAFA;
}
QWidget#sidebar_content {
    background-color: #FAFAFA;
}

/* === Sidebar groups === */
QWidget#group_widget {
    background-color: transparent;
}
QLabel#group_title {
    font-size: 11px;
    font-weight: bold;
    color: #555555;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 8px 12px 2px 12px;
    background-color: transparent;
}
QWidget#group_body {
    background-color: #FFFFFF;
    border: 1px solid #E0E0E0;
    border-radius: 5px;
    margin: 2px 8px 6px 8px;
    padding: 8px;
}

/* === Buttons in sidebar === */
QPushButton#btn_primary {
    background-color: #0078D4;
    border: none;
    border-radius: 4px;
    color: white;
    font-size: 13px;
    padding: 8px 16px;
    min-height: 36px;
}
QPushButton#btn_primary:hover {
    background-color: #106EBE;
}
QPushButton#btn_primary:disabled {
    background-color: #C0C0C0;
    color: #888888;
}
QPushButton#btn_secondary {
    background-color: #FFFFFF;
    border: 1px solid #C0C0C0;
    border-radius: 4px;
    color: #333333;
    padding: 6px 12px;
    min-height: 30px;
    font-size: 13px;
}
QPushButton#btn_secondary:hover {
    background-color: #F0F0F0;
    border-color: #0078D4;
    color: #0078D4;
}
QPushButton#btn_secondary:disabled {
    color: #AAAAAA;
    background-color: #F5F5F5;
    border-color: #DDDDDD;
}
QPushButton#btn_danger {
    background-color: #FFFFFF;
    border: 1px solid #D0D0D0;
    border-radius: 4px;
    color: #C42B1C;
    padding: 6px 12px;
    min-height: 30px;
    font-size: 13px;
}
QPushButton#btn_danger:hover {
    background-color: #FDE7E5;
    border-color: #C42B1C;
}

/* === Generic QPushButton === */
QPushButton {
    background-color: #FFFFFF;
    border: 1px solid #C0C0C0;
    border-radius: 4px;
    color: #333333;
    padding: 5px 10px;
    min-height: 28px;
    font-size: 13px;
}
QPushButton:hover {
    background-color: #F0F0F0;
    border-color: #0078D4;
}
QPushButton:pressed {
    background-color: #E0E0E0;
}
QPushButton:checked {
    background-color: #0078D4;
    color: white;
    border-color: #0078D4;
}
QPushButton:disabled {
    color: #AAAAAA;
    background-color: #F5F5F5;
    border-color: #DDDDDD;
}

/* === Canvas area (QGraphicsView) === */
QGraphicsView {
    border: none;
    background-color: #1E1E1E;
}

/* === Canvas toolbar === */
QWidget#canvas_toolbar {
    background-color: #F8F8F8;
    border-bottom: 1px solid #D0D0D0;
}
QLabel#canvas_label {
    font-size: 11px;
    color: #777777;
    padding: 0 4px;
    background-color: transparent;
}

/* === Right panel === */
QWidget#right_panel {
    background-color: #FAFAFA;
    border-left: 1px solid #D0D0D0;
}
QLabel#right_header {
    font-size: 10px;
    font-weight: bold;
    color: #888888;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 4px 0;
    background-color: transparent;
}
QLabel#nuclei_count_big {
    font-size: 48px;
    font-weight: bold;
    color: #0078D4;
    padding: 0 0 4px 0;
    background-color: transparent;
}
QLabel#status_text {
    font-size: 12px;
    color: #555555;
    padding: 2px 0;
    background-color: transparent;
}

/* === Table === */
QTableWidget {
    border: 1px solid #E0E0E0;
    border-radius: 4px;
    background-color: #FFFFFF;
    alternate-background-color: #F5F9FF;
    gridline-color: #E8E8E8;
    font-size: 12px;
    selection-background-color: #0078D4;
    selection-color: white;
}
QHeaderView::section {
    background-color: #F0F0F0;
    color: #333333;
    font-weight: bold;
    font-size: 11px;
    padding: 5px 8px;
    border: none;
    border-bottom: 1px solid #D0D0D0;
    border-right: 1px solid #E0E0E0;
}

/* === Separators === */
QFrame#vsep {
    color: #E0E0E0;
    max-width: 1px;
}
QFrame#hsep {
    color: #E0E0E0;
    max-height: 1px;
}

/* === Sliders === */
QSlider::groove:horizontal {
    height: 4px;
    background-color: #D0D0D0;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background-color: #0078D4;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal {
    background-color: #90CDF4;
    border-radius: 2px;
}

/* === ComboBox, SpinBox, LineEdit === */
QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {
    border: 1px solid #C0C0C0;
    border-radius: 4px;
    padding: 4px 8px;
    background-color: #FFFFFF;
    font-size: 13px;
    min-height: 24px;
}
QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus {
    border-color: #0078D4;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox QAbstractItemView {
    background-color: #FFFFFF;
    border: 1px solid #C0C0C0;
    selection-background-color: #0078D4;
    selection-color: white;
}

/* === CheckBox === */
QCheckBox {
    spacing: 6px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #B0B0B0;
    border-radius: 3px;
    background-color: white;
}
QCheckBox::indicator:checked {
    background-color: #0078D4;
    border-color: #0078D4;
}

/* === Scrollbars === */
QScrollBar:vertical {
    background: #F0F0F0;
    width: 8px;
    border: none;
}
QScrollBar::handle:vertical {
    background: #C0C0C0;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background: #A0A0A0;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    height: 8px;
    background: #F0F0F0;
    border: none;
}
QScrollBar::handle:horizontal {
    background: #C0C0C0;
    border-radius: 4px;
    min-width: 20px;
}
QScrollBar::handle:horizontal:hover {
    background: #A0A0A0;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}

/* === QGroupBox === */
QGroupBox {
    border: 1px solid #E0E0E0;
    border-radius: 5px;
    margin-top: 8px;
    padding: 14px 8px 6px 8px;
    font-weight: bold;
    font-size: 12px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    color: #555555;
}

/* === QDialog === */
QDialog {
    background-color: #F0F0F0;
}
QDialogButtonBox QPushButton {
    min-width: 80px;
    min-height: 30px;
}

/* === Cursor overlay on canvas === */
QLabel#coord_overlay {
    background-color: rgba(0, 0, 0, 150);
    color: white;
    font-size: 11px;
    padding: 3px 8px;
    border-radius: 3px;
}

/* === Layer legend === */
QLabel#layer_legend {
    font-size: 11px;
    padding: 2px 6px;
    border-radius: 3px;
}

/* === QProgressDialog === */
QProgressDialog {
    background-color: #FFFFFF;
    border: 1px solid #C0C0C0;
    min-width: 350px;
}
"""


def apply_light_theme(widget) -> None:
    widget.setStyleSheet(LIGHT_THEME_QSS)
