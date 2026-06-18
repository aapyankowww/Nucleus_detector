from __future__ import annotations

from ui.main_window import MainWindow


def create_app():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    window = MainWindow()
    return app, window
