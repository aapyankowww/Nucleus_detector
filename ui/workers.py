from __future__ import annotations

import multiprocessing as mp
import queue as queue_module

import backend
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot


def enhancement_to_slider_values(params: dict) -> tuple[int, int, int, int]:
    saturation = int(round(float(params.get("saturation", 1.0)) * 100.0))
    brightness = int(round(float(params.get("brightness", 0.0))))
    contrast = int(round(float(params.get("contrast", 1.0)) * 100.0))
    sharpness = int(round(float(params.get("sharpness", 1.0)) * 100.0))
    return saturation, brightness, contrast, sharpness


def slider_values_to_enhancement(
    saturation_value: int,
    brightness_value: int,
    contrast_value: int,
    sharpness_value: int,
) -> dict:
    return backend.normalize_enhancement_params(
        {
            "saturation": float(saturation_value) / 100.0,
            "brightness": float(brightness_value),
            "contrast": float(contrast_value) / 100.0,
            "sharpness": float(sharpness_value) / 100.0,
        }
    )


class DetectionWorker(QObject):
    finished = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(
        self,
        image_path: str,
        enhancement_params: dict | None = None,
        detection_params: dict | None = None,
        custom_model_path: str | None = None,
    ) -> None:
        super().__init__()
        self.image_path = image_path
        self.enhancement_params = backend.normalize_enhancement_params(enhancement_params)
        self.detection_params = dict(detection_params or backend.get_detection_params())
        self.custom_model_path = str(custom_model_path).strip() if custom_model_path else ""

    @pyqtSlot()
    def run(self) -> None:
        proc = None
        event_queue = None
        try:
            ctx = mp.get_context("spawn")
            event_queue = ctx.Queue()
            payload = {
                "mode": "single",
                "image_path": self.image_path,
                "enhancement_params": self.enhancement_params,
                "detection_params": self.detection_params,
                "custom_model_path": self.custom_model_path,
            }
            proc = ctx.Process(target=backend.run_detection_job, args=(payload, event_queue))
            proc.start()

            result_nuclei: list[dict] | None = None
            error_message: str | None = None

            while True:
                try:
                    message = event_queue.get(timeout=0.2)
                except queue_module.Empty:
                    message = None

                if isinstance(message, dict):
                    mtype = message.get("type")
                    if mtype == "result":
                        result_nuclei = message.get("nuclei", [])
                        break
                    if mtype == "error":
                        error_message = str(message.get("message", "Ошибка детекции в дочернем процессе"))
                        break

                if proc is not None and not proc.is_alive():
                    break

            while event_queue is not None:
                try:
                    message = event_queue.get_nowait()
                except queue_module.Empty:
                    break
                if isinstance(message, dict):
                    mtype = message.get("type")
                    if mtype == "result" and result_nuclei is None:
                        result_nuclei = message.get("nuclei", [])
                    elif mtype == "error" and error_message is None:
                        error_message = str(message.get("message", "Ошибка детекции в дочернем процессе"))

            if proc is not None:
                proc.join(timeout=1.0)

            if result_nuclei is not None:
                self.finished.emit(result_nuclei)
                return

            if error_message is not None:
                self.failed.emit(error_message)
                return

            exit_code = proc.exitcode if proc is not None else None
            if exit_code is None:
                self.failed.emit("Дочерний процесс детекции завершился без кода выхода")
            elif exit_code != 0:
                self.failed.emit(
                    "Дочерний процесс детекции аварийно завершился "
                    f"(код {exit_code}). Проверьте совместимость библиотек TensorFlow/LLVM."
                )
            else:
                self.failed.emit("Дочерний процесс детекции завершился без результата")
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if proc is not None and proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)


class BatchDetectionWorker(QObject):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(list)
    failed = pyqtSignal(str)
    canceled = pyqtSignal()

    def __init__(
        self,
        image_paths: list[str],
        rois_by_file: dict[str, list[dict]],
        pixels_per_mm: float,
        enhancement_params: dict | None = None,
        detection_params: dict | None = None,
        custom_model_path: str | None = None,
    ) -> None:
        super().__init__()
        self.image_paths = image_paths
        self.rois_by_file = rois_by_file
        self.pixels_per_mm = float(pixels_per_mm)
        self.enhancement_params = backend.normalize_enhancement_params(enhancement_params)
        self.detection_params = dict(detection_params or backend.get_detection_params())
        self.custom_model_path = str(custom_model_path).strip() if custom_model_path else ""
        self._cancel_requested = False

    @pyqtSlot()
    def request_cancel(self) -> None:
        self._cancel_requested = True

    @pyqtSlot()
    def run(self) -> None:
        proc = None
        event_queue = None
        try:
            ctx = mp.get_context("spawn")
            event_queue = ctx.Queue()
            payload = {
                "mode": "batch",
                "image_paths": self.image_paths,
                "rois_by_file": self.rois_by_file,
                "pixels_per_mm": self.pixels_per_mm,
                "enhancement_params": self.enhancement_params,
                "detection_params": self.detection_params,
                "custom_model_path": self.custom_model_path,
            }
            proc = ctx.Process(target=backend.run_detection_job, args=(payload, event_queue))
            proc.start()

            batch_rows: list[dict] | None = None
            error_message: str | None = None

            while True:
                if self._cancel_requested:
                    if proc is not None and proc.is_alive():
                        proc.terminate()
                        proc.join(timeout=1.0)
                    self.canceled.emit()
                    return

                try:
                    message = event_queue.get(timeout=0.2)
                except queue_module.Empty:
                    message = None

                if isinstance(message, dict):
                    mtype = message.get("type")
                    if mtype == "progress":
                        current = int(message.get("current", 0))
                        total = int(message.get("total", 0))
                        file_name = str(message.get("file_name", ""))
                        self.progress.emit(current, total, file_name)
                    elif mtype == "result":
                        batch_rows = message.get("rows", [])
                        break
                    elif mtype == "error":
                        error_message = str(
                            message.get("message", "Ошибка пакетной детекции в дочернем процессе")
                        )
                        break

                if proc is not None and not proc.is_alive():
                    break

            while event_queue is not None:
                try:
                    message = event_queue.get_nowait()
                except queue_module.Empty:
                    break
                if isinstance(message, dict):
                    mtype = message.get("type")
                    if mtype == "progress":
                        current = int(message.get("current", 0))
                        total = int(message.get("total", 0))
                        file_name = str(message.get("file_name", ""))
                        self.progress.emit(current, total, file_name)
                    elif mtype == "result" and batch_rows is None:
                        batch_rows = message.get("rows", [])
                    elif mtype == "error" and error_message is None:
                        error_message = str(
                            message.get("message", "Ошибка пакетной детекции в дочернем процессе")
                        )

            if proc is not None:
                proc.join(timeout=1.0)

            if self._cancel_requested:
                self.canceled.emit()
                return

            if batch_rows is not None:
                self.finished.emit(batch_rows)
                return

            if error_message is not None:
                self.failed.emit(error_message)
                return

            exit_code = proc.exitcode if proc is not None else None
            if exit_code is None:
                self.failed.emit("Дочерний процесс пакетной детекции завершился без кода выхода")
            elif exit_code != 0:
                self.failed.emit(
                    "Дочерний процесс пакетной детекции аварийно завершился "
                    f"(код {exit_code}). Проверьте совместимость библиотек TensorFlow/LLVM."
                )
            else:
                self.failed.emit("Дочерний процесс пакетной детекции завершился без результата")
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if proc is not None and proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)
