from __future__ import annotations

import cv2
import numpy as np


def enhance_nuclei_contrast(image_bgr: np.ndarray, method: str = "clahe") -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    work = image_bgr.copy()

    if method in ("clahe", "both"):
        lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
        l_ch = lab[..., 0]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_ch)
        lab[..., 0] = l_eq
        work = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if method in ("unsharp", "both"):
        blurred = cv2.GaussianBlur(work, (0, 0), sigmaX=1.5, sigmaY=1.5)
        amount = 1.0
        work = cv2.addWeighted(work, 1.0 + amount, blurred, -amount, 0.0)
        work = np.clip(work, 0, 255).astype(np.uint8)

    return work
