"""OCR 고정 전처리 — Jetson / ONNX / TRT용 단일 경로 (48x320)."""

from __future__ import annotations

import cv2
import numpy as np

from export_config import OCR_H, OCR_W


def preprocess_ocr_fixed(
    crop_bgr: np.ndarray,
    height: int = OCR_H,
    width: int = OCR_W,
) -> np.ndarray:
    """
    BGR crop → NCHW float32 [1,3,H,W], 값 범위 [0, 1].

    RecResizeImg와 같이 목표 크기로 직접 resize (letterbox 아님).
    """
    if crop_bgr is None or crop_bgr.size == 0:
        raise ValueError("empty crop")

    img = cv2.resize(crop_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(chw, axis=0)


def preprocess_ocr_bgr_u8(
    crop_bgr: np.ndarray,
    height: int = OCR_H,
    width: int = OCR_W,
) -> np.ndarray:
    """디버그용: 리사이즈만 한 BGR uint8."""
    if crop_bgr is None or crop_bgr.size == 0:
        raise ValueError("empty crop")
    return cv2.resize(crop_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
