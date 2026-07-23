"""번호판 OCR TensorRT + CTC 디코딩 (Paddle RecResizeImg 3x48x320 정합)."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from trt_engine import TensorRTEngine


def load_plate_dict(dict_path: str | Path) -> list[str]:
    path = Path(dict_path)
    if not path.exists():
        raise FileNotFoundError(f"OCR dictionary 파일 없음: {path}")
    chars = [ln.rstrip("\n\r") for ln in path.read_text(encoding="utf-8").splitlines()]
    while chars and chars[-1] == "":
        chars.pop()
    if not chars:
        raise RuntimeError(f"빈 dictionary: {path}")
    return chars


def resize_norm_img(img_bgr: np.ndarray, img_h: int = 48, img_w: int = 320) -> np.ndarray:
    """
    PaddleOCR RecResizeImg 스타일:
      - 비율 유지 resize, 높이를 img_h에 맞춤
      - 폭이 img_w보다 작으면 오른쪽 padding
      - 크면 가로로 압축
      - RGB, NCHW, float32 [0,1]
    """
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("번호판 crop 크기 부족/빈 이미지")

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    ratio = w / float(h)
    resized_w = int(np.ceil(img_h * ratio))
    if resized_w > img_w:
        resized_w = img_w
    resized = cv2.resize(rgb, (resized_w, img_h), interpolation=cv2.INTER_LINEAR)
    pad = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    pad[:, :resized_w, :] = resized
    x = pad.astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None, ...]
    return np.ascontiguousarray(x)


def ctc_decode(logits: np.ndarray, charset: list[str]) -> tuple[str, float]:
    """
    logits: (T, C) 또는 (1, T, C) 또는 (1, C, T)
    blank index = 0 (Paddle CTCLabelDecode 관례)
    charset[i] corresponds to class i+1
    """
    arr = logits
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        # (C, T) -> (T, C)
        if arr.shape[0] == len(charset) + 1 or arr.shape[0] < arr.shape[1]:
            # heuristic: if first dim looks like classes
            if arr.shape[0] <= arr.shape[1]:
                arr = arr.T

    if arr.ndim != 2:
        raise RuntimeError(f"CTC output shape 불일치: {logits.shape}")

    # softmax if not normalized
    if arr.min() < 0 or arr.max() > 1.5:
        arr = arr - arr.max(axis=1, keepdims=True)
        exp = np.exp(arr)
        probs = exp / exp.sum(axis=1, keepdims=True)
    else:
        probs = arr

    idxs = probs.argmax(axis=1)
    confs = probs.max(axis=1)

    blank = 0
    out_chars = []
    out_conf = []
    prev = None
    for i, conf in zip(idxs, confs):
        i = int(i)
        if i == blank:
            prev = i
            continue
        if i == prev:
            continue
        # charset index = i - 1
        ci = i - 1
        if ci < 0 or ci >= len(charset):
            prev = i
            continue
        out_chars.append(charset[ci])
        out_conf.append(float(conf))
        prev = i

    text = "".join(out_chars)
    confidence = float(np.mean(out_conf)) if out_conf else 0.0
    return text, confidence


class PlateTensorRTRecognizer:
    def __init__(
        self,
        engine_path: str | Path,
        dict_path: str | Path,
        img_h: int = 48,
        img_w: int = 320,
    ):
        self.engine = TensorRTEngine(engine_path)
        self.charset = load_plate_dict(dict_path)
        self.img_h = img_h
        self.img_w = img_w
        self.input_name = self.engine.input_names[0]
        self.last_timing = {}
        self._logged_output_shape = False

    def preprocess(self, plate_crop_bgr: np.ndarray) -> np.ndarray:
        h, w = plate_crop_bgr.shape[:2]
        if w < 1 or h < 1:
            raise ValueError("번호판 crop 크기 부족")
        x = resize_norm_img(plate_crop_bgr, self.img_h, self.img_w)
        dtype = self.engine.bindings[self.input_name]["dtype"]
        if x.dtype != dtype:
            x = x.astype(dtype)
        return x

    def predict(self, plate_crop_bgr: np.ndarray) -> dict:
        t0 = time.perf_counter()
        blob = self.preprocess(plate_crop_bgr)
        t1 = time.perf_counter()
        outputs = self.engine.infer({self.input_name: blob})
        t2 = time.perf_counter()

        out_name = self.engine.output_names[0]
        raw = outputs[out_name]
        if not self._logged_output_shape:
            print(f"[OCR] output '{out_name}' shape={raw.shape}")
            self._logged_output_shape = True
        text, conf = ctc_decode(raw, self.charset)
        t3 = time.perf_counter()

        self.last_timing = {
            "preprocess_ms": (t1 - t0) * 1000,
            "infer_ms": self.engine.last_infer_ms,
            "decode_ms": (t3 - t2) * 1000,
            "total_ms": (t3 - t0) * 1000,
        }
        return {"text": text, "confidence": conf}

    def close(self) -> None:
        self.engine.close()
