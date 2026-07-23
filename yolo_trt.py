"""YOLO TensorRT 탐지 (track 없음). letterbox 전처리 + NMS/엔드투엔드 출력 후처리."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from trt_engine import TensorRTEngine

# 기존 프로젝트 클래스 순서 (data.yaml / 학습 기준) — 임의 재정의 금지
DEFAULT_CLASS_NAMES = [
    "small_trash",
    "standing",
    "sitting",
    "lying",
    "bin_overflow",
    "bin_normal",
    "bag_outside",
    "licence",
]


def letterbox(
    image: np.ndarray,
    new_shape: int | tuple[int, int] = 640,
    color: tuple[int, int, int] = (114, 114, 114),
):
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    h, w = image.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape[0], new_shape[1], 3), color, dtype=np.uint8)
    top = (new_shape[0] - nh) // 2
    left = (new_shape[1] - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, r, left, top


def _xywh_to_xyxy(x: np.ndarray) -> np.ndarray:
    y = np.empty_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thres]
    return keep


class YoloTensorRTDetector:
    def __init__(
        self,
        engine_path: str | Path,
        input_size: int = 640,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        class_names: list[str] | None = None,
    ):
        self.engine = TensorRTEngine(engine_path)
        self.input_size = input_size
        self.conf_thres = confidence_threshold
        self.iou_thres = iou_threshold
        self.class_names = class_names or list(DEFAULT_CLASS_NAMES)
        self.input_name = self.engine.input_names[0]
        self.last_timing = {}
        self._logged_output_shape = False

    def preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        img, ratio, left, top = letterbox(frame_bgr, self.input_size)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        x = np.expand_dims(np.ascontiguousarray(x), 0)
        in_dtype = self.engine.bindings[self.input_name]["dtype"]
        if x.dtype != in_dtype:
            x = x.astype(in_dtype)
        return x, ratio, left, top

    def _scale_boxes(self, xyxy: np.ndarray, ratio: float, left: int, top: int, orig_shape) -> np.ndarray:
        out = xyxy.copy()
        out[:, [0, 2]] -= left
        out[:, [1, 3]] -= top
        out[:, :4] /= max(ratio, 1e-6)
        h, w = orig_shape[:2]
        out[:, [0, 2]] = out[:, [0, 2]].clip(0, w)
        out[:, [1, 3]] = out[:, [1, 3]].clip(0, h)
        return out

    def _parse_output(self, out: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        지원:
          - (1, N, 6) 또는 (N, 6): x1,y1,x2,y2,conf,cls  (YOLO export end-to-end)
          - (1, 4+nc, N) 또는 (1, N, 4+nc): 원시 출력 → conf/NMS
        """
        arr = out
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]

        # end-to-end: (N, 6)
        if arr.ndim == 2 and arr.shape[1] == 6:
            xyxy = arr[:, :4].astype(np.float32)
            scores = arr[:, 4].astype(np.float32)
            cls_ids = arr[:, 5].astype(np.int32)
            mask = scores >= self.conf_thres
            return xyxy[mask], scores[mask], cls_ids[mask]

        # (N, 4+nc)
        if arr.ndim == 2 and arr.shape[1] > 5:
            boxes = arr[:, :4]
            cls_scores = arr[:, 4:]
            cls_ids = cls_scores.argmax(1).astype(np.int32)
            scores = cls_scores.max(1).astype(np.float32)
            mask = scores >= self.conf_thres
            boxes, scores, cls_ids = boxes[mask], scores[mask], cls_ids[mask]
            xyxy = _xywh_to_xyxy(boxes)
            keep = nms_xyxy(xyxy, scores, self.iou_thres)
            return xyxy[keep], scores[keep], cls_ids[keep]

        # (4+nc, N)
        if arr.ndim == 2 and arr.shape[0] > 5:
            arr = arr.T
            return self._parse_output(arr)

        raise RuntimeError(f"알 수 없는 YOLO 출력 shape: {out.shape}")

    def predict(self, frame_bgr: np.ndarray) -> list[dict]:
        t0 = time.perf_counter()
        blob, ratio, left, top = self.preprocess(frame_bgr)
        t1 = time.perf_counter()
        outputs = self.engine.infer({self.input_name: blob})
        t2 = time.perf_counter()

        # 첫 출력 사용 (이름 로그는 engine 로드 시 출력됨)
        out_name = self.engine.output_names[0]
        raw = outputs[out_name]
        if not self._logged_output_shape:
            print(f"[YOLO] output '{out_name}' shape={raw.shape}")
            self._logged_output_shape = True

        xyxy, scores, cls_ids = self._parse_output(raw)
        xyxy = self._scale_boxes(xyxy, ratio, left, top, frame_bgr.shape)
        t3 = time.perf_counter()

        self.last_timing = {
            "preprocess_ms": (t1 - t0) * 1000,
            "infer_ms": self.engine.last_infer_ms,
            "postprocess_ms": (t3 - t2) * 1000,
            "total_ms": (t3 - t0) * 1000,
        }

        dets = []
        for i in range(len(xyxy)):
            cid = int(cls_ids[i])
            name = self.class_names[cid] if 0 <= cid < len(self.class_names) else str(cid)
            box = xyxy[i].astype(int).tolist()
            dets.append(
                {
                    "class_id": cid,
                    "class_name": name,
                    "confidence": float(scores[i]),
                    "xyxy": box,
                    "bbox": box,
                }
            )
        return dets

    def close(self) -> None:
        self.engine.close()
