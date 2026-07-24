"""YOLO TRT engine + Ultralytics BoT-SORT + OCR TRT 통합 (단일 이미지)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import config
from detector import UltralyticsTrackedDetector
from plate_trt import PlateTensorRTRecognizer
from yolo_trt import DEFAULT_CLASS_NAMES


def imread(path: str):
    img = cv2.imread(path)
    if img is not None:
        return img
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--yolo-engine", default=str(config.YOLO_ENGINE_PATH))
    p.add_argument("--ocr-engine", default=str(config.OCR_ENGINE_PATH))
    p.add_argument("--ocr-dict", default=str(config.OCR_DICT_PATH))
    p.add_argument("--out-dir", default="outputs/pipeline_trt_test")
    args = p.parse_args()

    frame = imread(args.image)
    if frame is None:
        raise FileNotFoundError(args.image)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    detector = UltralyticsTrackedDetector(
        engine_path=args.yolo_engine,
        tracker_config=config.TRACKER,
        input_size=config.IMGSZ,
        confidence_threshold=config.CONF,
        iou_threshold=config.IOU,
        class_names=list(DEFAULT_CLASS_NAMES),
    )
    ocr = PlateTensorRTRecognizer(args.ocr_engine, args.ocr_dict)
    try:
        dets = detector.track_frame(frame)
        print("detections:", dets)
        for i, d in enumerate(dets):
            if d["class_name"] != "licence":
                continue
            x1, y1, x2, y2 = map(int, d["bbox"])
            crop = frame[y1:y2, x1:x2]
            crop_path = out_dir / f"licence_{i}_id{d['track_id']}.jpg"
            cv2.imwrite(str(crop_path), crop)
            result = ocr.predict(crop)
            print(f"licence track={d['track_id']}: {result} crop={crop_path}")
        print(f"total_ms={(time.perf_counter()-t0)*1000:.1f}")
    finally:
        detector.close()
        ocr.close()


if __name__ == "__main__":
    main()
