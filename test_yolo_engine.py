"""YOLO TensorRT 단독 테스트."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from yolo_trt import YoloTensorRTDetector


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default=None)
    p.add_argument("--image", required=True)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--out", default="outputs/yolo_trt_test.jpg")
    args = p.parse_args()

    import config

    engine = args.engine or str(config.YOLO_ENGINE_PATH)
    imgsz = args.imgsz or config.IMGSZ

    img = cv2.imread(args.image)
    if img is None:
        # unicode path fallback
        import numpy as np

        data = np.fromfile(args.image, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.image)

    det = YoloTensorRTDetector(
        engine, input_size=imgsz, confidence_threshold=args.conf
    )
    try:
        results = det.predict(img)
        print("timing:", det.last_timing)
        print("detections:", results)
        for d in results:
            x1, y1, x2, y2 = map(int, d["xyxy"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{d['class_name']} {d['confidence']:.2f}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), img)
        print("saved:", out)
    finally:
        det.close()


if __name__ == "__main__":
    main()
