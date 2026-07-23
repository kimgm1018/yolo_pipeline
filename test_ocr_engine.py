"""OCR TensorRT 단독 테스트."""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from plate_trt import PlateTensorRTRecognizer


def imread(path: str):
    img = cv2.imread(path)
    if img is not None:
        return img
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default=None)
    p.add_argument("--dict", default=None)
    p.add_argument("--image", required=True)
    args = p.parse_args()

    import config

    engine = args.engine or str(config.OCR_ENGINE_PATH)
    dict_path = args.dict or str(config.OCR_DICT_PATH)

    img = imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    rec = PlateTensorRTRecognizer(engine, dict_path)
    try:
        result = rec.predict(img)
        print("timing:", rec.last_timing)
        print("text:", result["text"])
        print("confidence:", result["confidence"])
    finally:
        rec.close()


if __name__ == "__main__":
    main()
