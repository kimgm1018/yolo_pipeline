"""Jetson 배포용 고정 shape·경로. PIPELINE_ROOT = 상위 pipeline 폴더."""

from pathlib import Path

JETSON_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = JETSON_DIR.parent

# YOLO — 학습 imgsz=416
YOLO_IMGSZ = 416
YOLO_INPUT_SHAPE = (1, 3, YOLO_IMGSZ, YOLO_IMGSZ)  # NCHW
YOLO_PT = PIPELINE_ROOT / "models" / "best.pt"
YOLO_ONNX = JETSON_DIR / "models" / "best.onnx"
YOLO_ENGINE = JETSON_DIR / "models" / "yolo26_fp16.engine"

# OCR — ocr_finetune_rec/inference.yml RecResizeImg
OCR_H = 48
OCR_W = 320
OCR_INPUT_SHAPE = (1, 3, OCR_H, OCR_W)  # NCHW
OCR_PADDLE_DIR = PIPELINE_ROOT / "ocr_finetune_rec"
OCR_ONNX = JETSON_DIR / "models" / "ocr_rec.onnx"
OCR_ENGINE = JETSON_DIR / "models" / "ocr_rec_fp16.engine"

# ONNX / TRT
ONNX_OPSET = 17
TRT_FP16 = True
TRT_WORKSPACE_MIB = 4096
