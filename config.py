"""Jetson 배포 루트 설정.

보드 경로 예:
  /home/e103/yolo-pipeline/yolo_pipeline
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ultralytics: TRT .engine + Ultralytics track (BoT-SORT)
# tensorrt: yolo_trt + bytetrack_adapter (레거시 유지)
BACKEND = "ultralytics"

MODEL_PATH = ROOT / "models" / "best.pt"

YOLO_ENGINE_PATH = ROOT / "models" / "yolo26_fp16.engine"
OCR_ENGINE_PATH = ROOT / "models" / "plate_rec_fp16.engine"
OCR_DICT_PATH = ROOT / "models" / "plate_dict.txt"

OCR_MODEL_DIR = ROOT / "ocr_finetune_rec"
OCR_DEVICE = "cpu"
IMGSZ = 416

VIDEO_SOURCE = 0

CONF = 0.20
IOU = 0.50
TRACKER = str(ROOT / "trackers" / "botsort.yaml")
# 자체 ByteTrack 경로용 (TensorRTTrackedDetector)
BYTETRACK_CONFIG = str(ROOT / "trackers" / "bytetrack_stable.yaml")

ROBOT_ID = 1
CURRENT_X = 10.0
CURRENT_Y = 20.0

LOG_DIR = ROOT / "logs"

URGENT_CLASSES = {"lying"}

ALLOWED_PLATES = {
    # "12가3456",
}

DEFER_EVENT_CLASSES = {"licence"}

EVENT_INFO = {
    "small_trash": {
        "eventType": "SMALL_TRASH",
        "eventTitle": "작은 쓰레기 감지",
        "eventDetails": "통행로에서 작은 쓰레기가 감지되었습니다.",
        "riskLevel": "LOW",
    },
    "lying": {
        "eventType": "EMERGENCY_PERSON",
        "eventTitle": "응급상황 의심",
        "eventDetails": "바닥에 누워 있는 사람이 감지되었습니다.",
        "riskLevel": "HIGH",
    },
    "bin_overflow": {
        "eventType": "BIN_OVERFLOW",
        "eventTitle": "쓰레기통 포화 감지",
        "eventDetails": "쓰레기통이 가득 찬 상태입니다.",
        "riskLevel": "MEDIUM",
    },
    "bag_outside": {
        "eventType": "BAG_OUTSIDE",
        "eventTitle": "쓰레기 외부 적치 감지",
        "eventDetails": "쓰레기통 주변에 쓰레기가 적치되어 있습니다.",
        "riskLevel": "MEDIUM",
    },
}

UNMATCHED_PLATE_EVENT = {
    "eventType": "UNMATCHED_PLATE",
    "eventTitle": "미등록 번호판 감지",
    "eventDetails": "허용 목록에 없는 번호판이 감지되었습니다.",
    "riskLevel": "MEDIUM",
}
