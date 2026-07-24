"""
Jetson 라이브 파이프라인 (이 폴더 = 배포 루트)

보드 예:
  cd ~/yolo-pipeline/yolo_pipeline
  source venv/bin/activate
  python main.py --source 0 --no-ocr
  python main.py --source 0

MQTT 대신 logs/ 에 JSONL 기록.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2

import config
from detector import UltralyticsTrackedDetector
from event_manager import EventManager
from plate_collector import PlateCollector
from yolo_trt import DEFAULT_CLASS_NAMES


class EventLogger:
    """콘솔 + logs/*.jsonl (MQTT 대체)."""

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.urgent_path = self.log_dir / f"urgent_{stamp}.jsonl"
        self.batch_path = self.log_dir / f"batch_{stamp}.jsonl"
        self.session_path = self.log_dir / f"session_{stamp}.json"
        print(f"Log dir: {self.log_dir}")
        print(f"Urgent : {self.urgent_path.name}")
        print(f"Batch  : {self.batch_path.name}")

    def _append(self, path: Path, obj: dict) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def urgent(self, event: dict) -> None:
        print("[URGENT]", json.dumps(event, ensure_ascii=False))
        self._append(self.urgent_path, event)

    def batch(self, event: dict) -> None:
        print("[BATCH]", json.dumps(event, ensure_ascii=False))
        self._append(self.batch_path, event)

    def save_session(self, payload: dict) -> None:
        self.session_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Session summary → {self.session_path}")


def parse_source(value: str):
    if value.isdigit():
        return int(value)
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"영상 파일이 없습니다: {path}")
    return str(path)


def parse_args():
    p = argparse.ArgumentParser(description="Jetson TRT engine + Ultralytics BoT-SORT 파이프라인")
    p.add_argument("--source", default=str(config.VIDEO_SOURCE))
    p.add_argument("--yolo-engine", default=str(config.YOLO_ENGINE_PATH))
    p.add_argument("--ocr-engine", default=str(config.OCR_ENGINE_PATH))
    p.add_argument("--ocr-dict", default=str(config.OCR_DICT_PATH))
    p.add_argument("--conf", type=float, default=config.CONF)
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--no-ocr", action="store_true")
    return p.parse_args()


def draw_detections(frame, detections):
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        tid = d.get("track_id", -1)
        name = d["class_name"]
        conf = d["confidence"]
        label = f"{name} #{tid} {conf:.2f}"
        color = (0, 200, 255) if name == "licence" else (80, 220, 80)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out


def main():
    args = parse_args()
    source = parse_source(args.source)
    logger = EventLogger(config.LOG_DIR)
    event_manager = EventManager()
    event_queue: list[dict] = []

    yolo_engine = Path(args.yolo_engine)
    if not yolo_engine.exists():
        raise FileNotFoundError(
            f"YOLO engine 없음: {yolo_engine}\n"
            f"예상 위치: {config.ROOT}/models/yolo26_fp16.engine"
        )

    detector = UltralyticsTrackedDetector(
        engine_path=yolo_engine,
        tracker_config=config.TRACKER,
        input_size=config.IMGSZ,
        confidence_threshold=args.conf,
        iou_threshold=config.IOU,
        class_names=list(DEFAULT_CLASS_NAMES),
    )

    plate_collector = None
    if not args.no_ocr:
        ocr_engine = Path(args.ocr_engine)
        ocr_dict = Path(args.ocr_dict)
        if not ocr_engine.exists():
            raise FileNotFoundError(f"OCR engine 없음: {ocr_engine}")
        if not ocr_dict.exists():
            raise FileNotFoundError(f"OCR dict 없음: {ocr_dict}")
        plate_collector = PlateCollector(
            backend="tensorrt",
            engine_path=str(ocr_engine),
            dict_path=str(ocr_dict),
            min_conf=0.25,
        )

    camera = cv2.VideoCapture(source)
    if not camera.isOpened():
        raise RuntimeError(f"카메라/영상을 열 수 없습니다: {source}")

    print(f"ROOT   : {config.ROOT}")
    print(f"Backend: {config.BACKEND}  IMGSZ={config.IMGSZ}")
    print(f"YOLO   : {yolo_engine}")
    print(f"Tracker: {config.TRACKER}")
    print(f"OCR    : {'off' if args.no_ocr else args.ocr_engine}")
    print(f"Source : {source}")
    print(f"Robot  : {config.ROBOT_ID}")

    frame_idx = 0
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            detections = detector.track_frame(frame)
            view = draw_detections(frame, detections)

            if plate_collector is not None:
                plate_collector.update(detections, frame, frame_idx)

            batch_events, urgent_events = event_manager.create_events(
                detections=detections,
                x=config.CURRENT_X,
                y=config.CURRENT_Y,
                robot_id=config.ROBOT_ID,
            )

            for event in urgent_events:
                logger.urgent(event)

            event_queue.extend(batch_events)
            for event in batch_events:
                logger.batch(event)

            if not args.no_show:
                cv2.imshow("Patrol AI (Jetson)", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"frames={frame_idx} dets={len(detections)} queued={len(event_queue)}")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        detector.close()

    plates = []
    if plate_collector is not None:
        plates = plate_collector.all_plates()
        print("OCR 수집 번호판:")
        print(json.dumps(plates, ensure_ascii=False, indent=2))
        unmatched = event_manager.create_unmatched_plate_events(
            plates=plates,
            x=config.CURRENT_X,
            y=config.CURRENT_Y,
        )
        event_queue.extend(unmatched)
        for event in unmatched:
            logger.batch(event)

    batch_request = {
        "robotId": config.ROBOT_ID,
        "frames": frame_idx,
        "plates": plates,
        "events": event_queue,
    }
    logger.save_session(batch_request)
    print("최종 일괄 이벤트 수:", len(event_queue))


if __name__ == "__main__":
    main()
