"""YOLO 탐지·트래킹 계층.

- UltralyticsTrackedDetector: TensorRT .engine + Ultralytics BoT-SORT (기본)
- TensorRTTrackedDetector: yolo_trt + 자체 ByteTrackAdapter (유지)
"""

from __future__ import annotations

from pathlib import Path

from yolo_trt import DEFAULT_CLASS_NAMES


def load_model(model_path):
    from ultralytics import YOLO

    return YOLO(str(model_path))


def track_frame(model, frame, conf, tracker):
    """Ultralytics: 한 프레임 탐지 + 트래커. Results 1개 반환."""
    return model.track(
        frame,
        persist=True,
        conf=conf,
        tracker=tracker,
        verbose=False,
    )[0]


def get_detections(result):
    """Ultralytics Results → 공통 dict 목록."""
    detections = []

    if result.boxes is None:
        return detections

    names = result.names or {}
    for box in result.boxes:
        class_id = int(box.cls.item())

        track_id = None
        if box.id is not None:
            track_id = int(box.id.item())

        bbox = box.xyxy[0].cpu().numpy().astype(int).tolist()
        class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]

        detections.append(
            {
                "track_id": track_id,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(box.conf.item()),
                "bbox": bbox,
                "xyxy": bbox,
            }
        )

    return detections


def _apply_class_names(model, class_names: list[str]) -> None:
    """엔진에 이름이 없거나 기본값일 때 학습 클래스 순서로 덮어쓴다."""
    mapping = {i: name for i, name in enumerate(class_names)}
    try:
        model.model.names = mapping
    except Exception:
        pass
    try:
        model.names = mapping
    except Exception:
        pass


class UltralyticsTrackedDetector:
    """TensorRT .engine을 Ultralytics로 로드 + BoT-SORT(또는 yaml) track."""

    def __init__(
        self,
        engine_path: str | Path,
        tracker_config: str | Path,
        input_size: int = 416,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        class_names: list[str] | None = None,
    ):
        from ultralytics import YOLO

        self.engine_path = Path(engine_path)
        self.tracker_config = str(tracker_config)
        self.input_size = int(input_size)
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.class_names = list(class_names or DEFAULT_CLASS_NAMES)

        self.model = YOLO(str(self.engine_path), task="detect")
        _apply_class_names(self.model, self.class_names)
        self.last_timing: dict[str, float] = {}

    def track_frame(self, frame, conf=None, tracker=None):
        import time

        t0 = time.perf_counter()
        result = self.model.track(
            source=frame,
            persist=True,
            conf=self.confidence_threshold if conf is None else conf,
            iou=self.iou_threshold,
            imgsz=self.input_size,
            tracker=tracker or self.tracker_config,
            verbose=False,
        )[0]
        self.last_timing = {"infer_ms": (time.perf_counter() - t0) * 1000.0}
        return get_detections(result)

    def close(self):
        self.model = None


class TensorRTTrackedDetector:
    """YOLO TensorRT + 독립 ByteTrack. track_frame(frame) → detections with track_id."""

    def __init__(
        self,
        engine_path: str | Path,
        tracker_config: str | Path,
        input_size: int = 640,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        class_names: list[str] | None = None,
    ):
        from bytetrack_adapter import ByteTrackAdapter
        from yolo_trt import YoloTensorRTDetector

        self.yolo = YoloTensorRTDetector(
            engine_path=engine_path,
            input_size=input_size,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            class_names=class_names or list(DEFAULT_CLASS_NAMES),
        )
        self.tracker = ByteTrackAdapter(tracker_config)

    @property
    def last_timing(self) -> dict:
        return self.yolo.last_timing

    def track_frame(self, frame, conf=None, tracker=None):
        # conf/tracker args kept for call-site compatibility; thresholds set at init
        dets = self.yolo.predict(frame)
        return self.tracker.update(detections=dets, frame_shape=frame.shape)

    def close(self):
        self.yolo.close()
