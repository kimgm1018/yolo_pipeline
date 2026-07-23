"""YOLO 탐지·트래킹 계층 — ultralytics 또는 TensorRT+ByteTrack."""

from __future__ import annotations

from pathlib import Path

from yolo_trt import DEFAULT_CLASS_NAMES


def load_model(model_path):
    from ultralytics import YOLO

    return YOLO(str(model_path))


def track_frame(model, frame, conf, tracker):
    """Ultralytics: 한 프레임 탐지 + ByteTrack. Results 1개 반환."""
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

    for box in result.boxes:
        class_id = int(box.cls.item())

        track_id = None
        if box.id is not None:
            track_id = int(box.id.item())

        bbox = box.xyxy[0].cpu().numpy().astype(int).tolist()

        detections.append(
            {
                "track_id": track_id,
                "class_id": class_id,
                "class_name": result.names[class_id],
                "confidence": float(box.conf.item()),
                "bbox": bbox,
            }
        )

    return detections


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

    def track_frame(self, frame, conf=None, tracker=None):
        # conf/tracker args kept for call-site compatibility; thresholds set at init
        dets = self.yolo.predict(frame)
        return self.tracker.update(detections=dets, frame_shape=frame.shape)

    def close(self):
        self.yolo.close()
