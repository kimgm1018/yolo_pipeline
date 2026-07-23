"""번호판 OCR 수집 계층 — 탐지된 licence를 모아 최종 문자열로 확정."""

from plate_ocr import PlateOCR


class PlateCollector:
    """프레임마다 licence 박스를 넣고, track_id별 OCR 결과를 유지한다."""

    def __init__(
        self,
        model_dir=None,
        device="cpu",
        min_conf=0.25,
        backend="paddle",
        engine_path=None,
        dict_path=None,
    ):
        self.ocr = PlateOCR(
            model_dir=model_dir,
            device=device,
            min_conf=min_conf,
            backend=backend,
            engine_path=engine_path,
            dict_path=dict_path,
        )
        self.plates = {}

    def update(self, detections, frame, frame_idx):
        for det in detections:
            if det["class_name"] != "licence":
                continue

            tid = det["track_id"]
            if tid is None or tid < 0:
                continue

            _did, text, conf, _crop = self.ocr.consider_detection(
                tid, frame, det["bbox"], frame_idx
            )
            if text:
                self.plates[tid] = {"plate": text, "confidence": conf}

    def all_plates(self):
        return [
            {"track_id": tid, "plate": info["plate"], "confidence": info["confidence"]}
            for tid, info in self.plates.items()
        ]
