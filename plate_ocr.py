"""Korean licence-plate OCR via PaddleOCR TextRecognition only (no full OCR pipeline).

Strategy:
  - reject edge-cutoff / tiny / bad-aspect boxes
  - buffer 5~10 valid crops per track id
  - OCR only the top-K by quality score
  - majority-vote the readings
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEFAULT_MODEL = "korean_PP-OCRv5_mobile_rec"
ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT / "ocr_finetune_rec"

_HANGUL_DIGIT = re.compile(r"[^0-9가-힣]")

# Soft crop gates — prefer OCR over dropping candidates
MIN_PLATE_WIDTH = 40
MIN_PLATE_HEIGHT = 12
MIN_ASPECT = 1.5  # reject near-square crops (e.g. 145x151 ≈ 0.96)
MAX_ASPECT = 10.0
EDGE_MARGIN_RATIO = 0.0  # only reject boxes that actually touch the image border
CANDIDATE_BUFFER = 10
TOP_K_OCR = 3
MIN_CANDIDATES = 1  # OCR even with a single valid crop (best available)
QUALITY_GAIN = 1.10
CONFIRMATION_QUALITY_FLOOR = 0.75
HIGH_QUALITY_WIDTH = 80
HIGH_QUALITY_HEIGHT = 20
HIGH_QUALITY_SHARPNESS = 30.0
DEFAULT_PAD_X = 0.10
DEFAULT_PAD_Y = 0.10


def _patch_paddlex_opencv_dep() -> None:
    """PaddleX checks distribution 'opencv-contrib-python'; cv2 from opencv-python is enough."""
    import importlib
    import sys

    import paddlex.utils.deps as px_deps

    if getattr(px_deps, "_opencv_patched", False):
        return

    _orig = px_deps.is_dep_available

    def _is_dep_available(dep, /, check_version=False):
        if dep == "opencv-contrib-python":
            try:
                import cv2  # noqa: F401

                return True
            except ImportError:
                return False
        return _orig(dep, check_version=check_version)

    px_deps.is_dep_available = _is_dep_available
    px_deps._opencv_patched = True

    for name, mod in list(sys.modules.items()):
        if name.startswith("paddlex.") and mod is not None and not hasattr(mod, "cv2"):
            try:
                import cv2 as _cv2

                mod.cv2 = _cv2
            except ImportError:
                pass
    for mod_name in (
        "paddlex.inference.common.reader.image_reader",
        "paddlex.inference.utils.io.readers",
    ):
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])


def normalize_plate(text: str) -> str:
    if not text:
        return ""
    return _HANGUL_DIGIT.sub("", text.strip())


def is_plausible_plate(text: str) -> bool:
    """Reject obvious partial/garbage reads without overfitting one plate format."""
    if not (5 <= len(text) <= 12):
        return False
    if not re.search(r"[가-힣]", text):
        return False
    if sum(ch.isdigit() for ch in text) < 4:
        return False
    return True


def expand_box(xyxy, w: int, h: int, pad_x: float = DEFAULT_PAD_X, pad_y: float = DEFAULT_PAD_Y):
    x1, y1, x2, y2 = map(float, xyxy)
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * pad_x))
    y1 = max(0, int(y1 - bh * pad_y))
    x2 = min(w, int(x2 + bw * pad_x))
    y2 = min(h, int(y2 + bh * pad_y))
    return x1, y1, x2, y2


def is_box_cutoff(box, image_width: int, image_height: int, margin_ratio: float = EDGE_MARGIN_RATIO) -> bool:
    x1, y1, x2, y2 = map(float, box)
    margin_x = image_width * margin_ratio
    margin_y = image_height * margin_ratio
    return (
        x1 <= margin_x
        or y1 <= margin_y
        or x2 >= image_width - margin_x
        or y2 >= image_height - margin_y
    )


def valid_plate_crop(
    box,
    frame_width: int,
    frame_height: int,
    min_width: int = MIN_PLATE_WIDTH,
    min_height: int = MIN_PLATE_HEIGHT,
    edge_margin_ratio: float = EDGE_MARGIN_RATIO,
    min_aspect: float = MIN_ASPECT,
    max_aspect: float = MAX_ASPECT,
) -> bool:
    x1, y1, x2, y2 = map(int, box)
    width = x2 - x1
    height = y2 - y1
    if width < min_width or height < min_height:
        return False
    aspect = width / max(height, 1)
    if aspect < min_aspect or aspect > max_aspect:
        return False
    if is_box_cutoff((x1, y1, x2, y2), frame_width, frame_height, edge_margin_ratio):
        return False
    return True


def crop_sharpness(crop: np.ndarray) -> float:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calculate_crop_score(crop: np.ndarray) -> float:
    if crop is None or crop.size == 0:
        return -1.0
    height, width = crop.shape[:2]
    sharpness = crop_sharpness(crop)
    area_score = float(width * height)
    aspect = width / max(height, 1)
    ratio_score = 1.0 if 1.5 <= aspect <= 8.0 else 0.5
    return area_score * ratio_score + sharpness * 10.0


def is_high_quality_crop(crop: np.ndarray) -> bool:
    height, width = crop.shape[:2]
    aspect = width / max(height, 1)
    return (
        width >= HIGH_QUALITY_WIDTH
        and height >= HIGH_QUALITY_HEIGHT
        and 1.5 <= aspect <= 8.0
        and crop_sharpness(crop) >= HIGH_QUALITY_SHARPNESS
    )


def _variants(crop: np.ndarray) -> list[np.ndarray]:
    h, w = crop.shape[:2]
    up = cv2.resize(crop, (max(w * 2, 160), max(h * 2, 48)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return [crop, up, gray_bgr]


def _extract_rec(result: Any) -> tuple[str, float]:
    data = None
    if isinstance(result, dict):
        data = result.get("res", result)
    else:
        for attr in ("res", "json", "data"):
            if hasattr(result, attr):
                val = getattr(result, attr)
                if callable(val):
                    try:
                        val = val()
                    except TypeError:
                        pass
                if isinstance(val, dict):
                    data = val.get("res", val)
                    break
        if data is None and hasattr(result, "__dict__"):
            data = getattr(result, "__dict__", {})

    if not isinstance(data, dict):
        data = {}

    text = data.get("rec_text") or data.get("text") or ""
    score = data.get("rec_score")
    if score is None:
        score = data.get("score", 0.0)
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.0
    return str(text), score


@dataclass
class CropCandidate:
    frame_idx: int
    score: float
    crop: np.ndarray
    box: tuple[int, int, int, int]


class PlateOCR:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        model_dir: str | Path | None = DEFAULT_MODEL_DIR,
        device: str | None = None,
        min_conf: float = 0.3,
        vote_window: int = 3,
        candidate_buffer: int = CANDIDATE_BUFFER,
        top_k_ocr: int = TOP_K_OCR,
        min_candidates_before_ocr: int = MIN_CANDIDATES,
        quality_gain: float = QUALITY_GAIN,
        backend: str = "paddle",
        engine_path: str | Path | None = None,
        dict_path: str | Path | None = None,
    ):
        self.backend = backend
        self.min_conf = min_conf
        self.vote_window = max(2, vote_window)
        self.candidate_buffer = max(3, candidate_buffer)
        self.top_k_ocr = max(1, top_k_ocr)
        self.min_candidates_before_ocr = max(1, min_candidates_before_ocr)
        self.quality_gain = max(1.0, quality_gain)

        self._candidates: dict[int, list[CropCandidate]] = defaultdict(list)
        self._ocr_done_frames: dict[int, set[int]] = defaultdict(set)
        self._readings: dict[int, list[tuple[str, float]]] = defaultdict(list)
        self._ocr_attempts: dict[int, int] = defaultdict(int)
        self._best_ocr_quality: dict[int, float] = defaultdict(float)
        self._final: dict[int, tuple[str, float]] = {}
        self.model = None
        self.trt_recognizer = None

        if backend == "tensorrt":
            from plate_trt import PlateTensorRTRecognizer

            if engine_path is None or dict_path is None:
                raise FileNotFoundError("tensorrt OCR에는 engine_path, dict_path가 필요합니다.")
            self.trt_recognizer = PlateTensorRTRecognizer(engine_path, dict_path)
            print(f"OCR ready: TensorRT ({engine_path})")
            print(f"OCR dict: {dict_path}")
        else:
            # Import torch first so CUDA DLLs are resolved before PaddleX/modelscope.
            import torch  # noqa: F401

            _patch_paddlex_opencv_dep()
            from paddleocr import TextRecognition

            resolved_dir = Path(model_dir) if model_dir else None
            if resolved_dir is not None and not (
                (resolved_dir / "inference.pdiparams").exists()
                and (
                    (resolved_dir / "inference.json").exists()
                    or (resolved_dir / "inference.pdmodel").exists()
                )
            ):
                print(f"[warn] OCR model_dir incomplete: {resolved_dir} — falling back to official {model_name}")
                resolved_dir = None

            kwargs: dict[str, Any] = {"model_name": model_name}
            if resolved_dir is not None:
                kwargs["model_dir"] = str(resolved_dir)
            if device:
                kwargs["device"] = device
            self.model = TextRecognition(**kwargs)
            src = str(resolved_dir) if resolved_dir is not None else f"official:{model_name}"
            print(f"OCR ready: Paddle TextRecognition ({model_name})")
            print(f"OCR weights: {src}")

        print(
            "OCR policy: soft filter → pad 10% → OCR from 1+ candidates "
            f"(max {self.top_k_ocr} attempts / track)"
        )

    def predict_crop(self, crop: np.ndarray) -> tuple[str, float]:
        if crop is None or crop.size == 0:
            return "", 0.0

        if self.backend == "tensorrt":
            try:
                result = self.trt_recognizer.predict(crop)
            except Exception as e:
                print(f"[OCR] TensorRT predict 실패: {e}")
                return "", 0.0
            text = normalize_plate(result.get("text", ""))
            score = float(result.get("confidence", 0.0))
            if text and is_plausible_plate(text) and score >= self.min_conf:
                return text, score
            return "", 0.0

        best_text, best_score = "", 0.0
        for variant in _variants(crop):
            outputs = self.model.predict(input=variant, batch_size=1)
            for res in outputs:
                text, score = _extract_rec(res)
                text = normalize_plate(text)
                if (
                    text
                    and is_plausible_plate(text)
                    and score >= self.min_conf
                    and score >= best_score
                ):
                    best_text, best_score = text, score
        return best_text, best_score

    def consider_detection(
        self,
        tid: int,
        frame: np.ndarray,
        xyxy,
        frame_idx: int,
        pad_x: float = DEFAULT_PAD_X,
        pad_y: float = DEFAULT_PAD_Y,
    ) -> tuple[bool, str, float, np.ndarray | None]:
        """
        Collect a valid crop for track id. When enough candidates exist,
        OCR top-K not-yet-processed crops and update majority vote.

        Returns: (did_ocr, plate_text, conf, saved_crop_or_None)
        """
        h, w = frame.shape[:2]
        # validate on raw detector box (before pad) for edge/cutoff
        raw = tuple(map(int, map(float, xyxy)))
        if not valid_plate_crop(raw, w, h):
            return False, *self.final_for(tid), None

        x1, y1, x2, y2 = expand_box(xyxy, w, h, pad_x=pad_x, pad_y=pad_y)
        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return False, *self.final_for(tid), None

        score = calculate_crop_score(crop)
        if score < 0:
            return False, *self.final_for(tid), None

        cand = CropCandidate(frame_idx=frame_idx, score=score, crop=crop.copy(), box=(x1, y1, x2, y2))
        buf = self._candidates[tid]
        buf.append(cand)
        buf.sort(key=lambda c: c.score, reverse=True)
        del buf[self.candidate_buffer :]

        # A clearly good first crop should not be lost just because the track is short.
        ready = len(buf) >= self.min_candidates_before_ocr or is_high_quality_crop(crop)
        if not ready or self._ocr_attempts[tid] >= self.top_k_ocr:
            return False, *self.final_for(tid), None

        # OCR one candidate per frame. Later attempts require a meaningful quality gain.
        candidate = next(
            (c for c in buf if c.frame_idx not in self._ocr_done_frames[tid]),
            None,
        )
        if candidate is None:
            return False, *self.final_for(tid), None

        best_done = self._best_ocr_quality[tid]
        reading_count = len(self._readings.get(tid) or [])
        if reading_count and best_done > 0:
            # Allow similarly good top crops to confirm the first reading.
            # Once enough readings exist, only a materially better crop may retry.
            required_ratio = (
                CONFIRMATION_QUALITY_FLOOR
                if reading_count < self.vote_window
                else self.quality_gain
            )
            if candidate.score < best_done * required_ratio:
                return False, *self.final_for(tid), None

        text, conf = self.predict_crop(candidate.crop)
        self._ocr_done_frames[tid].add(candidate.frame_idx)
        self._ocr_attempts[tid] += 1
        self._best_ocr_quality[tid] = max(best_done, candidate.score)
        if text:
            self._readings[tid].append((text, conf))
            if len(self._readings[tid]) > self.vote_window:
                self._readings[tid] = self._readings[tid][-self.vote_window :]
            self._final[tid] = self._majority(tid)

        return True, *self.final_for(tid), candidate.crop

    def _majority(self, tid: int) -> tuple[str, float]:
        readings = self._readings.get(tid) or []
        if not readings:
            return "", 0.0
        counts = Counter(text for text, _ in readings)
        confidence_sums: dict[str, float] = defaultdict(float)
        for text, conf in readings:
            confidence_sums[text] += conf
        plate = max(counts, key=lambda text: (counts[text], confidence_sums[text]))
        plate_conf = max(conf for text, conf in readings if text == plate)
        return plate, float(plate_conf)

    def final_for(self, tid: int) -> tuple[str, float]:
        if tid in self._final:
            return self._final[tid]
        return self._majority(tid)

    def all_finals(self) -> dict[int, tuple[str, float]]:
        tids = set(self._final) | set(self._readings) | set(self._candidates)
        return {tid: self.final_for(tid) for tid in tids}


# Apply OpenCV dep patch lazily inside PlateOCR.__init__ (after torch is imported)
