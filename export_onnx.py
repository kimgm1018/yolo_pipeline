"""YOLO / OCR ONNX export (PC). OCR은 paddle2onnx 안내 + 스텁."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_config import (  # noqa: E402
    JETSON_DIR,
    OCR_ONNX,
    OCR_PADDLE_DIR,
    ONNX_OPSET,
    YOLO_IMGSZ,
    YOLO_ONNX,
    YOLO_PT,
)


def export_yolo() -> Path:
    if not YOLO_PT.exists():
        raise FileNotFoundError(f"YOLO 가중치 없음: {YOLO_PT}")

    from ultralytics import YOLO

    YOLO_ONNX.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(YOLO_PT))
    # ultralytics는 보통 가중치 옆에 onnx를 씀 → jetson_pipeline/models 로 이동
    out = model.export(
        format="onnx",
        imgsz=YOLO_IMGSZ,
        simplify=True,
        dynamic=False,
        opset=ONNX_OPSET,
    )
    src = Path(out)
    if src.resolve() != YOLO_ONNX.resolve():
        shutil.copy2(src, YOLO_ONNX)
    print(f"YOLO ONNX → {YOLO_ONNX}")
    print(f"shape: 1x3x{YOLO_IMGSZ}x{YOLO_IMGSZ}, opset={ONNX_OPSET}, dynamic=False")
    return YOLO_ONNX


def export_ocr_help() -> None:
    """
    PaddleOCR 3.x fine-tune 모델 → ONNX는 환경에 따라 paddle2onnx / paddlex 사용.
    여기서는 명령 가이드를 출력하고, 이미 onnx가 있으면 경로만 확인.
    """
    guide = JETSON_DIR / "docs" / "ocr_onnx_export.md"
    guide.parent.mkdir(parents=True, exist_ok=True)
    guide.write_text(
        f"""# OCR ONNX export 가이드

Paddle 모델: `{OCR_PADDLE_DIR}`
목표: `{OCR_ONNX}` (입력 `1x3x48x320`)

## paddle2onnx (가능한 경우)

```bash
paddle2onnx \\
  --model_dir {OCR_PADDLE_DIR} \\
  --model_filename inference.json \\
  --params_filename inference.pdiparams \\
  --save_file {OCR_ONNX} \\
  --opset_version {ONNX_OPSET} \\
  --enable_onnx_checker True
```

Paddle 3 PIR(`inference.json`)이면 도구 버전에 따라 실패할 수 있다.
실패 시 PaddleX export 문서를 따르거나, 우선 YOLO ONNX만 진행한다.

전처리는 `ocr_preprocess.preprocess_ocr_fixed`와 동일해야 한다.
보드 쪽 인식기는 루트 `plate_trt.py`를 사용한다.
""",
        encoding="utf-8",
    )
    print(f"OCR export 가이드 → {guide}")
    if OCR_ONNX.exists():
        print(f"이미 존재: {OCR_ONNX}")
    else:
        print(f"아직 없음: {OCR_ONNX} (위 가이드로 생성)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--yolo", action="store_true")
    p.add_argument("--ocr", action="store_true", help="OCR ONNX 가이드 작성")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    if not (args.yolo or args.ocr or args.all):
        args.all = True

    if args.yolo or args.all:
        export_yolo()
    if args.ocr or args.all:
        export_ocr_help()


if __name__ == "__main__":
    main()
