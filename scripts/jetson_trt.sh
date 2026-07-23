#!/usr/bin/env bash
# Jetson Orin Nano — ONNX → TensorRT FP16
# Usage (on device, this folder = yolo_pipeline / jetson_pipeline):
#   bash scripts/jetson_trt.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$ROOT/models"
WS="${TRT_WORKSPACE:-4096}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

mkdir -p "$MODELS"

if [[ ! -x "$TRTEXEC" ]]; then
  echo "trtexec 없음: $TRTEXEC"
  exit 1
fi

echo "ROOT=$ROOT"
echo "trtexec=$TRTEXEC"

# YOLO: fixed 1x3x640x640
if [[ -f "$MODELS/best.onnx" ]]; then
  echo "=== YOLO FP16 ==="
  "$TRTEXEC" \
    --onnx="$MODELS/best.onnx" \
    --saveEngine="$MODELS/yolo26_fp16.engine" \
    --fp16 \
    --memPoolSize=workspace:${WS}MiB
  echo "OK → $MODELS/yolo26_fp16.engine"
else
  echo "SKIP (missing): $MODELS/best.onnx"
fi

# OCR: dynamic W → fix to RecResize 48x320 (input name: x)
OCR_ONNX=""
if [[ -f "$MODELS/plate_rec.onnx" ]]; then
  OCR_ONNX="$MODELS/plate_rec.onnx"
elif [[ -f "$MODELS/ocr_rec.onnx" ]]; then
  OCR_ONNX="$MODELS/ocr_rec.onnx"
fi

if [[ -n "$OCR_ONNX" ]]; then
  echo "=== OCR FP16 ($OCR_ONNX) ==="
  "$TRTEXEC" \
    --onnx="$OCR_ONNX" \
    --saveEngine="$MODELS/plate_rec_fp16.engine" \
    --fp16 \
    --memPoolSize=workspace:${WS}MiB \
    --minShapes=x:1x3x48x160 \
    --optShapes=x:1x3x48x320 \
    --maxShapes=x:1x3x48x320
  echo "OK → $MODELS/plate_rec_fp16.engine"
else
  echo "SKIP (missing): plate_rec.onnx / ocr_rec.onnx"
fi

echo "=== done ==="
ls -lh "$MODELS"/*.engine 2>/dev/null || true
