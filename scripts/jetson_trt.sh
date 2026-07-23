#!/usr/bin/env bash
# Jetson Orin Nano — ONNX → TensorRT FP16
# Usage (on device):
#   cd jetson_pipeline
#   bash scripts/jetson_trt.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$ROOT/models"
WS="${TRT_WORKSPACE:-4096}"

mkdir -p "$MODELS"

build_one () {
  local onnx="$1"
  local engine="$2"
  if [[ ! -f "$onnx" ]]; then
    echo "SKIP (missing): $onnx"
    return 0
  fi
  echo "Building $engine from $onnx"
  /usr/src/tensorrt/bin/trtexec \
    --onnx="$onnx" \
    --saveEngine="$engine" \
    --fp16 \
    --memPoolSize=workspace:${WS}MiB \
    --verbose
  echo "OK → $engine"
}

build_one "$MODELS/best.onnx" "$MODELS/best_fp16.engine"
build_one "$MODELS/ocr_rec.onnx" "$MODELS/ocr_rec_fp16.engine"

echo "Done. Copy engines stay under $MODELS"
