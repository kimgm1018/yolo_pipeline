#!/usr/bin/env bash
# Jetson Orin Nano — ONNX → TensorRT FP16
#
# Usage:
#   bash scripts/jetson_trt.sh
#   bash scripts/jetson_trt.sh --yolo-only
#   bash scripts/jetson_trt.sh --ocr-only
#
# Orin Nano에서 YOLO(어텐션) 빌드 시 workspace 부족이 흔함.
# 환경변수로 조절:
#   TRT_WORKSPACE=8192|12288     (MiB, 기본 8192)
#   TRT_BUILDER_OPT=0|1|2|3     (기본 1, 낮을수록 빌드 메모리↓)
#   TRTEXEC=/usr/src/tensorrt/bin/trtexec

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$ROOT/models"
WS="${TRT_WORKSPACE:-8192}"
OPT="${TRT_BUILDER_OPT:-1}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

DO_YOLO=1
DO_OCR=1
for arg in "$@"; do
  case "$arg" in
    --yolo-only) DO_OCR=0 ;;
    --ocr-only)  DO_YOLO=0 ;;
    -h|--help)
      sed -n '2,14p' "$0"
      exit 0
      ;;
  esac
done

mkdir -p "$MODELS"

if [[ ! -x "$TRTEXEC" ]]; then
  echo "trtexec 없음: $TRTEXEC"
  exit 1
fi

echo "ROOT=$ROOT"
echo "trtexec=$TRTEXEC"
echo "workspace=${WS}MiB  builderOptimizationLevel=$OPT"
echo "--- memory ---"
free -h || true
echo "--------------"

build_engine () {
  # $1=onnx $2=engine $3=extra args...
  local onnx="$1"
  local engine="$2"
  shift 2
  local extra=("$@")

  local try_ws="$WS"
  local try_opt="$OPT"
  local attempt=1
  local max_attempts=3

  while (( attempt <= max_attempts )); do
    echo "=== build attempt $attempt/$max_attempts ==="
    echo "  onnx=$onnx"
    echo "  engine=$engine"
    echo "  workspace=${try_ws}MiB  opt=$try_opt"

    set +e
    "$TRTEXEC" \
      --onnx="$onnx" \
      --saveEngine="$engine" \
      --fp16 \
      --memPoolSize=workspace:${try_ws}MiB \
      --builderOptimizationLevel="$try_opt" \
      "${extra[@]}"
    local rc=$?
    set -e

    if [[ $rc -eq 0 && -f "$engine" ]]; then
      echo "OK → $engine ($(du -h "$engine" | awk '{print $1}'))"
      return 0
    fi

    echo "FAIL (exit=$rc). 재시도: workspace↑ / optimizationLevel↓"
    # 12288 → 16384, opt 1→0
    if (( try_ws < 12288 )); then
      try_ws=12288
    elif (( try_ws < 16384 )); then
      try_ws=16384
    fi
    if (( try_opt > 0 )); then
      try_opt=$((try_opt - 1))
    fi
    attempt=$((attempt + 1))
  done

  echo "ERROR: engine 빌드 실패: $engine"
  echo "힌트: 다른 GPU 프로세스 종료, swap 확보 후"
  echo "  TRT_WORKSPACE=16384 TRT_BUILDER_OPT=0 bash scripts/jetson_trt.sh --yolo-only"
  return 1
}

# YOLO: fixed 1x3x640x640
if [[ "$DO_YOLO" -eq 1 ]]; then
  if [[ -f "$MODELS/best.onnx" ]]; then
    echo "=== YOLO FP16 ==="
    build_engine \
      "$MODELS/best.onnx" \
      "$MODELS/yolo26_fp16.engine"
  else
    echo "SKIP (missing): $MODELS/best.onnx"
  fi
fi

# OCR: dynamic W → RecResize 48x320 (input name: x)
if [[ "$DO_OCR" -eq 1 ]]; then
  OCR_ONNX=""
  if [[ -f "$MODELS/plate_rec.onnx" ]]; then
    OCR_ONNX="$MODELS/plate_rec.onnx"
  elif [[ -f "$MODELS/ocr_rec.onnx" ]]; then
    OCR_ONNX="$MODELS/ocr_rec.onnx"
  fi

  if [[ -n "$OCR_ONNX" ]]; then
    echo "=== OCR FP16 ($OCR_ONNX) ==="
    build_engine \
      "$OCR_ONNX" \
      "$MODELS/plate_rec_fp16.engine" \
      --minShapes=x:1x3x48x160 \
      --optShapes=x:1x3x48x320 \
      --maxShapes=x:1x3x48x320
  else
    echo "SKIP (missing): plate_rec.onnx / ocr_rec.onnx"
  fi
fi

echo "=== done ==="
ls -lh "$MODELS"/*.engine 2>/dev/null || true
