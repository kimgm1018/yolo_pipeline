#!/usr/bin/env bash
# Jetson Orin Nano — ONNX → TensorRT FP16
#
# Usage:
#   bash scripts/jetson_trt.sh
#   bash scripts/jetson_trt.sh --yolo-only
#   bash scripts/jetson_trt.sh --ocr-only
#
# 중요 (TRT 10 / trtexec):
#   --memPoolSize 접미사는 B|K|M|G 만 유효.
#   "8192MiB" 는 무시되고 workspace가 거의 0이 되어 빌드가 실패할 수 있음.
#   반드시 "8192M" 형태를 쓴다.
#
# 환경변수:
#   TRT_WORKSPACE=2048|4096|6144   (M = MiB 대략값, 기본은 가용 RAM 기반)
#   TRT_BUILDER_OPT=0..5           (기본 3; Orin에서 실패 시 0↔5 둘 다 시도)
#   TRTEXEC=/usr/src/tensorrt/bin/trtexec

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$ROOT/models"
OPT="${TRT_BUILDER_OPT:-3}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

DO_YOLO=1
DO_OCR=1
for arg in "$@"; do
  case "$arg" in
    --yolo-only) DO_OCR=0 ;;
    --ocr-only)  DO_YOLO=0 ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
  esac
done

mkdir -p "$MODELS"

if [[ ! -x "$TRTEXEC" ]]; then
  echo "trtexec 없음: $TRTEXEC"
  exit 1
fi

# MemAvailable (kB) → 권장 workspace (M). Orin 8GB에서 16G 요청은 의미 없음.
avail_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
avail_mb=$((avail_kb / 1024))
if [[ -n "${TRT_WORKSPACE:-}" ]]; then
  WS="$TRT_WORKSPACE"
else
  # 가용 RAM의 ~50%, 1024~4096M 클램프
  WS=$((avail_mb / 2))
  if (( WS < 1024 )); then WS=1024; fi
  if (( WS > 4096 )); then WS=4096; fi
fi

echo "ROOT=$ROOT"
echo "trtexec=$TRTEXEC"
echo "MemAvailable≈${avail_mb}M  workspace=${WS}M  builderOptimizationLevel=$OPT"
echo "--- memory ---"
free -h || true
swapon --show || true
echo "--------------"
echo "팁: GUI/브라우저/카메라 끄기, swap 8~16G 권장"
echo "    sudo systemctl isolate multi-user.target   # GUI off (선택)"
echo ""

# 빌드 로그에서 workspace가 실제로 잡혔는지 확인용 패턴 안내
check_workspace_log () {
  local logf="$1"
  if grep -qiE 'Maximum workspace size:.*bytes' "$logf" 2>/dev/null; then
    grep -i 'Maximum workspace size' "$logf" | tail -n 1
  fi
  # MiB 오용 징후: workspace가 수 KB~수 MB만 잡힘
  if grep -qiE 'Maximum workspace size:[[:space:]]*[0-9]{1,7} bytes' "$logf" 2>/dev/null; then
    echo "WARN: workspace가 비정상적으로 작을 수 있음. 접미사 M/G 사용 여부 확인."
  fi
}

build_engine () {
  local onnx="$1"
  local engine="$2"
  shift 2
  local extra=("$@")

  # 시도 순서: (ws, opt) — 잘못된 MiB 없이 M만 사용. 16G로 올리지 않음.
  local attempts=(
    "${WS}:${OPT}"
    "${WS}:0"
    "2048:5"
    "4096:0"
    "1024:0"
  )

  local attempt=0
  for pair in "${attempts[@]}"; do
    attempt=$((attempt + 1))
    local try_ws="${pair%%:*}"
    local try_opt="${pair##*:}"
    local logf
    logf="$(mktemp /tmp/trtexec.XXXXXX.log)"

    echo "=== build attempt $attempt ==="
    echo "  onnx=$onnx"
    echo "  engine=$engine"
    echo "  --memPoolSize=workspace:${try_ws}M  --builderOptimizationLevel=$try_opt"

    set +e
    "$TRTEXEC" \
      --onnx="$onnx" \
      --saveEngine="$engine" \
      --fp16 \
      --buildOnly \
      --memPoolSize=workspace:${try_ws}M \
      --builderOptimizationLevel="$try_opt" \
      "${extra[@]}" \
      2>&1 | tee "$logf"
    local rc=${PIPESTATUS[0]}
    set -e

    check_workspace_log "$logf"
    rm -f "$logf"

    if [[ $rc -eq 0 && -f "$engine" ]]; then
      echo "OK → $engine ($(du -h "$engine" | awk '{print $1}'))"
      return 0
    fi
    echo "FAIL (exit=$rc)"
    rm -f "$engine" 2>/dev/null || true
  done

  echo "ERROR: engine 빌드 실패: $engine"
  echo ""
  echo "다음을 확인하세요:"
  echo "  1) free -h / swapon — MemAvailable이 작으면 swap 추가 (8~16G)"
  echo "  2) GUI 종료: sudo systemctl isolate multi-user.target"
  echo "  3) 로그에 'Maximum workspace size' 가 MB~GB 단위인지 (수 KB면 접미사 오류)"
  echo "  4) 그래도 실패 시 PC에서 imgsz=320 ONNX를 다시 export 후 재시도"
  echo "     TRT_WORKSPACE=2048 TRT_BUILDER_OPT=0 bash scripts/jetson_trt.sh --yolo-only"
  return 1
}

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
