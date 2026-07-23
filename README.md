# jetson_pipeline — ONNX/TRT 변환·MQTT 보조

실서비스 추론은 **저장소 루트** (`main.py --backend tensorrt`)를 사용한다.  
보드 설치·실행 절차는 [`../JETSON_SETUP.md`](../JETSON_SETUP.md)를 본다.

이 폴더는 PC에서 ONNX를 만들거나, MQTT 스텁을 둘 때만 쓴다.

```text
jetson_pipeline/
├── export_config.py      # shape·경로 상수
├── export_onnx.py        # YOLO → ONNX (OCR은 가이드 출력)
├── ocr_preprocess.py     # OCR 고정 전처리 (48×320)
├── mqtt_client.py        # 긴급/배치 MQTT 스텁 (main 미연결)
├── scripts/jetson_trt.sh # 보드에서 trtexec FP16
├── docs/                 # shape / OCR ONNX 메모
└── models/               # *.onnx / *.engine 생성물 (Git 제외)
```

## YOLO ONNX (PC)

```powershell
cd jetson_pipeline
..\venv\Scripts\python.exe export_onnx.py --yolo
```

출력: `models/best.onnx` (`1x3x416x416`, 학습 imgsz와 동일)

## TensorRT FP16 (Jetson)

생성된 ONNX를 보드 `models/`로 옮긴 뒤:

```bash
cd yolo_pipeline   # 또는 jetson_pipeline
bash scripts/jetson_trt.sh
# YOLO만 / OCR만
bash scripts/jetson_trt.sh --yolo-only
bash scripts/jetson_trt.sh --ocr-only
```

Orin Nano에서 YOLO 빌드가 workspace 부족으로 실패하면:

1. **접미사**: `trtexec`는 `MiB`가 아니라 **`M`/`G`** 만 인식한다.  
   (`8192MiB` → workspace가 사실상 0에 가까워질 수 있음)
2. GUI/브라우저 끄고 **swap 8~16G** 확보
3. 스크립트 재실행 (기본이 `…M` + 자동 재시도)

```bash
# 수동 예 (접미사 M 주의)
/usr/src/tensorrt/bin/trtexec \
  --onnx=models/best.onnx \
  --saveEngine=models/yolo26_fp16.engine \
  --fp16 --skipInference \
  --memPoolSize=workspace:4096M \
  --builderOptimizationLevel=3
```

```bash
TRT_WORKSPACE=2048 TRT_BUILDER_OPT=0 bash scripts/jetson_trt.sh --yolo-only
```

스크립트 기본: 가용 RAM 기반 workspace(`M`), `builderOptimizationLevel=3`, 실패 시 여러 (ws,opt) 조합 재시도. **16G workspace로 올리지 않음** (8GB 보드에 무의미).

## MQTT

`mqtt_client.py`는 dry-run 기본. 루트 `main.send_urgent_event`와는 아직 연결되지 않았다.
