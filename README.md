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

출력: `models/best.onnx` (`1x3x640x640`)

## TensorRT FP16 (Jetson)

생성된 ONNX를 보드로 옮긴 뒤:

```bash
cd jetson_pipeline
bash scripts/jetson_trt.sh
```

또는 루트 `models/`에 엔진을 두고 `JETSON_SETUP.md`의 `trtexec` 명령을 그대로 써도 된다.

## MQTT

`mqtt_client.py`는 dry-run 기본. 루트 `main.send_urgent_event`와는 아직 연결되지 않았다.
