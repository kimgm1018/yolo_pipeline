# jetson_pipeline = 보드 배포 루트
# 경로 예: /home/e103/yolo-pipeline/yolo_pipeline

```bash
cd ~/yolo-pipeline/yolo_pipeline
source venv/bin/activate

python check_jetson_env.py
python main.py --source 0 --no-ocr      # YOLO+ByteTrack만
python main.py --source 0              # + OCR
python main.py --source 0 --no-show    # GUI 없이
```

이벤트는 MQTT 대신 `logs/` 에 JSONL/JSON으로 저장된다.

| 파일 | 역할 |
|------|------|
| `models/yolo26_fp16.engine` | YOLO TRT (416) |
| `models/plate_rec_fp16.engine` | OCR TRT |
| `models/plate_dict.txt` | CTC dict |
| `main.py` | 카메라 루프 |
| `config.py` | 경로·IMGSZ=416 |

엔진 재빌드: `bash scripts/jetson_trt.sh --yolo-only` / `--ocr-only`
