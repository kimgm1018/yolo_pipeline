# jetson_pipeline = 보드 배포 루트
# 경로 예: /home/e103/yolo-pipeline/yolo_pipeline

```bash
cd ~/yolo-pipeline/yolo_pipeline
source venv/bin/activate

python check_jetson_env.py

# 브라우저 미리보기 + 이벤트 로그 (SSH 추천)
python view_camera_web.py --rotate 0 --port 8765
python view_camera_web.py --rotate 0 --port 8765 --ocr
# 노트북: http://<보드IP>:8765  (오른쪽 events 패널)

# 이벤트 파이프라인
python main.py --source 0 --no-ocr --no-show
python main.py --source 0 --no-show
```

이벤트는 MQTT 대신 `logs/` 에 JSONL/JSON으로 저장된다.

| 파일 | 역할 |
|------|------|
| `models/yolo26_fp16.engine` | YOLO TRT (416) |
| `models/plate_rec_fp16.engine` | OCR TRT |
| `models/plate_dict.txt` | CTC dict |
| `main.py` | 이벤트 루프 |
| `view_camera_web.py` | 브라우저 MJPEG 미리보기 |
| `config.py` | 경로·IMGSZ=416·BoT-SORT |
| `trackers/botsort.yaml` | Ultralytics BoT-SORT |

엔진 재빌드: `bash scripts/jetson_trt.sh --yolo-only` / `--ocr-only`
