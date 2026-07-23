# Export / 입력 shape

코드: [`export_config.py`](../export_config.py), 전처리: [`ocr_preprocess.py`](../ocr_preprocess.py)

## YOLO

| 항목 | 값 |
|------|-----|
| imgsz | 640 |
| NCHW | `1, 3, 640, 640` |

```powershell
cd jetson_pipeline
..\venv\Scripts\python.exe export_onnx.py --yolo
```

## OCR (Paddle TextRecognition)

`../ocr_finetune_rec/inference.yml` RecResizeImg:

| 항목 | 값 |
|------|-----|
| shape | `3, 48, 320` → NCHW `1,3,48,320` |
| TRT 전처리 | `preprocess_ocr_fixed()` 한 경로만 |

자세한 OCR ONNX 변환: [`ocr_onnx_export.md`](ocr_onnx_export.md)
