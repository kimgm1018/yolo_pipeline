# OCR ONNX export 가이드

Paddle 모델: `../ocr_finetune_rec`  
목표: `../jetson_pipeline/models/ocr_rec.onnx` (입력 `1x3x48x320`)

## paddle2onnx (가능한 경우)

```bash
paddle2onnx \
  --model_dir ../ocr_finetune_rec \
  --model_filename inference.json \
  --params_filename inference.pdiparams \
  --save_file models/ocr_rec.onnx \
  --opset_version 17 \
  --enable_onnx_checker True
```

Paddle 3 PIR(`inference.json`)이면 도구 버전에 따라 실패할 수 있다.  
실패 시 PaddleX export 문서를 따르거나, 우선 YOLO ONNX만 진행한다.

전처리는 `ocr_preprocess.preprocess_ocr_fixed`와 동일해야 한다.  
보드 쪽 인식기는 루트 `plate_trt.py`를 사용한다.
