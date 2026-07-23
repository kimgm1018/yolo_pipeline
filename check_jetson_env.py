"""Jetson 환경 검사 — ~/yolo-pipeline/yolo_pipeline 에서 실행."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def ok(msg):
    print(f"[OK]  {msg}")


def fail(msg):
    print(f"[FAIL] {msg}")


def warn(msg):
    print(f"[WARN] {msg}")


def main():
    failures = 0
    print("=== check_jetson_env ===")
    print(f"ROOT: {ROOT}")
    print(f"Python: {sys.version.split()[0]}")
    ok(f"arch={platform.machine()}")

    try:
        import numpy as np

        ok(f"numpy {np.__version__}")
    except Exception as e:
        fail(f"numpy: {e}")
        failures += 1

    try:
        import cv2

        ok(f"opencv {cv2.__version__}")
    except Exception as e:
        fail(f"opencv: {e}")
        failures += 1

    try:
        import yaml  # noqa: F401

        ok("PyYAML")
    except Exception as e:
        fail(f"PyYAML: {e}")
        failures += 1

    arch = platform.machine().lower()
    if arch in ("aarch64", "arm64"):
        try:
            import tensorrt as trt

            ok(f"tensorrt {trt.__version__}")
        except Exception as e:
            fail(f"tensorrt: {e}")
            failures += 1
        try:
            import pycuda.driver  # noqa: F401

            ok("pycuda")
        except Exception:
            try:
                from cuda import cudart  # noqa: F401

                ok("cuda-python")
            except Exception as e:
                fail(f"CUDA binding: {e}")
                failures += 1
    else:
        warn("PC host - tensorrt check skipped")

    for rel in (
        "main.py",
        "config.py",
        "models/yolo26_fp16.engine",
        "models/plate_rec_fp16.engine",
        "models/plate_dict.txt",
        "trackers/bytetrack_stable.yaml",
    ):
        p = ROOT / rel
        if p.exists():
            extra = f" ({p.stat().st_size} bytes)" if p.is_file() else ""
            ok(f"{rel}{extra}")
        else:
            fail(f"missing {rel}")
            failures += 1

    import config as cfg

    ok(f"IMGSZ={cfg.IMGSZ} BACKEND={cfg.BACKEND}")

    videos = sorted(Path("/dev").glob("video*")) if Path("/dev").exists() else []
    if videos:
        ok("cameras: " + ", ".join(str(v) for v in videos))
    else:
        warn("/dev/video* 없음")

    print("=== summary ===")
    if failures:
        fail(f"{failures} failed")
        sys.exit(1)
    ok("ready")
    print("  python main.py --source 0 --no-ocr")
    print("  python main.py --source 0")
    sys.exit(0)


if __name__ == "__main__":
    main()
