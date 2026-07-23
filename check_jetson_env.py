"""Jetson 환경 / 모델 파일 검사."""

from __future__ import annotations

import os
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
    py = sys.version.split()[0]
    print(f"Python: {py}")
    if not py.startswith("3.10"):
        warn(f"권장 Python 3.10.x (현재 {py})")
    else:
        ok(f"Python {py}")

    machine = platform.machine().lower()
    if machine in ("aarch64", "arm64"):
        ok(f"arch={machine}")
    else:
        warn(f"arch={machine} (Jetson은 aarch64)")

    try:
        import numpy as np

        ok(f"numpy {np.__version__}")
    except Exception as e:
        fail(f"numpy: {e}")
        failures += 1

    try:
        import cv2

        ok(f"opencv {cv2.__version__}")
        build = cv2.getBuildInformation()
        if "GStreamer" in build and "YES" in build.split("GStreamer")[1][:40]:
            ok("OpenCV GStreamer: YES")
        else:
            warn("OpenCV GStreamer 확인 필요")
    except Exception as e:
        fail(f"opencv: {e}")
        failures += 1

    try:
        import tensorrt as trt

        ok(f"tensorrt {trt.__version__}")
    except Exception as e:
        fail(f"tensorrt: {e}")
        failures += 1

    try:
        import pycuda.driver as cuda  # noqa: F401

        ok("pycuda available")
    except Exception:
        try:
            from cuda import cudart  # noqa: F401

            ok("cuda-python available")
        except Exception as e:
            fail(f"CUDA python binding (pycuda/cuda-python): {e}")
            failures += 1

    for rel in (
        "models/yolo26_fp16.engine",
        "models/plate_rec_fp16.engine",
        "models/plate_dict.txt",
    ):
        p = ROOT / rel
        if p.exists():
            ok(f"exists {rel} ({p.stat().st_size} bytes)")
        else:
            fail(f"missing {rel}")
            failures += 1

    # memory
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        data = meminfo.read_text()
        for key in ("MemTotal", "MemAvailable", "SwapTotal"):
            for line in data.splitlines():
                if line.startswith(key):
                    ok(line.strip())
                    break
    else:
        warn("/proc/meminfo 없음 (Windows?)")

    videos = sorted(Path("/dev").glob("video*")) if Path("/dev").exists() else []
    if videos:
        ok("cameras: " + ", ".join(str(v) for v in videos))
    else:
        warn("/dev/video* 없음")

    print("=== summary ===")
    if failures:
        fail(f"{failures} check(s) failed")
        sys.exit(1)
    ok("all critical checks passed")
    sys.exit(0)


if __name__ == "__main__":
    main()
