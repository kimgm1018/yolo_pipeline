#!/usr/bin/env python3
"""
브라우저 미리보기 (SSH + Wi‑Fi용, VNC 불필요)

보드:
  cd ~/yolo-pipeline/yolo_pipeline
  source venv/bin/activate
  python view_camera_web.py --rotate 0 --port 8765

노트북 브라우저:
  http://<보드IP>:8765
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

import config
from detector import TensorRTTrackedDetector
from yolo_trt import DEFAULT_CLASS_NAMES


class State:
    def __init__(self):
        self.cv = threading.Condition()
        self.jpeg = None
        self.status = {"state": "starting"}
        self.running = True


state = State()


def orient(frame, rotate: int):
    if rotate == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotate == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotate == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def draw(frame, detections):
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        tid = d.get("track_id", -1)
        name = d["class_name"]
        conf = d["confidence"]
        label = f"{name} #{tid} {conf:.2f}"
        color = (0, 200, 255) if name == "licence" else (80, 220, 80)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            label,
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def worker(args):
    cap = None
    detector = None
    try:
        cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {args.device}")

        engine = Path(args.engine)
        if not engine.exists():
            raise FileNotFoundError(f"engine 없음: {engine}")

        detector = TensorRTTrackedDetector(
            engine_path=engine,
            tracker_config=config.TRACKER,
            input_size=args.imgsz,
            confidence_threshold=args.conf,
            iou_threshold=config.IOU,
            class_names=list(DEFAULT_CLASS_NAMES),
        )

        ema = None
        frames = 0
        started = time.perf_counter()

        while state.running:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                continue

            frame = orient(frame, args.rotate)
            detections = detector.track_frame(frame)
            view = draw(frame, detections)

            dt = time.perf_counter() - t0
            fps = (1.0 / dt) if dt > 0 else 0.0
            ema = fps if ema is None else 0.9 * ema + 0.1 * fps
            frames += 1

            infer_ms = float(detector.yolo.last_timing.get("infer_ms", 0.0))
            cv2.putText(
                view,
                f"FPS {ema:.1f} | infer {infer_ms:.1f} ms | conf {args.conf:.2f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            ok, jpg = cv2.imencode(
                ".jpg", view, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
            )
            if not ok:
                continue

            counts: dict[str, int] = {}
            for d in detections:
                n = d["class_name"]
                counts[n] = counts.get(n, 0) + 1

            with state.cv:
                state.jpeg = jpg.tobytes()
                state.status = {
                    "state": "running",
                    "fps": round(ema, 2),
                    "inference_ms": round(infer_ms, 2),
                    "frames": frames,
                    "uptime_s": round(time.perf_counter() - started, 1),
                    "detections": counts,
                    "rotate": args.rotate,
                    "engine": str(engine),
                }
                state.cv.notify_all()

    except Exception as e:
        with state.cv:
            state.status = {"state": "error", "error": str(e)}
            state.cv.notify_all()
        print(f"[error] {e}", flush=True)
    finally:
        if detector is not None:
            detector.close()
        if cap is not None:
            cap.release()


HTML = b"""<!doctype html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width">
<title>yolo_pipeline web view</title>
<style>
body{margin:0;background:#111;color:#eee;font-family:sans-serif;text-align:center}
h2{margin:10px}.wrap{max-width:1280px;margin:auto}
img{width:100%;height:auto;background:#222}
pre{text-align:left;padding:10px;margin:0;overflow:auto}
</style></head>
<body>
<div class="wrap">
<h2>Jetson TensorRT Detection</h2>
<img src="/stream.mjpg">
<pre id="s">connecting...</pre>
</div>
<script>
setInterval(()=>fetch('/status').then(r=>r.json()).then(x=>{
  s.textContent=JSON.stringify(x,null,2)
}).catch(()=>{}),1000)
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = HTML
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/status":
            body = json.dumps(state.status).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/snapshot.jpg":
            with state.cv:
                jpg = state.jpeg
            if not jpg:
                self.send_error(503)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.end_headers()
            self.wfile.write(jpg)
            return

        if self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            last = None
            try:
                while state.running:
                    with state.cv:
                        state.cv.wait_for(
                            lambda: state.jpeg is not None and state.jpeg is not last,
                            timeout=2,
                        )
                        jpg = state.jpeg
                    if not jpg or jpg is last:
                        continue
                    last = jpg
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpg)).encode()
                        + b"\r\n\r\n"
                        + jpg
                        + b"\r\n"
                    )
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        self.send_error(404)


def main():
    p = argparse.ArgumentParser(description="Jetson web camera + TensorRT view")
    p.add_argument("--engine", default=str(config.YOLO_ENGINE_PATH))
    p.add_argument("--device", default="/dev/video0")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--imgsz", type=int, default=config.IMGSZ)
    p.add_argument("--conf", type=float, default=config.CONF)
    p.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0)
    p.add_argument("--jpeg-quality", type=int, default=75)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    thread = threading.Thread(target=worker, args=(args,), daemon=True)
    thread.start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"Open http://<jetson-ip>:{args.port}  rotate={args.rotate} engine={args.engine}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
