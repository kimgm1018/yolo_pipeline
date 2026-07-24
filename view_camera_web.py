#!/usr/bin/env python3
"""
브라우저 미리보기 + 실시간 이벤트 로그 (SSH + Wi‑Fi)

보드:
  cd ~/yolo-pipeline/yolo_pipeline
  source venv/bin/activate
  python view_camera_web.py --rotate 0 --port 8765
  python view_camera_web.py --rotate 0 --port 8765 --ocr   # OCR 포함

노트북:
  http://<보드IP>:8765
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

import config
from detector import UltralyticsTrackedDetector
from event_manager import EventManager
from plate_collector import PlateCollector
from yolo_trt import DEFAULT_CLASS_NAMES


class State:
    def __init__(self, max_events: int = 80):
        self.cv = threading.Condition()
        self.jpeg = None
        self.status = {"state": "starting"}
        self.events = deque(maxlen=max_events)  # newest last
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


def push_event(kind: str, event: dict, log_fp) -> None:
    row = {
        "ts": datetime.now().astimezone().isoformat(),
        "kind": kind,
        "event": event,
    }
    with state.cv:
        state.events.append(row)
        state.cv.notify_all()
    line = json.dumps(row, ensure_ascii=False)
    print(f"[{kind}]", json.dumps(event, ensure_ascii=False), flush=True)
    if log_fp is not None:
        log_fp.write(line + "\n")
        log_fp.flush()


def worker(args):
    cap = None
    detector = None
    plate_collector = None
    log_fp = None
    try:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = config.LOG_DIR / f"web_events_{stamp}.jsonl"
        log_fp = log_path.open("a", encoding="utf-8")
        print(f"Event log → {log_path}", flush=True)

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

        detector = UltralyticsTrackedDetector(
            engine_path=engine,
            tracker_config=config.TRACKER,
            input_size=args.imgsz,
            confidence_threshold=args.conf,
            iou_threshold=config.IOU,
            class_names=list(DEFAULT_CLASS_NAMES),
        )
        event_manager = EventManager()

        if args.ocr:
            ocr_engine = Path(args.ocr_engine)
            ocr_dict = Path(args.ocr_dict)
            if not ocr_engine.exists():
                raise FileNotFoundError(f"OCR engine 없음: {ocr_engine}")
            if not ocr_dict.exists():
                raise FileNotFoundError(f"OCR dict 없음: {ocr_dict}")
            plate_collector = PlateCollector(
                backend="tensorrt",
                engine_path=str(ocr_engine),
                dict_path=str(ocr_dict),
                min_conf=0.25,
            )

        ema = None
        frames = 0
        started = time.perf_counter()
        event_count = 0

        while state.running:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                continue

            frame = orient(frame, args.rotate)
            detections = detector.track_frame(frame)
            view = draw(frame, detections)

            if plate_collector is not None:
                plate_collector.update(detections, frame, frames)

            batch_events, urgent_events = event_manager.create_events(
                detections=detections,
                x=config.CURRENT_X,
                y=config.CURRENT_Y,
                robot_id=config.ROBOT_ID,
            )
            for ev in urgent_events:
                push_event("URGENT", ev, log_fp)
                event_count += 1
            for ev in batch_events:
                push_event("BATCH", ev, log_fp)
                event_count += 1

            dt = time.perf_counter() - t0
            fps = (1.0 / dt) if dt > 0 else 0.0
            ema = fps if ema is None else 0.9 * ema + 0.1 * fps
            frames += 1

            infer_ms = float(detector.last_timing.get("infer_ms", 0.0))
            cv2.putText(
                view,
                f"FPS {ema:.1f} | infer {infer_ms:.1f} ms | events {event_count}",
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
                    "event_count": event_count,
                    "ocr": bool(args.ocr),
                    "rotate": args.rotate,
                    "engine": str(engine),
                }
                state.cv.notify_all()

        # session end: unmatched plates
        if plate_collector is not None:
            plates = plate_collector.all_plates()
            unmatched = event_manager.create_unmatched_plate_events(
                plates=plates,
                x=config.CURRENT_X,
                y=config.CURRENT_Y,
            )
            for ev in unmatched:
                push_event("BATCH", ev, log_fp)

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
        if log_fp is not None:
            log_fp.close()


HTML = b"""<!doctype html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width">
<title>yolo_pipeline web view</title>
<style>
body{margin:0;background:#111;color:#eee;font-family:sans-serif}
.wrap{max-width:1280px;margin:auto;padding:8px}
h2{margin:8px 0;text-align:center}
img{width:100%;height:auto;background:#222;display:block}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}
@media (max-width:900px){.grid{grid-template-columns:1fr}}
pre{text-align:left;padding:10px;margin:0;overflow:auto;background:#1a1a1a;
  border:1px solid #333;max-height:280px;font-size:12px}
.urgent{color:#ff8a80}.batch{color:#80cbc4}h3{margin:4px 0 0;font-size:14px}
</style></head>
<body>
<div class="wrap">
<h2>Jetson TensorRT + Event Log</h2>
<img src="/stream.mjpg" alt="stream">
<div class="grid">
  <div>
    <h3>status</h3>
    <pre id="s">connecting...</pre>
  </div>
  <div>
    <h3>events (newest top)</h3>
    <pre id="e">waiting...</pre>
  </div>
</div>
</div>
<script>
function fmtEvents(list){
  if(!list||!list.length) return '(no events yet)';
  return list.slice().reverse().map(row=>{
    const k=row.kind||'';
    const ev=row.event||{};
    const title=ev.eventTitle||ev.eventType||'';
    const t=row.ts||'';
    return '['+k+'] '+t+'\\n  '+title+' | '+JSON.stringify(ev);
  }).join('\\n\\n');
}
setInterval(()=>{
  fetch('/status').then(r=>r.json()).then(x=>{
    s.textContent=JSON.stringify(x,null,2)
  }).catch(()=>{});
  fetch('/events').then(r=>r.json()).then(x=>{
    e.textContent=fmtEvents(x.events||[])
  }).catch(()=>{});
},500)
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
            with state.cv:
                status = dict(state.status)
                status["recent_events"] = len(state.events)
            self._json(status)
            return

        if self.path == "/events":
            with state.cv:
                events = list(state.events)
            self._json({"events": events, "count": len(events)})
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
    p = argparse.ArgumentParser(description="Jetson web view + event log")
    p.add_argument("--engine", default=str(config.YOLO_ENGINE_PATH))
    p.add_argument("--ocr-engine", default=str(config.OCR_ENGINE_PATH))
    p.add_argument("--ocr-dict", default=str(config.OCR_DICT_PATH))
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
    p.add_argument("--ocr", action="store_true", help="enable plate OCR")
    args = p.parse_args()

    thread = threading.Thread(target=worker, args=(args,), daemon=True)
    thread.start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"Open http://<jetson-ip>:{args.port}  rotate={args.rotate} ocr={args.ocr}",
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
