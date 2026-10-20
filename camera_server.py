"""Local camera inference: python camera_server.py (no web framework required)."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlsplit
import warnings
from PIL import Image, ImageOps, UnidentifiedImageError

# Keep the standard decoder before Ultralytics installs its HEIF fallback wrapper.
PIL_OPEN = Image.open
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD = 6 * 1024 * 1024
NAMES_ZH = ["安全帽", "口罩", "未戴安全帽", "未戴口罩", "未穿安全背心", "人员", "安全锥", "安全背心", "机械", "车辆"]


def default_checkpoint():
    candidates = [ROOT / "models/best.pt", ROOT.parent / "runs/ppe/yolov8n_stable/weights/best.pt"]
    return next((p for p in candidates if p.is_file()), candidates[0])


def demo_image():
    data = ROOT.parent / "construction_safety_class9_data/valid"
    labels = sorted((data / "labels").glob("*.txt"))
    for label in labels:
        classes = [int(line.split()[0]) for line in label.read_text().splitlines() if line.strip()]
        image = data / "images" / (label.stem + ".jpg")
        if 0 in classes and 7 in classes and 3 <= len(classes) <= 15 and image.is_file():
            return image
    return next(iter(sorted((data / "images").glob("*.jpg"))), None)


class CameraServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, body, content_type="application/json; charset=utf-8"):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        path = urlsplit(self.path).path
        files = {"/": ("index.html", "text/html; charset=utf-8"),
                 "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                 "/style.css": ("style.css", "text/css; charset=utf-8")}
        if path in files:
            name, mime = files[path]
            self.reply(200, (ROOT / "web" / name).read_bytes(), mime)
        elif path == "/api/health":
            self.reply(200, {"ready": True, "model_label": self.server.model_label,
                             "device": "CPU", "demo_available": self.server.demo is not None})
        elif path == "/api/demo" and self.server.demo is not None:
            self.reply(200, self.server.demo.read_bytes(), "image/jpeg")
        elif path == "/favicon.ico":
            self.reply(204, b"", "image/x-icon")
        else:
            self.reply(404, {"error": "页面不存在"})

    def do_POST(self):
        url = urlsplit(self.path)
        if url.path != "/api/predict":
            return self.reply(404, {"error": "接口不存在"})
        origin = self.headers.get("Origin")
        allowed = {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
        if origin and origin not in allowed:
            return self.reply(403, {"error": "请从本机检测网页发送请求"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            confidence = float(parse_qs(url.query).get("conf", ["0.25"])[0])
        except (ValueError, TypeError):
            return self.reply(400, {"error": "请求参数无效"})
        if not math.isfinite(confidence) or not 0.05 <= confidence <= 0.95:
            return self.reply(400, {"error": "置信度须在 0.05 到 0.95 之间"})
        if not 0 < length <= MAX_UPLOAD:
            return self.reply(413, {"error": "图片应小于 6 MB"})
        if self.headers.get("Content-Type", "").split(";")[0] not in {"image/jpeg", "image/png", "image/webp"}:
            return self.reply(415, {"error": "请发送 JPEG、PNG 或 WebP 图片"})
        if not self.server.predict_lock.acquire(blocking=False):
            return self.reply(429, {"error": "模型正在处理另一帧，请稍后重试"})
        try:
            started = time.perf_counter()
            raw = self.rfile.read(length)
            if len(raw) != length:
                return self.reply(400, {"error": "图片未接收完整"})
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with PIL_OPEN(io.BytesIO(raw), formats=("JPEG", "PNG", "WEBP")) as opened:
                        if opened.width * opened.height > 16_000_000:
                            return self.reply(413, {"error": "图片尺寸过大"})
                        image = ImageOps.exif_transpose(opened).convert("RGB")
                        image.thumbnail((1600, 1600))
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning):
                return self.reply(400, {"error": "无法读取这张图片"})
            result = self.server.model.predict(image, imgsz=640, conf=confidence,
                                               iou=0.7, max_det=100, device="cpu", verbose=False)[0]
            boxes = []
            for cls, score, xyxy in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(),
                                         result.boxes.xyxyn.tolist()):
                values = [score, *xyxy]
                if not all(math.isfinite(v) for v in values):
                    continue
                category = int(cls)
                boxes.append({"class_id": category, "label": NAMES_ZH[category],
                              "confidence": score, "xyxy": [min(1.0, max(0.0, v)) for v in xyxy]})
            self.reply(200, {"boxes": boxes, "inference_ms": round((time.perf_counter() - started) * 1000, 1),
                             "width": image.width, "height": image.height})
        except Exception:
            import traceback
            traceback.print_exc()
            self.reply(500, {"error": "本次预测失败，请重试并查看服务日志"})
        finally:
            self.server.predict_lock.release()


def main():
    parser = argparse.ArgumentParser(description="施工现场安全装备网页摄像头检测")
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint())
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"找不到模型权重：{args.checkpoint}")
    if args.open_browser:
        import urllib.request
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/api/health", timeout=1) as response:
                existing = json.load(response)
            if existing.get("ready") and "YOLOv8n" in existing.get("model_label", ""):
                import webbrowser
                webbrowser.open(f"http://127.0.0.1:{args.port}")
                print("已打开正在运行的检测服务。", flush=True)
                return
        except (OSError, ValueError):
            pass
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    print("正在加载模型…", flush=True)
    model = YOLO(str(args.checkpoint))
    expected = ["Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest", "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle"]
    if [model.names[i] for i in range(len(model.names))] != expected:
        parser.error("权重类别与本项目十类顺序不匹配")
    with CameraServer(("127.0.0.1", args.port), Handler) as server:
        server.model = model
        server.model_label = "YOLOv8n · 第 90 轮最佳模型" if args.checkpoint.resolve() == default_checkpoint().resolve() else "YOLOv8n · 自定义权重"
        server.demo = demo_image()
        server.predict_lock = threading.Lock()
        address = f"http://127.0.0.1:{server.server_port}"
        print(f"READY {address}\n摄像头画面仅在本机处理；按 Ctrl+C 结束服务。", flush=True)
        if args.open_browser:
            import webbrowser
            webbrowser.open(address)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("服务已停止。", flush=True)


if __name__ == "__main__":
    main()
