"""Live dashboard in a browser window: the same full frame as the recorded video (bench view, cameras with
the AI detector, LOOK / FEEL / DECIDE, sensors, results), streamed as MJPEG from localhost.

The MuJoCo window keeps the clean 3D scene; this window shows every panel at full size, side by side.
Needs nothing extra: Python's http.server + OpenCV's JPEG encoder."""
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

PAGE = b"""<!doctype html><html><head><meta charset="utf-8"><title>Grip Lab - live dashboard</title>
<style>html,body{margin:0;height:100%;background:#121212}
img{display:block;width:100vw;height:100vh;object-fit:contain}</style></head>
<body><img src="/stream" alt="waiting for the simulation..."></body></html>"""


class LiveDashboard:
    def __init__(self, port=8765, open_browser=True):
        self.jpg, self.cond = None, threading.Condition()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if not self.path.startswith("/stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(PAGE)))
                    self.end_headers()
                    self.wfile.write(PAGE)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                last = None
                try:
                    while True:
                        with outer.cond:
                            outer.cond.wait_for(lambda: outer.jpg is not None and outer.jpg is not last, timeout=1.0)
                            jpg = outer.jpg
                        if jpg is None or jpg is last:
                            continue
                        last = jpg
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

            def log_message(self, *a):
                pass

        for p in range(port, port + 20):                 # first free port
            try:
                self.srv = ThreadingHTTPServer(("127.0.0.1", p), Handler)
                break
            except OSError:
                continue
        else:
            raise RuntimeError(f"no free port in {port}-{port + 19}")
        self.srv.daemon_threads = True
        self.url = f"http://localhost:{self.srv.server_address[1]}/"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        print(f"live dashboard: {self.url}", flush=True)
        if open_browser:
            webbrowser.open(self.url)

    def publish(self, rgb, quality=85):
        ok, buf = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            with self.cond:
                self.jpg = buf.tobytes()
                self.cond.notify_all()

    def close(self):
        self.srv.shutdown()
