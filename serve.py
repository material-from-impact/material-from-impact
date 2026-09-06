"""
Range-capable static server for the project page.

Why: VS Code "Live Server" (ritwickdey) and Python's built-in http.server do NOT
support HTTP Range (206) requests, so <video>/<audio> stop after the first buffered
chunk ("plays 1 second then spins"). Opening via file:// fixes media but breaks
<model-viewer> (browsers block fetch()/ES-modules over file://).

This server supports Range requests AND serves over http, so video, audio, and the
interactive 3D models all work at once.

Run from anywhere:
    python gitpage/serve.py            # serves the gitpage/ folder on :8000
    python gitpage/serve.py 8080       # custom port
then open  http://localhost:8000/index.html
"""
import os
import sys
import http.server
import socketserver

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
ROOT = os.path.dirname(os.path.abspath(__file__))          # the gitpage/ folder

MIME = {".glb": "model/gltf-binary", ".gltf": "model/gltf+json",
        ".wav": "audio/wav", ".mp4": "video/mp4", ".js": "text/javascript",
        ".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml"}


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler + HTTP Range (206) support."""

    def guess_type(self, path):
        ext = os.path.splitext(path)[1].lower()
        return MIME.get(ext) or super().guess_type(path)

    def do_GET(self):
        rng = self.headers.get("Range")
        if rng is None:
            return super().do_GET()
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().do_GET()
        size = os.path.getsize(path)
        try:
            unit, rest = rng.split("=")
            start_s, end_s = rest.split("-")
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else size - 1
        except ValueError:
            return super().do_GET()
        start = max(0, start); end = min(end, size - 1)
        if start > end:
            self.send_error(416, "Requested Range Not Satisfiable")
            return
        length = end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


os.chdir(ROOT)
with Server(("127.0.0.1", PORT), RangeHandler) as httpd:
    print(f"Serving {ROOT}\n  ->  http://localhost:{PORT}/index.html   (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
