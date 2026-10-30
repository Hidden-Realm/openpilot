#!/usr/bin/env python3
"""
Wi-Fi viewer for saved dash cam clips. Runs whenever the device is on, including
while driving, so clips can be saved to a phone without turning the car off. A clip
can be downloaded once the minute after its press has been recorded.

Open it by scanning the QR code under settings -> dash cam on the device. The
phone must be on the device's hotspot or the same Wi-Fi network as the device.
"""
import ipaddress
import json
import mimetypes
import os
import re
import shutil
import threading
import urllib.parse
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from openpilot.sunnypilot.dashcam.clips import CAMERAS, Clip, ClipStore, clip_bytes, clip_cameras
from openpilot.sunnypilot.dashcam.export import export_clip, make_thumbnail
from openpilot.sunnypilot.dashcam.hold import WHEEL_BUTTONS, set_wheel_button, wheel_button
from openpilot.sunnypilot.dashcam.upload import STATE_FILE, check_servers, clip_finished, load_config, read_status, update_config
from openpilot.sunnypilot.dashcam.viewer_url import PORT, get_token

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
COOKIE = "dashcam"
MAX_CACHED_EXPORTS = 3
CHUNK = 1024 * 1024
LABEL_TO_FILE = {label: name for name, label in CAMERAS.items()}

FORBIDDEN_PAGE = b"""<!doctype html><meta name=viewport content='width=device-width'>
<body style='font-family:sans-serif;padding:24px'><h2>Dash cam</h2>
<p>Scan the QR code under <b>settings &rarr; dash cam</b> on the device to open this page.</p>"""

# shown at / without a valid code, so the bare address can be bookmarked and the code typed in once
CODE_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark light">
<title>Dash Cam</title>
<style>
  body { margin: 0; padding: 32px 20px; font: 17px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #0f1012; color: #f2f3f5; }
  @media (prefers-color-scheme: light) { body { background: #f4f5f7; color: #14161a; } }
  main { max-width: 420px; margin: 0 auto; }
  h1 { font-size: 28px; margin: 0 0 8px; }
  p { color: #9aa0a8; margin: 0 0 20px; }
  input { width: 100%; box-sizing: border-box; font: 600 24px/1 ui-monospace, Menlo, monospace; letter-spacing: 4px;
          padding: 14px; border-radius: 12px; border: 1px solid #3a3d44; background: transparent; color: inherit; }
  button { width: 100%; margin-top: 12px; padding: 14px; font: inherit; font-weight: 600; border: 0; border-radius: 12px;
           background: #3b82f6; color: #fff; }
  .wrong { color: #ef4444; }
</style></head>
<body><main>
  <h1>Dash Cam</h1>
  <p>Enter the code shown under <b>settings &rarr; dash cam</b> on the device.</p>
  {wrong}
  <form action="/" method="get">
    <input name="k" autocomplete="off" autocapitalize="none" autocorrect="off" spellcheck="false" maxlength="16"
           placeholder="code" aria-label="code" autofocus>
    <button type="submit">Open clips</button>
  </form>
</main></body></html>
"""
STILL_RECORDING = "Still recording. It can be saved about a minute after you saved the clip."
WRONG_CODE_HTML = '<p class="wrong">That code didn\'t match. Check the device screen and try again.</p>'


def normalize_code(code: str) -> str:
  return "".join(code.split()).lower()


class Viewer:
  def __init__(self, store: ClipStore, roots: list[str], token: str, current_route=lambda: None):
    self.store = store
    self.roots = roots
    self.token = token
    self.current_route = current_route  # the drive being recorded now, None when parked
    self._locks: dict[str, threading.Lock] = {}
    self._locks_lock = threading.Lock()

  def _lock(self, key: str) -> threading.Lock:
    with self._locks_lock:
      return self._locks.setdefault(key, threading.Lock())

  def recording(self, clip: Clip) -> bool:
    """The minute after the press is still being recorded; exporting now would cut it short."""
    return not clip_finished(clip, self.current_route(), self.roots[0])

  def clip_info(self, clip: Clip, upload_state: dict) -> dict:
    return {
      "id": clip.id,
      "created": clip.created,
      "duration": round(clip.duration),
      "cameras": [CAMERAS[c] for c in clip_cameras(clip, self.roots)],
      "bytes": clip_bytes(clip, self.roots),
      "uploaded": "done" in upload_state.get(clip.id, []),
      "recording": self.recording(clip),
    }

  def upload_state(self) -> dict:
    try:
      with open(os.path.join(self.store.root, STATE_FILE)) as f:
        return json.load(f)
    except (OSError, ValueError):
      return {}

  def storage(self) -> dict:
    try:
      usage = shutil.disk_usage(self.roots[0])
      return {"free": usage.free, "total": usage.total}
    except OSError:
      return {"free": 0, "total": 0}

  def thumbnail(self, clip: Clip) -> str | None:
    path = os.path.join(self.store.thumbs_dir, f"{clip.id}.jpg")
    with self._lock(f"thumb:{clip.id}"):
      if os.path.isfile(path) or make_thumbnail(clip, self.roots, path):
        return path
    return None

  def export(self, clip: Clip, label: str) -> str | None:
    cam_file = LABEL_TO_FILE.get(label)
    if cam_file is None:
      return None
    path = os.path.join(self.store.exports_dir, f"{clip.id}.{label}.mp4")
    with self._lock(f"export:{path}"):
      if os.path.isfile(path):
        os.utime(path)
        return path
      os.makedirs(self.store.exports_dir, exist_ok=True)
      self._trim_exports()
      return path if export_clip(clip, cam_file, self.roots, path) else None

  def _trim_exports(self) -> None:
    """Exports are full-quality copies, so keep only the few most recently used."""
    try:
      files = [os.path.join(self.store.exports_dir, f) for f in os.listdir(self.store.exports_dir) if f.endswith(".mp4")]
    except OSError:
      return
    files.sort(key=os.path.getmtime, reverse=True)
    for old in files[MAX_CACHED_EXPORTS - 1:]:
      try:
        os.remove(old)
      except OSError:
        pass


def make_handler(viewer: Viewer):
  class Handler(BaseHTTPRequestHandler):
    server_version = "dashcam"

    def log_message(self, *args):
      pass

    # ---- helpers
    def _authed(self, query: dict) -> bool:
      if normalize_code(query.get("k", [""])[0]) == viewer.token:
        return True
      jar = cookies.SimpleCookie(self.headers.get("Cookie", ""))
      return COOKIE in jar and jar[COOKIE].value == viewer.token

    def _local_client(self) -> bool:
      try:
        ip = ipaddress.ip_address(self.client_address[0])
        return ip.is_private or ip.is_loopback
      except ValueError:
        return False

    def _send(self, code: int, body: bytes, ctype: str, headers: dict | None = None) -> None:
      self.send_response(code)
      self.send_header("Content-Type", ctype)
      self.send_header("Content-Length", str(len(body)))
      self.send_header("Cache-Control", "no-store")
      for k, v in (headers or {}).items():
        self.send_header(k, v)
      self.end_headers()
      if self.command != "HEAD":
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
      self._send(code, json.dumps(obj).encode(), "application/json")

    def _forbidden(self, path: str = "", query: dict | None = None) -> None:
      if path == "/" and self._local_client():
        wrong = WRONG_CODE_HTML if (query or {}).get("k") else ""
        return self._send(403, CODE_PAGE.replace("{wrong}", wrong).encode(), "text/html; charset=utf-8")
      self._send(403, FORBIDDEN_PAGE, "text/html; charset=utf-8")

    def _file(self, path: str, ctype: str, download_name: str | None = None) -> None:
      """Serve a file with Range support; Safari needs it to play and download video."""
      size = os.path.getsize(path)
      start, end = 0, size - 1
      rng = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "").strip())
      if rng and (rng.group(1) or rng.group(2)):
        if rng.group(1):
          start = int(rng.group(1))
          end = min(int(rng.group(2)), size - 1) if rng.group(2) else size - 1
        else:
          start = max(0, size - int(rng.group(2)))
        if start > end or start >= size:
          self.send_response(416)
          self.send_header("Content-Range", f"bytes */{size}")
          self.end_headers()
          return
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
      else:
        self.send_response(200)
      self.send_header("Content-Type", ctype)
      self.send_header("Content-Length", str(end - start + 1))
      self.send_header("Accept-Ranges", "bytes")
      if download_name:
        self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
      self.end_headers()
      if self.command == "HEAD":
        return
      with open(path, "rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
          chunk = f.read(min(CHUNK, remaining))
          if not chunk:
            break
          try:
            self.wfile.write(chunk)
          except (BrokenPipeError, ConnectionResetError):
            return
          remaining -= len(chunk)

    def _route(self):
      url = urllib.parse.urlsplit(self.path)
      return url.path, urllib.parse.parse_qs(url.query)

    # ---- methods
    def do_HEAD(self):
      self.do_GET()

    def do_GET(self):
      path, query = self._route()
      if not self._local_client() or not self._authed(query):
        return self._forbidden(path, query)

      if path == "/":
        with open(os.path.join(STATIC_DIR, "index.html"), "rb") as f:
          body = f.read()
        cookie = f"{COOKIE}={viewer.token}; Path=/; Max-Age=31536000; SameSite=Strict; HttpOnly"
        return self._send(200, body, "text/html; charset=utf-8", {"Set-Cookie": cookie})

      if path == "/api/clips":
        state = viewer.upload_state()
        clips = [viewer.clip_info(c, state) for c in viewer.store.list()]
        return self._json({"clips": clips, "storage": viewer.storage(), "saved_bytes": sum(c["bytes"] for c in clips)})

      if path == "/api/upload":
        return self._json({"config": load_config(viewer.store.root).public(), "status": read_status(viewer.store.root)})

      if path == "/api/settings":
        return self._json({"wheel_button": wheel_button(viewer.store.root)})

      if path == "/api/diagnostics.zip":
        from openpilot.sunnypilot.diagnostics.logs import bundle, bundle_name, diag_dir
        return self._send(200, bundle(diag_dir()), "application/zip", {"Content-Disposition": f'attachment; filename="{bundle_name()}"'})

      m = re.fullmatch(r"/thumb/([\w.-]+)\.jpg", path)
      if m:
        clip = viewer.store.get(m.group(1))
        thumb = viewer.thumbnail(clip) if clip else None
        return self._file(thumb, "image/jpeg") if thumb else self._send(404, b"", "text/plain")

      m = re.fullmatch(r"/video/([\w.-]+)/(\w+)\.mp4", path)
      if m:
        clip = viewer.store.get(m.group(1))
        if clip and viewer.recording(clip):
          return self._send(409, STILL_RECORDING.encode(), "text/plain")
        mp4 = viewer.export(clip, m.group(2)) if clip else None
        if mp4 is None:
          return self._send(404, b"video not found", "text/plain")
        name = None
        if query.get("dl"):
          name = re.sub(r"[^\w.-]", "_", query.get("name", [f"dashcam_{m.group(2)}"])[0])[:80] + ".mp4"
        return self._file(mp4, mimetypes.types_map.get(".mp4", "video/mp4"), name)

      self._send(404, b"not found", "text/plain")

    def do_POST(self):
      path, query = self._route()
      if not self._local_client() or not self._authed(query):
        return self._forbidden()

      m = re.fullmatch(r"/api/clips/([\w.-]+)/prepare/(\w+)", path)
      if m:
        clip = viewer.store.get(m.group(1))
        if clip and viewer.recording(clip):
          return self._json({"error": STILL_RECORDING}, 409)
        mp4 = viewer.export(clip, m.group(2)) if clip else None
        if mp4 is None:
          return self._json({"error": "no video for this clip"}, 404)
        return self._json({"url": f"/video/{clip.id}/{m.group(2)}.mp4", "bytes": os.path.getsize(mp4)})

      m = re.fullmatch(r"/api/clips/([\w.-]+)/delete", path)
      if m:
        viewer.store.delete(m.group(1))
        return self._json({"ok": True})

      if path in ("/api/upload", "/api/settings"):
        try:
          length = int(self.headers.get("Content-Length", "0"))
          changes = json.loads(self.rfile.read(min(length, 64 * 1024)) or b"{}")
        except ValueError:
          return self._json({"error": "bad settings"}, 400)
        if not isinstance(changes, dict):
          return self._json({"error": "bad settings"}, 400)
        if path == "/api/settings":
          if changes.get("wheel_button") not in WHEEL_BUTTONS:
            return self._json({"error": "bad settings"}, 400)
          return self._json({"wheel_button": set_wheel_button(viewer.store.root, changes["wheel_button"])})
        return self._json({"config": update_config(viewer.store.root, changes).public()})

      if path == "/api/upload/test":
        return self._json({"results": check_servers(load_config(viewer.store.root))})

      self._send(404, b"not found", "text/plain")

  return Handler


def serve(viewer: Viewer, host: str = "0.0.0.0", port: int = PORT) -> ThreadingHTTPServer:
  httpd = ThreadingHTTPServer((host, port), make_handler(viewer))
  httpd.daemon_threads = True
  return httpd


def main():
  from openpilot.common.params import Params
  from openpilot.common.swaglog import cloudlog
  from openpilot.sunnypilot.dashcam.clips import dashcam_root, log_roots

  try:
    os.nice(10)  # also runs while driving; never compete with driving processes
  except (AttributeError, OSError):
    pass

  params = Params()
  root = dashcam_root()
  store = ClipStore(root)
  store.clear_exports()
  viewer = Viewer(store, log_roots(), get_token(root), lambda: params.get("CurrentRoute"))
  httpd = serve(viewer)
  cloudlog.warning(f"dashcam viewer listening on port {PORT}")
  try:
    httpd.serve_forever()
  finally:
    httpd.server_close()
    store.clear_exports()


if __name__ == "__main__":
  main()
