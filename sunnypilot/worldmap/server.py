"""
The live 3D world viewer, on the device's Wi-Fi (port 8100).

The device does no rendering for it. It streams small JSON updates (where the car is, what the model and the
detector see, how the map compares) ten times a second over server-sent events, and serves map tiles as PNG
images: red = road, green = road edges, blue = lane lines, alpha = how often the cell was observed. The
browser draws everything with WebGL, so an iPad does the 3D work.

Access works like the dash cam viewer and shares its code: one code, shown on the device, opens both. The
browser keeps it in a cookie, and cookies are per host, so opening either page unlocks the other.
"""
import ipaddress
import json
import os
import re
import threading
import time
import urllib.parse
from collections import OrderedDict
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from openpilot.sunnypilot.worldmap import geo
from openpilot.sunnypilot.worldmap.imgops import png_rgba
from openpilot.sunnypilot.worldmap.tiles import Tile, TileStore, parse_key

PORT = 8100
COOKIE = "dashcam"            # shared with the dash cam viewer on port 8099
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
STREAM_PERIOD = 0.1
MAX_LIST_TILES = 600
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css",
                ".txt": "text/plain; charset=utf-8"}
VENDORED = ("three.module.min.js", "OrbitControls.js", "three-LICENSE.txt")

CODE_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark"><title>World View</title>
<style>
  body { margin: 0; padding: 32px 20px; font: 17px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #0b1020; color: #e8ecf4; }
  main { max-width: 420px; margin: 0 auto; }
  h1 { font-size: 28px; margin: 0 0 8px; }
  p { color: #93a0b8; margin: 0 0 20px; }
  input { width: 100%; box-sizing: border-box; font: 600 24px/1 ui-monospace, Menlo, monospace; letter-spacing: 4px;
          padding: 14px; border-radius: 12px; border: 1px solid #2c3550; background: transparent; color: inherit; }
  button { width: 100%; margin-top: 12px; padding: 14px; font: inherit; font-weight: 600; border: 0; border-radius: 12px;
           background: #2dd4bf; color: #04201c; }
  .wrong { color: #f87171; }
</style></head>
<body><main>
  <h1>World View</h1>
  <p>Enter the code shown under <b>settings &rarr; 3D world</b> (or dash cam) on the device.</p>
  {wrong}
  <form action="/" method="get">
    <input name="k" autocomplete="off" autocapitalize="none" spellcheck="false" inputmode="text" placeholder="code" autofocus>
    <button type="submit">Open</button>
  </form>
</main></body></html>
"""
WRONG_CODE_HTML = '<p class="wrong">That code didn\'t match. Check the device screen and try again.</p>'


def normalize_code(code: str) -> str:
  return "".join(code.split()).lower()


def tile_png(tile: Tile) -> bytes:
  """RGBA image of a tile: R road, G edges, B lane lines, A how well observed. Row 0 is north."""
  seen = tile.layers["seen"].astype(np.float32)
  denom = np.maximum(seen, 1.0)

  def ratio(name: str, gain: float) -> np.ndarray:
    return np.clip(tile.layers[name].astype(np.float32) / denom * gain * 255.0, 0, 255).astype(np.uint8)

  observed = np.clip(seen / 4.0 * 255.0, 0, 255).astype(np.uint8)
  return png_rgba(np.dstack([ratio("road", 1.0), ratio("edge", 2.5), ratio("lane", 2.5), observed]))


def tile_version(store: TileStore, key: tuple[int, int]) -> str:
  """Changes whenever the tile's content does: its revision while in memory, else its file's time."""
  tile = store.cached(key)
  if tile is not None:
    return f"m{id(tile)}.{tile.revision}"
  try:
    return f"d{os.path.getmtime(store.path(key)):.3f}"
  except OSError:
    return "none"


class TileImages:
  """PNG cache keyed by tile version."""

  def __init__(self, store: TileStore, size: int = 64):
    self.store = store
    self.size = size
    self._cache: OrderedDict[tuple[int, int], tuple[str, bytes]] = OrderedDict()
    self._lock = threading.Lock()

  def get(self, key: tuple[int, int]) -> bytes | None:
    version = tile_version(self.store, key)
    with self._lock:
      hit = self._cache.get(key)
      if hit is not None and hit[0] == version:
        self._cache.move_to_end(key)
        return hit[1]
    tile = self.store.cached(key)
    if tile is None:
      path = self.store.path(key)
      if not os.path.isfile(path):
        return None
      try:
        tile = Tile.load(key, path)  # read on the side: viewing the map never pushes tiles the car needs out of memory
      except (OSError, ValueError, KeyError):
        return None
    png = tile_png(tile)
    with self._lock:
      self._cache[key] = (version, png)
      while len(self._cache) > self.size:
        self._cache.popitem(last=False)
    return png


def tile_listing(builder, store: TileStore, radius: float | None) -> dict:
  """Tiles with their corners in the live local frame (the one the stream's pose uses)."""
  frame = builder.pose.frame
  if frame is None:
    return {"origin": None, "tiles": []}
  e, n, _ = builder.corrected()
  rows = []
  for key in store.keys():
    corners = geo.tile_corners_ll(key)
    ce, cn = frame.to_en([c[0] for c in corners], [c[1] for c in corners])
    d = float(np.hypot(np.mean(ce) - e, np.mean(cn) - n))
    if radius is not None and d > radius + geo.TILE_M:
      continue
    rows.append((d, {"key": f"{key[0]}_{key[1]}", "rev": tile_version(store, key),
                     "corners": [[round(float(a), 2), round(float(b), 2)] for a, b in zip(ce, cn, strict=True)]}))
  rows.sort(key=lambda r: r[0])
  return {"origin": {"id": builder.pose.frame_id, "lat": frame.lat0, "lon": frame.lon0}, "tiles": [r[1] for r in rows[:MAX_LIST_TILES]]}


def make_handler(builder, store: TileStore, images: TileImages, token_fn, on_erase):
  class Handler(BaseHTTPRequestHandler):
    server_version = "worldmap"
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
      pass

    def _authed(self, query: dict) -> bool:
      token = token_fn()
      if normalize_code(query.get("k", [""])[0]) == token:
        return True
      jar = cookies.SimpleCookie(self.headers.get("Cookie", ""))
      return COOKIE in jar and jar[COOKIE].value == token

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
      for k, v in (headers or {"Cache-Control": "no-store"}).items():
        self.send_header(k, v)
      self.end_headers()
      if self.command != "HEAD":
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
      self._send(code, json.dumps(obj).encode(), "application/json")

    def _route(self):
      url = urllib.parse.urlsplit(self.path)
      return url.path, urllib.parse.parse_qs(url.query)

    def do_HEAD(self):
      self.do_GET()

    def do_GET(self):
      path, query = self._route()
      if not self._local_client():
        return self._send(403, b"forbidden", "text/plain")
      if not self._authed(query):
        if path == "/":
          wrong = WRONG_CODE_HTML if query.get("k") else ""
          return self._send(403, CODE_PAGE.replace("{wrong}", wrong).encode(), "text/html; charset=utf-8")
        return self._send(403, b"forbidden", "text/plain")

      if path == "/":
        with open(os.path.join(STATIC_DIR, "index.html"), "rb") as f:
          body = f.read()
        cookie = f"{COOKIE}={token_fn()}; Path=/; Max-Age=31536000; SameSite=Strict; HttpOnly"
        return self._send(200, body, "text/html; charset=utf-8", {"Set-Cookie": cookie, "Cache-Control": "no-store"})

      m = re.fullmatch(r"/static/([\w.-]+)", path)
      if m:
        name = m.group(1)
        full = os.path.join(STATIC_DIR, name)
        ext = os.path.splitext(name)[1]
        if ext not in STATIC_TYPES or not os.path.isfile(full):
          return self._send(404, b"not found", "text/plain")
        # the viewer's own files change with updates; only the vendored three.js files are safe to keep
        cache = "max-age=86400" if name in VENDORED else "no-cache"
        with open(full, "rb") as f:
          return self._send(200, f.read(), STATIC_TYPES[ext], {"Cache-Control": cache})

      if path == "/api/live":
        return self._stream()

      if path == "/api/tiles":
        radius = None if query.get("all") else float(query.get("r", ["400"])[0])
        return self._json(tile_listing(builder, store, radius))

      m = re.fullmatch(r"/api/tile/(-?\d+_-?\d+)\.png", path)
      if m:
        key = parse_key(m.group(1) + ".npz")
        png = images.get(key) if key is not None else None
        if not png:
          return self._send(404, b"", "image/png")
        return self._send(200, png, "image/png", {"Cache-Control": "no-cache"})

      if path == "/api/diagnostics.zip":
        from openpilot.sunnypilot.diagnostics.logs import bundle, bundle_name, diag_dir
        return self._send(200, bundle(diag_dir()), "application/zip", {"Content-Disposition": f'attachment; filename="{bundle_name()}"'})

      if path == "/api/stats":
        return self._json({"tiles": len(store.keys()), "bytes": store.total_bytes, "limit": store.limit_bytes,
                           "steer": builder.live.get("steer", {})})

      self._send(404, b"not found", "text/plain")

    def do_POST(self):
      path, query = self._route()
      if not self._local_client() or not self._authed(query):
        return self._send(403, b"forbidden", "text/plain")
      if path == "/api/erase":
        try:
          length = int(self.headers.get("Content-Length", "0"))
          body = json.loads(self.rfile.read(min(length, 4096)) or b"{}")
        except ValueError:
          body = {}
        if not isinstance(body, dict) or body.get("confirm") != "erase":
          return self._json({"error": "send {\"confirm\": \"erase\"} to erase the map"}, 400)
        store.erase_all()
        builder.stats.compared_km = builder.stats.agreed_km = 0.0
        builder.stats.qualified = False
        builder.window = None
        if on_erase is not None:
          on_erase()
        return self._json({"ok": True})
      self._send(404, b"not found", "text/plain")

    def _stream(self) -> None:
      self.send_response(200)
      self.send_header("Content-Type", "text/event-stream")
      self.send_header("Cache-Control", "no-store")
      self.send_header("Connection", "keep-alive")
      self.end_headers()
      self.close_connection = True
      last = None
      try:
        while True:
          live = builder.live
          if live is not last:
            self.wfile.write(b"data: " + json.dumps(live, separators=(",", ":")).encode() + b"\n\n")
            self.wfile.flush()
            last = live
          time.sleep(STREAM_PERIOD)
      except (BrokenPipeError, ConnectionResetError, OSError):
        return

  return Handler


def serve(builder, store: TileStore, token_fn, host: str = "0.0.0.0", port: int = PORT, on_erase=None) -> ThreadingHTTPServer:
  httpd = ThreadingHTTPServer((host, port), make_handler(builder, store, TileImages(store), token_fn, on_erase))
  httpd.daemon_threads = True
  return httpd


def start_server(builder, store: TileStore, on_erase=None) -> ThreadingHTTPServer | None:
  from openpilot.common.swaglog import cloudlog
  from openpilot.sunnypilot.dashcam.clips import dashcam_root
  from openpilot.sunnypilot.dashcam.viewer_url import get_token

  root = dashcam_root()
  try:
    httpd = serve(builder, store, lambda: get_token(root), on_erase=on_erase)
  except OSError:
    cloudlog.exception("worldmap: viewer could not start")
    return None
  threading.Thread(target=httpd.serve_forever, daemon=True).start()
  cloudlog.warning(f"worldmap viewer listening on port {PORT}")
  return httpd
