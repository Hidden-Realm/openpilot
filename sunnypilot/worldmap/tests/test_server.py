import json
import math
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from openpilot.sunnypilot.worldmap import builder as B
from openpilot.sunnypilot.worldmap.server import serve
from openpilot.sunnypilot.worldmap.tests.helpers import drive_straight
from openpilot.sunnypilot.worldmap.tiles import TileStore

TOKEN = "worldcode"
AUTH = {"Cookie": f"dashcam={TOKEN}"}


@pytest.fixture
def world(tmp_path, monkeypatch):
  monkeypatch.setattr(B, "wall_now", lambda: 1.8e9)
  store = TileStore(str(tmp_path))
  b = B.WorldBuilder(store)
  drive_straight(b, 0.0, 10.0)
  erased = []
  httpd = serve(b, store, lambda: TOKEN, host="127.0.0.1", port=0, on_erase=lambda: erased.append(True))
  threading.Thread(target=httpd.serve_forever, daemon=True).start()
  yield b, store, f"http://127.0.0.1:{httpd.server_address[1]}", erased
  httpd.shutdown()
  httpd.server_close()


def request(url, method="GET", headers=None, data=None, timeout=5):
  req = urllib.request.Request(url, method=method, headers=headers or {}, data=data)
  try:
    with urllib.request.urlopen(req, timeout=timeout) as r:
      return r.status, dict(r.headers), r.read() if "event-stream" not in r.headers.get("Content-Type", "") else r.readline() + r.readline()
  except urllib.error.HTTPError as e:
    return e.code, dict(e.headers), e.read()


def test_needs_the_code(world):
  _, _, base, _ = world
  status, _, body = request(f"{base}/")
  assert status == 403 and b"Enter the code" in body
  assert request(f"{base}/api/tiles")[0] == 403
  status, headers, _ = request(f"{base}/?k=World%20Code")  # spaces and case are forgiven
  assert status == 200 and f"dashcam={TOKEN}" in headers["Set-Cookie"]


def test_tiles_and_images(world):
  b, store, base, _ = world
  status, _, body = request(f"{base}/api/tiles?r=300", headers=AUTH)
  listing = json.loads(body)
  assert status == 200 and listing["origin"]["id"] == b.pose.frame_id and listing["tiles"]
  key = listing["tiles"][0]["key"]
  status, headers, png = request(f"{base}/api/tile/{key}.png", headers=AUTH)
  assert status == 200 and png[:8] == b"\x89PNG\r\n\x1a\n"
  assert request(f"{base}/api/tile/1_1.png", headers=AUTH)[0] == 404


def test_live_stream(world):
  _, _, base, _ = world
  status, _, first = request(f"{base}/api/live", headers=AUTH)
  assert status == 200 and first.startswith(b"data: ")
  live = json.loads(first[6:])
  assert live["valid"] and live["pose"]["std"] < 3.0 and math.isfinite(live["speed"])


def test_static_files(world):
  _, _, base, _ = world
  status, headers, body = request(f"{base}/static/app.js", headers=AUTH)
  assert status == 200 and headers["Cache-Control"] == "no-cache" and b"THREE" in body
  assert request(f"{base}/static/three.module.min.js", headers=AUTH)[1]["Cache-Control"].startswith("max-age")
  assert request(f"{base}/static/..%2Fworldd.py", headers=AUTH)[0] == 404


def test_erase_needs_confirmation(world):
  b, store, base, erased = world
  assert request(f"{base}/api/erase", method="POST", headers=AUTH, data=b"{}")[0] == 400
  status, _, _ = request(f"{base}/api/erase", method="POST", headers={**AUTH, "Content-Type": "application/json"},
                         data=json.dumps({"confirm": "erase"}).encode())
  assert status == 200 and erased and store.keys() == [] and not b.stats.qualified
  assert np.isfinite(b.stats.compared_km)
