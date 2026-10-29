import json
import os
import threading
import urllib.error
import urllib.request

import pytest

from openpilot.sunnypilot.dashcam.clips import ClipStore, make_clip
from openpilot.sunnypilot.dashcam.server import Viewer, serve
from openpilot.sunnypilot.dashcam.tests.helpers import make_hevc, requires_hevc_encoder

ROUTE = "0000002a--6c3b9f3e1d"
TOKEN = "testtoken"


@pytest.fixture
def viewer(tmp_path):
  log_root = tmp_path / "realdata"
  for n in range(2, 7):
    (log_root / f"{ROUTE}--{n}").mkdir(parents=True)
  store = ClipStore(str(tmp_path / "dashcam"))
  clip = make_clip(ROUTE, 5, 30.0, now=1761660000.0)
  store.save(clip)

  v = Viewer(store, [str(log_root)], TOKEN)
  httpd = serve(v, host="127.0.0.1", port=0)
  threading.Thread(target=httpd.serve_forever, daemon=True).start()
  base = f"http://127.0.0.1:{httpd.server_address[1]}"
  yield v, clip, log_root, base
  httpd.shutdown()
  httpd.server_close()


def request(url, method="GET", headers=None):
  req = urllib.request.Request(url, method=method, headers=headers or {})
  try:
    with urllib.request.urlopen(req) as r:
      return r.status, dict(r.headers), r.read()
  except urllib.error.HTTPError as e:
    return e.code, dict(e.headers), e.read()


AUTH = {"Cookie": f"dashcam={TOKEN}"}


def test_requires_access_code(viewer):
  _, _, _, base = viewer
  assert request(f"{base}/")[0] == 403
  assert request(f"{base}/api/clips")[0] == 403
  assert request(f"{base}/api/clips", headers={"Cookie": "dashcam=wrong"})[0] == 403


def test_bare_address_asks_for_code(viewer):
  _, _, _, base = viewer
  status, _, body = request(f"{base}/")
  assert status == 403
  assert b'name="k"' in body and b"didn't match" not in body

  status, _, body = request(f"{base}/?k=nope")
  assert status == 403
  assert b"didn't match" in body


def test_typed_code_ignores_case_and_spaces(viewer):
  _, _, _, base = viewer
  status, headers, _ = request(f"{base}/?k=%20TestTo%20ken%20")
  assert status == 200
  assert f"dashcam={TOKEN}" in headers["Set-Cookie"]


def test_qr_link_sets_cookie(viewer):
  _, _, _, base = viewer
  status, headers, body = request(f"{base}/?k={TOKEN}")
  assert status == 200
  assert b"Dash Cam" in body
  assert f"dashcam={TOKEN}" in headers["Set-Cookie"]


def test_lists_clips(viewer):
  _, clip, _, base = viewer
  status, _, body = request(f"{base}/api/clips", headers=AUTH)
  data = json.loads(body)
  assert status == 200
  assert [c["id"] for c in data["clips"]] == [clip.id]
  assert data["clips"][0]["duration"] == 240
  assert data["clips"][0]["cameras"] == []  # no camera files yet


def test_delete(viewer):
  v, clip, _, base = viewer
  assert request(f"{base}/api/clips/{clip.id}/delete", method="POST", headers=AUTH)[0] == 200
  assert v.store.list() == []


def test_upload_settings_never_return_password(viewer):
  _, _, _, base = viewer
  body = json.dumps({"home_url": "http://10.0.0.5/dav", "username": "me", "password": "hunter2"}).encode()
  req = urllib.request.Request(f"{base}/api/upload", data=body, method="POST", headers={**AUTH, "Content-Type": "application/json"})
  with urllib.request.urlopen(req) as r:
    saved = json.loads(r.read())["config"]
  assert saved["home_url"] == "http://10.0.0.5/dav" and saved["password_set"]

  status, _, body = request(f"{base}/api/upload", headers=AUTH)
  assert status == 200
  assert b"hunter2" not in body
  assert json.loads(body)["config"]["username"] == "me"

  # settings can't be read or changed without the access code
  assert request(f"{base}/api/upload")[0] == 403
  assert request(f"{base}/api/upload", method="POST")[0] == 403


def test_wheel_button_setting(viewer):
  _, _, _, base = viewer
  status, _, body = request(f"{base}/api/settings", headers=AUTH)
  assert status == 200 and json.loads(body) == {"wheel_button": "lkas"}

  def post(value):
    req = urllib.request.Request(f"{base}/api/settings", data=json.dumps({"wheel_button": value}).encode(), method="POST",
                                 headers={**AUTH, "Content-Type": "application/json"})
    try:
      with urllib.request.urlopen(req) as r:
        return r.status
    except urllib.error.HTTPError as e:
      return e.code

  assert post("off") == 200
  assert json.loads(request(f"{base}/api/settings", headers=AUTH)[2])["wheel_button"] == "off"
  assert post("horn") == 400
  assert request(f"{base}/api/settings", method="POST")[0] == 403


def test_missing_video_is_404(viewer):
  _, clip, _, base = viewer
  assert request(f"{base}/api/clips/{clip.id}/prepare/road", method="POST", headers=AUTH)[0] == 404
  assert request(f"{base}/video/{clip.id}/nope.mp4", headers=AUTH)[0] == 404


@requires_hevc_encoder
def test_prepare_download_and_range(viewer):
  v, clip, log_root, base = viewer
  for n in range(2, 7):
    make_hevc(str(log_root / f"{ROUTE}--{n}" / "fcamera.hevc"), 60, shade=n * 40)

  status, _, body = request(f"{base}/api/clips/{clip.id}/prepare/road", method="POST", headers=AUTH)
  assert status == 200
  info = json.loads(body)
  size = info["bytes"]
  assert size > 0

  status, headers, body = request(f"{base}{info['url']}?dl=1&name=dashcam_2026-10-28_0900_road", headers=AUTH)
  assert status == 200
  assert len(body) == size
  assert headers["Content-Type"] == "video/mp4"
  assert headers["Content-Disposition"] == 'attachment; filename="dashcam_2026-10-28_0900_road.mp4"'

  status, headers, body = request(f"{base}{info['url']}", headers={**AUTH, "Range": "bytes=0-99"})
  assert status == 206
  assert headers["Content-Range"] == f"bytes 0-99/{size}"
  assert len(body) == 100

  status, _, body = request(f"{base}{info['url']}", headers={**AUTH, "Range": "bytes=-10"})
  assert status == 206 and len(body) == 10

  assert request(f"{base}/thumb/{clip.id}.jpg", headers=AUTH)[0] == 200

  # deleting the clip removes its cached export
  request(f"{base}/api/clips/{clip.id}/delete", method="POST", headers=AUTH)
  assert not any(f.startswith(clip.id) for f in os.listdir(v.store.exports_dir))


@requires_hevc_encoder
def test_export_cache_is_bounded(viewer):
  v, clip, log_root, _ = viewer
  for n in range(2, 7):
    make_hevc(str(log_root / f"{ROUTE}--{n}" / "fcamera.hevc"), 30)
  for i in range(5):
    c = make_clip(ROUTE, 5, float(i), now=float(i))
    v.store.save(c)
    assert v.export(c, "road")
  assert len(os.listdir(v.store.exports_dir)) <= 3
