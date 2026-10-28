import base64
import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from openpilot.sunnypilot.dashcam.clips import ClipStore, make_clip
from openpilot.sunnypilot.dashcam.upload import (STATE_FILE, UploadConfig, Uploader, check_servers, clip_finished, load_config,
                                                pick_server, read_status, remote_name, save_config, update_config)

ROUTE = "0000002a--6c3b9f3e1d"
USER, PASSWORD = "cam", "secret"


class FakeDav:
  """Just enough WebDAV: folders, PUT, HEAD, PROPFIND, basic auth."""

  def __init__(self):
    self.files: dict[str, bytes] = {}
    self.folders: set[str] = {""}
    self.fail_put = False
    self.truncate_put = False
    dav = self

    class Handler(BaseHTTPRequestHandler):
      def log_message(self, *args):
        pass

      def _auth(self):
        want = "Basic " + base64.b64encode(f"{USER}:{PASSWORD}".encode()).decode()
        if self.headers.get("Authorization") != want:
          self.send_response(401)
          self.send_header("Content-Length", "0")
          self.end_headers()
          return False
        return True

      def _path(self):
        return self.path.strip("/")

      def _reply(self, code, length=0):
        self.send_response(code)
        self.send_header("Content-Length", str(length))
        self.end_headers()

      def do_PROPFIND(self):
        if self._auth():
          self._reply(207)

      def do_MKCOL(self):
        if not self._auth():
          return
        p = self._path()
        if p in dav.folders:
          return self._reply(405)
        dav.folders.add(p)
        self._reply(201)

      def do_PUT(self):
        if not self._auth():
          return
        data = self.rfile.read(int(self.headers["Content-Length"]))
        if dav.fail_put:
          return self._reply(507)
        dav.files[self._path()] = data[:len(data) // 2] if dav.truncate_put else data
        self._reply(201)

      def do_HEAD(self):
        if not self._auth():
          return
        p = self._path()
        self._reply(200, len(dav.files[p])) if p in dav.files else self._reply(404)

    self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
    threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

  def close(self):
    self.httpd.shutdown()
    self.httpd.server_close()


@pytest.fixture
def dav():
  d = FakeDav()
  yield d
  d.close()


def closed_port_url():
  with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    return f"http://127.0.0.1:{s.getsockname()[1]}"


def fake_export(clip, cam_file, roots, dst):
  with open(dst, "wb") as f:
    f.write(f"{clip.id}:{cam_file}".encode() * 100)
  return 600


def fake_thumb(clip, roots, dst):
  with open(dst, "wb") as f:
    f.write(b"\xff\xd8jpeg")
  return True


@pytest.fixture
def setup(tmp_path, dav):
  log_root = tmp_path / "realdata"
  for n in range(2, 8):
    (log_root / f"{ROUTE}--{n}").mkdir(parents=True)
    (log_root / f"{ROUTE}--{n}" / "fcamera.hevc").write_bytes(b"x")
    (log_root / f"{ROUTE}--{n}" / "ecamera.hevc").write_bytes(b"x")
  root = str(tmp_path / "dashcam")
  store = ClipStore(root)
  clip = make_clip(ROUTE, 5, 30.0, now=1761661380.0)  # 2025-10-28 14:23:00 UTC
  store.save(clip)
  save_config(root, UploadConfig(enabled=True, home_url=dav.url, username=USER, password=PASSWORD))
  uploader = Uploader(root, [str(log_root)], fake_export, fake_thumb)
  return uploader, store, clip, str(log_root)


class TestConfig:
  def test_blank_password_keeps_saved_one(self, tmp_path):
    root = str(tmp_path)
    update_config(root, {"username": "me", "password": "pw1"})
    cfg = update_config(root, {"home_url": " http://10.0.0.5/dav ", "password": ""})
    assert cfg.password == "pw1"
    assert cfg.home_url == "http://10.0.0.5/dav"

  def test_public_view_hides_password(self, tmp_path):
    cfg = update_config(str(tmp_path), {"password": "pw1"})
    pub = cfg.public()
    assert "password" not in pub and pub["password_set"] is True
    assert "pw1" not in json.dumps(pub)

  def test_ignores_unknown_cameras_and_keys(self, tmp_path):
    cfg = update_config(str(tmp_path), {"cameras": ["road", "../etc"], "bogus": 1})
    assert cfg.cameras == ["road"]
    assert load_config(str(tmp_path)).cameras == ["road"]

  def test_defaults_without_a_file(self, tmp_path):
    assert load_config(str(tmp_path)) == UploadConfig()


class TestServerChoice:
  def test_prefers_home(self, dav):
    picked = pick_server(UploadConfig(home_url=dav.url, away_url=closed_port_url(), username=USER, password=PASSWORD))
    assert picked is not None and picked[0] == "home"

  def test_falls_back_to_away(self, dav):
    picked = pick_server(UploadConfig(home_url=closed_port_url(), away_url=dav.url, username=USER, password=PASSWORD))
    assert picked is not None and picked[0] == "away"

  def test_none_reachable(self):
    assert pick_server(UploadConfig(home_url=closed_port_url())) is None

  def test_check_reports_reasons(self, dav):
    results = check_servers(UploadConfig(home_url=dav.url, away_url=closed_port_url(), username=USER, password="wrong"))
    assert results == {"home": "wrong username or password", "away": "can't reach the server"}


class TestUploader:
  def test_uploads_and_frees_the_clip(self, setup, dav):
    uploader, store, clip, log_root = setup
    assert uploader.run_once(current_route=None, log_root=log_root) == "uploaded"

    prefix = "dashcam/2025-10-28/dashcam_2025-10-28_142300UTC"
    assert set(dav.files) == {f"{prefix}_road.mp4", f"{prefix}_wide.mp4", f"{prefix}.jpg"}
    assert dav.files[f"{prefix}_road.mp4"].startswith(f"{clip.id}:fcamera.hevc".encode())
    assert store.list() == []  # segments go back to normal rolling deletion
    assert read_status(uploader.root)["pending"] == 0
    assert not os.listdir(uploader.tmp_dir)

  def test_keep_on_device_marks_done(self, setup, dav):
    uploader, store, clip, log_root = setup
    update_config(uploader.root, {"delete_after_upload": False})
    uploader.run_once(None, log_root)
    assert [c.id for c in store.list()] == [clip.id]
    assert uploader.run_once(None, log_root) == "idle"  # not uploaded twice

  def test_waits_for_the_minute_after(self, setup, dav):
    uploader, store, clip, log_root = setup
    # still driving, and segment 6 (the minute after the press) is the newest
    seg7 = os.path.join(log_root, f"{ROUTE}--7")
    for f in os.listdir(seg7):
      os.remove(os.path.join(seg7, f))
    os.rmdir(seg7)
    assert not clip_finished(clip, ROUTE, log_root)
    assert uploader.run_once(ROUTE, log_root) == "idle"
    assert dav.files == {}

    os.makedirs(os.path.join(log_root, f"{ROUTE}--7"))
    assert clip_finished(clip, ROUTE, log_root)

  def test_failed_upload_keeps_clip_and_resumes(self, setup, dav):
    uploader, store, clip, log_root = setup
    dav.fail_put = True
    assert uploader.run_once(None, log_root) == "error"
    assert [c.id for c in store.list()] == [clip.id]
    assert "refused" in read_status(uploader.root)["message"]

    dav.fail_put = False
    assert uploader.run_once(None, log_root) == "uploaded"
    assert store.list() == []

  def test_incomplete_upload_is_not_trusted(self, setup, dav):
    uploader, store, clip, log_root = setup
    dav.truncate_put = True
    assert uploader.run_once(None, log_root) == "error"
    assert "didn't verify" in read_status(uploader.root)["message"]
    assert [c.id for c in store.list()] == [clip.id]

  def test_resumes_after_partial_upload(self, setup, dav):
    uploader, store, clip, log_root = setup
    with open(os.path.join(uploader.root, STATE_FILE), "w") as f:
      json.dump({clip.id: ["road"]}, f)
    uploader.run_once(None, log_root)
    assert not any(name.endswith("_road.mp4") for name in dav.files)
    assert any(name.endswith("_wide.mp4") for name in dav.files)

  def test_off_and_unreachable(self, setup):
    uploader, store, clip, log_root = setup
    update_config(uploader.root, {"enabled": False})
    assert uploader.run_once(None, log_root) == "off"
    update_config(uploader.root, {"enabled": True, "home_url": closed_port_url()})
    assert uploader.run_once(None, log_root) == "unreachable"
    assert [c.id for c in store.list()] == [clip.id]


def test_remote_name_without_clock():
  assert remote_name(make_clip(ROUTE, 5, 0.0, now=0.0)).startswith("unknown-time/")
