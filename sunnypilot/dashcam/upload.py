"""
Uploading saved clips to the user's own server over WebDAV.

WebDAV is plain HTTP: MKCOL makes a folder and PUT uploads a file. Nextcloud,
Synology, TrueNAS and most home-cloud software accept it as-is, and a custom
site only needs to accept a PUT.

Two server addresses are configured. The home address (on the LAN) is tried
first, so uploads stay on the home network when the device is on home Wi-Fi.
Otherwise the away address (port-forwarded, reachable from the internet, e.g.
over a phone hotspot) is used.
"""
import json
import os
import time
import urllib.parse
from dataclasses import asdict, dataclass, field, fields

import requests

from openpilot.sunnypilot.dashcam.clips import CAMERAS, Clip, ClipStore, clip_cameras, current_segment

CONFIG_FILE = "upload.json"
STATE_FILE = "upload_state.json"
STATUS_FILE = "upload_status.json"

CONNECT_TIMEOUT = 5
READ_TIMEOUT = 120


@dataclass
class UploadConfig:
  enabled: bool = False
  home_url: str = ""          # e.g. http://192.168.1.50:8080/remote.php/dav/files/me
  away_url: str = ""          # e.g. https://myserver.example.com:8443/remote.php/dav/files/me
  username: str = ""
  password: str = ""
  folder: str = "dashcam"
  cameras: list[str] = field(default_factory=lambda: ["road", "wide"])
  delete_after_upload: bool = True
  insecure_tls: bool = False  # accept a self-signed certificate

  def public(self) -> dict:
    d = asdict(self)
    d.pop("password")
    d["password_set"] = bool(self.password)
    return d


def _read_json(path: str, default):
  try:
    with open(path) as f:
      return json.load(f)
  except (OSError, ValueError):
    return default


def _write_json(path: str, data) -> None:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  tmp = path + ".tmp"
  with open(tmp, "w") as f:
    json.dump(data, f, indent=2)
  os.replace(tmp, path)


def load_config(root: str) -> UploadConfig:
  data = _read_json(os.path.join(root, CONFIG_FILE), {})
  known = {f.name for f in fields(UploadConfig)}
  try:
    return UploadConfig(**{k: v for k, v in data.items() if k in known})
  except TypeError:
    return UploadConfig()


def save_config(root: str, cfg: UploadConfig) -> None:
  _write_json(os.path.join(root, CONFIG_FILE), asdict(cfg))
  try:
    os.chmod(os.path.join(root, CONFIG_FILE), 0o600)  # holds the server password
  except OSError:
    pass


def update_config(root: str, changes: dict) -> UploadConfig:
  """Apply settings from the viewer. A blank password keeps the saved one."""
  cfg = load_config(root)
  for key in ("home_url", "away_url", "username", "folder"):
    if isinstance(changes.get(key), str):
      setattr(cfg, key, changes[key].strip())
  for key in ("enabled", "delete_after_upload", "insecure_tls"):
    if key in changes:
      setattr(cfg, key, bool(changes[key]))
  if isinstance(changes.get("cameras"), list):
    cfg.cameras = [c for c in changes["cameras"] if c in CAMERAS.values()]
  if changes.get("password"):
    cfg.password = changes["password"]
  cfg.folder = cfg.folder.strip("/") or "dashcam"
  save_config(root, cfg)
  return cfg


def read_status(root: str) -> dict:
  return _read_json(os.path.join(root, STATUS_FILE), {})


def write_status(root: str, **status) -> None:
  _write_json(os.path.join(root, STATUS_FILE), {**read_status(root), **status, "updated": time.time()})  # noqa: TID251


class UploadError(Exception):
  pass


class WebDav:
  def __init__(self, base_url: str, username: str, password: str, verify_tls: bool = True):
    self.base_url = base_url.rstrip("/")
    self.session = requests.Session()
    if username:
      self.session.auth = (username, password)
    self.session.verify = verify_tls

  def url(self, path: str) -> str:
    return f"{self.base_url}/{urllib.parse.quote(path.strip('/'))}"

  def check(self) -> None:
    """Raise UploadError with a readable reason if the server can't be used."""
    try:
      r = self.session.request("PROPFIND", self.base_url + "/", headers={"Depth": "0"}, timeout=(CONNECT_TIMEOUT, 15))
    except requests.exceptions.SSLError as e:
      raise UploadError("certificate not trusted (self-signed? tick 'accept self-signed certificate')") from e
    except requests.exceptions.RequestException as e:
      raise UploadError("can't reach the server") from e
    if r.status_code in (401, 403):
      raise UploadError("wrong username or password")
    if r.status_code == 404:
      raise UploadError("address not found on the server (check the path)")
    if r.status_code >= 400 and r.status_code not in (405, 501):
      raise UploadError(f"server answered {r.status_code}")

  def ensure_folder(self, path: str) -> None:
    parts = [p for p in path.strip("/").split("/") if p]
    for i in range(1, len(parts) + 1):
      sub = "/".join(parts[:i])
      try:
        r = self.session.request("MKCOL", self.url(sub) + "/", timeout=(CONNECT_TIMEOUT, 30))
      except requests.exceptions.RequestException as e:
        raise UploadError(f"couldn't create folder {sub}") from e
      # 201 created, 405 already exists; some servers answer 301/409 for existing folders
      if r.status_code not in (200, 201, 204, 301, 405, 409):
        raise UploadError(f"couldn't create folder {sub} ({r.status_code})")

  def put(self, path: str, local_path: str) -> None:
    size = os.path.getsize(local_path)
    try:
      with open(local_path, "rb") as f:
        r = self.session.put(self.url(path), data=f, headers={"Content-Length": str(size)},
                             timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    except requests.exceptions.RequestException as e:
      raise UploadError(f"upload of {os.path.basename(path)} interrupted") from e
    if r.status_code not in (200, 201, 204):
      raise UploadError(f"server refused {os.path.basename(path)} ({r.status_code})")

    # confirm the whole file arrived before the device lets go of its copy
    try:
      h = self.session.head(self.url(path), timeout=(CONNECT_TIMEOUT, 30))
    except requests.exceptions.RequestException as e:
      raise UploadError("couldn't confirm the upload") from e
    remote = h.headers.get("Content-Length")
    if h.status_code >= 400 or (remote is not None and int(remote) != size):
      raise UploadError(f"upload of {os.path.basename(path)} didn't verify")


def pick_server(cfg: UploadConfig) -> tuple[str, WebDav] | None:
  """The first configured address that answers: home, then away."""
  for name, url in (("home", cfg.home_url), ("away", cfg.away_url)):
    if not url:
      continue
    dav = WebDav(url, cfg.username, cfg.password, verify_tls=not cfg.insecure_tls)
    try:
      dav.check()
      return name, dav
    except UploadError:
      continue
  return None


def check_servers(cfg: UploadConfig) -> dict:
  results = {}
  for name, url in (("home", cfg.home_url), ("away", cfg.away_url)):
    if not url:
      continue
    try:
      WebDav(url, cfg.username, cfg.password, verify_tls=not cfg.insecure_tls).check()
      results[name] = "ok"
    except UploadError as e:
      results[name] = str(e)
  return results


def clip_finished(clip: Clip, current_route: str | None, log_root: str) -> bool:
  """True once the minute after the press has been fully recorded, or the drive ended."""
  if clip.route != current_route:
    return True
  seg = current_segment(log_root, clip.route)
  return seg is not None and seg > clip.last_segment


def remote_name(clip: Clip) -> str:
  t = time.gmtime(clip.created)
  if t.tm_year < 2020:  # clock wasn't synced yet
    return f"unknown-time/{clip.id}"
  return f"{time.strftime('%Y-%m-%d', t)}/dashcam_{time.strftime('%Y-%m-%d_%H%M%S', t)}UTC"


class Uploader:
  def __init__(self, root: str, roots: list[str], export_fn, thumb_fn):
    self.root = root
    self.roots = roots
    self.store = ClipStore(root)
    self.tmp_dir = os.path.join(root, "upload_tmp")
    self.export_fn = export_fn
    self.thumb_fn = thumb_fn

  def _state(self) -> dict:
    return _read_json(os.path.join(self.root, STATE_FILE), {})

  def _mark(self, clip_id: str, item: str) -> None:
    state = self._state()
    state.setdefault(clip_id, [])
    if item not in state[clip_id]:
      state[clip_id].append(item)
    _write_json(os.path.join(self.root, STATE_FILE), state)

  def _forget(self, clip_id: str) -> None:
    state = self._state()
    if state.pop(clip_id, None) is not None:
      _write_json(os.path.join(self.root, STATE_FILE), state)

  def pending(self, cfg: UploadConfig, current_route: str | None, log_root: str) -> list[Clip]:
    state = self._state()
    clips = [c for c in self.store.list() if clip_finished(c, current_route, log_root)]
    return sorted((c for c in clips if "done" not in state.get(c.id, [])), key=lambda c: c.created)

  def upload_clip(self, clip: Clip, cfg: UploadConfig, dav: WebDav) -> None:
    done = set(self._state().get(clip.id, []))
    name = remote_name(clip)
    folder = f"{cfg.folder}/{os.path.dirname(name)}"
    base = os.path.basename(name)
    dav.ensure_folder(folder)
    os.makedirs(self.tmp_dir, exist_ok=True)

    available = {CAMERAS[f]: f for f in clip_cameras(clip, self.roots)}
    for label in cfg.cameras:
      if label in done or label not in available:
        continue
      tmp = os.path.join(self.tmp_dir, f"{clip.id}.{label}.mp4")
      try:
        if not self.export_fn(clip, available[label], self.roots, tmp):
          continue
        dav.put(f"{folder}/{base}_{label}.mp4", tmp)
      finally:
        if os.path.exists(tmp):
          os.remove(tmp)
      self._mark(clip.id, label)

    if "thumb" not in done:
      tmp = os.path.join(self.tmp_dir, f"{clip.id}.jpg")
      try:
        if self.thumb_fn(clip, self.roots, tmp):
          dav.put(f"{folder}/{base}.jpg", tmp)
      finally:
        if os.path.exists(tmp):
          os.remove(tmp)
      self._mark(clip.id, "thumb")

    if cfg.delete_after_upload:
      self.store.delete(clip.id)
      self._forget(clip.id)
    else:
      self._mark(clip.id, "done")

  def run_once(self, current_route: str | None, log_root: str) -> str:
    """Upload what's waiting. Returns a short status for the viewer."""
    cfg = load_config(self.root)
    if not cfg.enabled or not (cfg.home_url or cfg.away_url):
      return "off"
    pending = self.pending(cfg, current_route, log_root)
    if not pending:
      return "idle"
    picked = pick_server(cfg)
    if picked is None:
      write_status(self.root, state="unreachable", message="can't reach your server", pending=len(pending))
      return "unreachable"
    where, dav = picked
    for i, clip in enumerate(pending):
      write_status(self.root, state="uploading", message=f"uploading via {where} address", pending=len(pending) - i)
      try:
        self.upload_clip(clip, cfg, dav)
      except UploadError as e:
        write_status(self.root, state="error", message=str(e), pending=len(pending) - i)
        return "error"
      write_status(self.root, last_upload=time.time(), uploaded=read_status(self.root).get("uploaded", 0) + 1)  # noqa: TID251
    write_status(self.root, state="idle", message=f"all clips uploaded via {where} address", pending=0)
    return "uploaded"


def main():
  from openpilot.common.params import Params
  from openpilot.common.swaglog import cloudlog
  from openpilot.system.hardware.hw import Paths
  from openpilot.sunnypilot.dashcam.clips import dashcam_root, log_roots
  from openpilot.sunnypilot.dashcam.export import export_clip, make_thumbnail

  try:
    os.nice(10)  # never compete with driving processes
  except (AttributeError, OSError):
    pass

  params = Params()
  uploader = Uploader(dashcam_root(), log_roots(), export_clip, make_thumbnail)
  while True:
    try:
      result = uploader.run_once(params.get("CurrentRoute"), Paths.log_root())
    except Exception:
      cloudlog.exception("dashcam upload failed")
      result = "error"
    time.sleep({"uploaded": 5, "idle": 30, "off": 60}.get(result, 120))


if __name__ == "__main__":
  main()
