import time
from typing import Optional

import numpy as np
import pyray as rl
import qrcode

from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.ui.mici.widgets.button import BigButton
from openpilot.sunnypilot.dashcam.clips import ClipStore, dashcam_root
from openpilot.sunnypilot.dashcam.viewer_url import PORT, TETHERING_IP, get_token, viewer_url, wifi_qr_payload, wlan_ip
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.wifi_manager import WifiManager
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.system.ui.widgets.nav_widget import NavWidget

REFRESH_SECONDS = 2.0


def _qr_texture(payload: str) -> Optional[rl.Texture]:
  try:
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=0)
    qr.add_data(payload)
    qr.make(fit=True)
    pil_img = qr.make_image(fill_color="white", back_color="black").convert('RGBA')
    img_array = np.array(pil_img, dtype=np.uint8)

    rl_image = rl.Image()
    rl_image.data = rl.ffi.cast("void *", img_array.ctypes.data)
    rl_image.width = pil_img.width
    rl_image.height = pil_img.height
    rl_image.mipmaps = 1
    rl_image.format = rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
    return rl.load_texture_from_image(rl_image)
  except Exception as e:
    cloudlog.warning(f"dashcam QR code generation failed: {e}")
    return None


class DashcamDialog(NavWidget):
  """How to get saved clips onto a phone. Tap to go to the next step.

  On the device's hotspot: a QR the iPhone camera uses to join the hotspot,
  then the QR that opens the viewer, then the same address and code as text
  for typing in or bookmarking. On a shared Wi-Fi network the join step is skipped.
  Other Wi-Fi viewers (the 3D world) reuse it with their own title, port and summary.
  """
  TITLE = "dash cam"
  PORT = PORT
  OPEN_TEXT = "scan to open clips"

  def __init__(self, wifi_manager: WifiManager):
    super().__init__()
    self._wifi_manager = wifi_manager
    self._store = ClipStore(dashcam_root())
    self._token = get_token(dashcam_root())

    self._qr_texture: rl.Texture | None = None
    self._qr_payload: str | None = None
    self._step = 0
    self._last_refresh = float("-inf")

    self._title = UnifiedLabel(self.TITLE, font_size=48, font_weight=FontWeight.BOLD, line_height=0.8)
    self._body = UnifiedLabel("", font_size=32, font_weight=FontWeight.MEDIUM, text_color=rl.Color(255, 255, 255, int(255 * 0.65)))
    self.set_click_callback(self._next_step)

  def show_event(self):
    super().show_event()
    self._step = 0
    self._last_refresh = float("-inf")

  def _next_step(self):
    self._step += 1
    self._last_refresh = float("-inf")

  def _set_qr(self, payload: str | None) -> None:
    if payload == self._qr_payload:
      return
    if self._qr_texture and self._qr_texture.id != 0:
      rl.unload_texture(self._qr_texture)
    self._qr_texture = _qr_texture(payload) if payload else None
    self._qr_payload = payload

  def _summary(self) -> str:
    clips = len(self._store.list())
    return f"{clips} saved clip{'s' if clips != 1 else ''}"

  def _url(self, ip: str) -> str:
    return viewer_url(ip, self._token)

  def _refresh(self) -> None:
    count = self._summary()
    ip = wlan_ip()

    if ip is None:
      self._set_qr(None)
      self._body.set_text(f"{count}\n\nturn on tethering in network settings, or join the same Wi-Fi as your phone")
    else:
      steps = ["join", "open", "type"] if ip == TETHERING_IP else ["open", "type"]
      step = steps[self._step % len(steps)]
      n = f"{steps.index(step) + 1}. "
      if step == "join":
        ssid = self._wifi_manager._tethering_ssid
        self._set_qr(wifi_qr_payload(ssid, self._wifi_manager.tethering_password))
        self._body.set_text(f"{n}scan to join\n{ssid}\n\ntap for next step")
      elif step == "open":
        self._set_qr(self._url(ip))
        where = "" if ip == TETHERING_IP else "\n(phone on same Wi-Fi)"
        self._body.set_text(f"{n}{self.OPEN_TEXT}{where}\n\n{count}\ntap for next step")
      else:
        self._set_qr(None)
        self._body.set_text(f"{n}or in Safari go to\n{ip}:{self.PORT}\nand enter code  {self._token}")

  def _render(self, rect: rl.Rectangle):
    now = time.monotonic()
    if now - self._last_refresh >= REFRESH_SECONDS:
      self._refresh()
      self._last_refresh = now

    label_x = self._rect.x + 16
    if self._qr_texture:
      scale = self._rect.height / self._qr_texture.height
      rl.draw_texture_ex(self._qr_texture, rl.Vector2(round(self._rect.x + 8), round(self._rect.y)), 0.0, scale, rl.WHITE)
      label_x = self._rect.x + 8 + self._rect.height + 24

    max_width = int(self._rect.x + self._rect.width - label_x - 8)
    self._title.set_max_width(max_width)
    self._title.set_position(label_x, self._rect.y + 16)
    self._title.render()
    self._body.set_max_width(max_width)
    self._body.set_position(label_x, self._rect.y + 72)
    self._body.render()

  def __del__(self):
    if self._qr_texture and self._qr_texture.id != 0:
      rl.unload_texture(self._qr_texture)


class DashcamBigButton(BigButton):
  def __init__(self, wifi_manager: WifiManager):
    super().__init__("dash cam", "", gui_app.texture("icons_mici/settings/device_icon.png", 72, 58))
    self._store = ClipStore(dashcam_root())
    self._dialog = DashcamDialog(wifi_manager)
    self._last_refresh = float("-inf")
    self.set_click_callback(lambda: gui_app.push_widget(self._dialog))

  def _get_label_font_size(self):
    return 64

  def _update_state(self):
    super()._update_state()
    now = time.monotonic()
    if now - self._last_refresh >= REFRESH_SECONDS:
      clips = len(self._store.list())
      self.set_value(f"{clips} saved clip{'s' if clips != 1 else ''}")
      self._last_refresh = now
