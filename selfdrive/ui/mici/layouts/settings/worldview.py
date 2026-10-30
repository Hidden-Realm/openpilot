import time

from openpilot.common.params import Params
from openpilot.selfdrive.ui.mici.layouts.settings.dashcam import DashcamDialog
from openpilot.selfdrive.ui.mici.widgets.button import BigButton, BigToggle
from openpilot.selfdrive.ui.mici.widgets.dialog import BigDialog
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import MousePos, gui_app
from openpilot.system.ui.lib.wifi_manager import WifiManager
from openpilot.sunnypilot.worldmap.qualify import QUALIFY_KM

WORLD_PORT = 8100  # sunnypilot/worldmap/server.py
REFRESH_SECONDS = 2.0


def map_status(params: Params) -> tuple[bool, float, float]:
  """(qualified, km compared, km still needed) from what worldd saves."""
  stats = params.get("WorldMapStats")
  stats = stats if isinstance(stats, dict) else {}
  compared = float(stats.get("compared_km", 0.0))
  return params.get_bool("WorldMapQualified"), compared, max(QUALIFY_KM - compared, 0.0)


class WorldViewDialog(DashcamDialog):
  """QR codes for the live 3D world viewer: join the hotspot, open the page, or type the address."""
  TITLE = "3D world"
  PORT = WORLD_PORT
  OPEN_TEXT = "scan to open the 3D world"

  def __init__(self, wifi_manager: WifiManager):
    super().__init__(wifi_manager)
    self._params = Params()

  def _summary(self) -> str:
    qualified, compared, _ = map_status(self._params)
    return "map steering ready" if qualified else f"{compared:.1f} km compared"

  def _url(self, ip: str) -> str:
    return f"http://{ip}:{WORLD_PORT}/?k={self._token}"


class WorldViewBigButton(BigButton):
  def __init__(self, wifi_manager: WifiManager):
    super().__init__("3D world", "", gui_app.texture("icons_mici/settings/network/wifi_strength_full.png", 76, 56))
    self._dialog = WorldViewDialog(wifi_manager)
    self._params = Params()
    self._last_refresh = float("-inf")
    self.set_click_callback(lambda: gui_app.push_widget(self._dialog))

  def _get_label_font_size(self):
    return 64

  def _update_state(self):
    super()._update_state()
    now = time.monotonic()
    if now - self._last_refresh >= REFRESH_SECONDS:
      qualified, compared, _ = map_status(self._params)
      self.set_value("map steering ready" if qualified else f"{compared:.1f} km compared")
      self._last_refresh = now


class MapSteerToggle(BigToggle):
  """Device settings switch for map steering. It stays locked until the world map passes its confidence check,
  and it can only be turned on while disengaged (off works any time)."""

  def __init__(self):
    self._params = Params()
    super().__init__("map steering", "", initial_state=self._params.get_bool("WorldMapSteer"))
    self._qualified = False
    self._km_to_go = 0.0
    self._last_refresh = float("-inf")

  def _update_state(self):
    super()._update_state()
    now = time.monotonic()
    if now - self._last_refresh >= REFRESH_SECONDS:
      self._qualified, _, self._km_to_go = map_status(self._params)
      self.set_checked(self._params.get_bool("WorldMapSteer"))
      if self._qualified:
        self.set_value("on: map helps steer" if self._checked else "ready")
      else:
        self.set_value(f"locked: {self._km_to_go:.1f} km to go")
      self._last_refresh = now

  def _handle_mouse_release(self, mouse_pos: MousePos):
    turning_on = not self._checked
    if turning_on and not self._qualified:
      text = f"Map steering unlocks once the map has matched the driving model on about {self._km_to_go:.0f} more km of mapped roads"
      gui_app.push_widget(BigDialog("", text))
      return
    if turning_on and ui_state.engaged:
      gui_app.push_widget(BigDialog("", "Disengage to turn on map steering"))
      return
    super()._handle_mouse_release(mouse_pos)
    self._params.put_bool("WorldMapSteer", self._checked, block=True)
    self._last_refresh = float("-inf")
