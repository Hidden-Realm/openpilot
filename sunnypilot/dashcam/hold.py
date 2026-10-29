"""
Steering wheel button that saves a clip when held.

The button is chosen in the phone viewer and stored next to the clips, so it
stays on the device. LKAS is the default: holding it saves a clip and a short
press still toggles MADS steering (on release, see ButtonTap). Distance only
works with stock ACC, because with openpilot longitudinal a distance hold
already switches experimental mode.
"""
import json
import os

HOLD_SECONDS = 1.0

SETTINGS_FILE = "settings.json"
WHEEL_BUTTONS = ("lkas", "distance", "off")
DEFAULT_WHEEL_BUTTON = "lkas"


class ButtonHold:
  """Fires once when a button has been held down for HOLD_SECONDS. Releasing re-arms it."""

  def __init__(self, hold_seconds: float = HOLD_SECONDS):
    self.hold_seconds = hold_seconds
    self._pressed_at: float | None = None
    self._fired = False

  def update(self, pressed: bool | None, now: float) -> bool:
    """`pressed` is True/False on a press/release event for this button, None when nothing changed."""
    if pressed is True and self._pressed_at is None:
      self._pressed_at = now
      self._fired = False
    elif pressed is False:
      self._pressed_at = None

    if self._pressed_at is not None and not self._fired and now - self._pressed_at >= self.hold_seconds:
      self._fired = True
      return True
    return False

  def reset(self) -> None:
    self._pressed_at = None
    self._fired = False


class ButtonTap:
  """Fires on release of a press shorter than HOLD_SECONDS, so a held press can mean something else."""

  def __init__(self, hold_seconds: float = HOLD_SECONDS):
    self.hold_seconds = hold_seconds
    self._pressed_at: float | None = None

  def update(self, pressed: bool | None, now: float) -> bool:
    if pressed is True:
      self._pressed_at = now
    elif pressed is False and self._pressed_at is not None:
      held = now - self._pressed_at
      self._pressed_at = None
      return held < self.hold_seconds
    return False


def wheel_button(root: str) -> str:
  try:
    with open(os.path.join(root, SETTINGS_FILE)) as f:
      value = json.load(f).get("wheel_button")
  except (OSError, ValueError, AttributeError):
    return DEFAULT_WHEEL_BUTTON
  return value if value in WHEEL_BUTTONS else DEFAULT_WHEEL_BUTTON


def set_wheel_button(root: str, value: str) -> str:
  if value not in WHEEL_BUTTONS:
    raise ValueError(f"unknown button {value!r}")
  path = os.path.join(root, SETTINGS_FILE)
  os.makedirs(root, exist_ok=True)
  tmp = path + ".tmp"
  with open(tmp, "w") as f:
    json.dump({"wheel_button": value}, f)
  os.replace(tmp, path)
  return value
