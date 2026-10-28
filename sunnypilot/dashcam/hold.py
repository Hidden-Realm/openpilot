HOLD_SECONDS = 1.0


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
