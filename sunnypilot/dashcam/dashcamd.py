#!/usr/bin/env python3
"""
Turns bookmarks into saved dash cam clips.

feedbackd publishes userBookmark when the road view is held or swiped, or the
chosen steering wheel button is held (see hold.py). Each one becomes a clip record covering the three
minutes before the press and the minute after; the deleter keeps those
segments and the Wi-Fi viewer exports them.
"""
import time

import cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware.hw import Paths
from openpilot.sunnypilot.dashcam.clips import CAMERA_FPS, ClipStore, current_segment, dashcam_root, make_clip

DEBOUNCE_SECONDS = 3.0


class SegmentClock:
  """Tracks how far into the current segment we are.

  The time since we saw a segment folder appear is accurate to the poll interval.
  For the segment that was already open when we started, fall back to the encoder's
  frame index within the segment.
  """

  def __init__(self):
    self._route: str | None = None
    self._segment: int | None = None
    self._seen_at: float | None = None

  def update(self, route: str, segment: int, now: float) -> None:
    if route != self._route:
      self._route, self._segment, self._seen_at = route, segment, None
    elif segment != self._segment:
      self._segment, self._seen_at = segment, now

  def offset(self, now: float, encoder_frame: int | None) -> float:
    if self._seen_at is not None:
      return now - self._seen_at
    if encoder_frame is not None:
      return encoder_frame / CAMERA_FPS
    return 0.0


def main():
  params = Params()
  store = ClipStore(dashcam_root())
  sm = messaging.SubMaster(['userBookmark', 'roadEncodeIdx'])
  clock = SegmentClock()
  last_save = float("-inf")

  while True:
    sm.update(250)
    now = time.monotonic()

    route = params.get("CurrentRoute")
    segment = current_segment(Paths.log_root(), route) if route else None
    if segment is None:
      continue
    clock.update(route, segment, now)

    if not sm.updated['userBookmark'] or now - last_save < DEBOUNCE_SECONDS:
      continue

    encoder_frame = sm['roadEncodeIdx'].segmentId if sm.recv_frame['roadEncodeIdx'] > 0 else None
    clip = make_clip(route, segment, clock.offset(now, encoder_frame))
    try:
      store.save(clip)
      last_save = now
      cloudlog.warning(f"dashcam: saved clip {clip.id} ({clip.first_segment}-{clip.last_segment})")
    except OSError:
      cloudlog.exception("dashcam: failed to save clip")


if __name__ == "__main__":
  main()
