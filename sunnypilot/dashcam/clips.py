"""
Saved dash cam clips.

A clip is a small JSON record pointing at a window of time inside a route: the
minutes before the button press and the minute after. The video itself stays in
the normal segment folders under the log root; the record is what tells the
deleter to keep those segments and tells the viewer what to export.

Records live outside the log root on purpose: the deleter and uploader treat
every folder in the log root as a segment.
"""
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass

SEGMENT_SECONDS = 60
CAMERA_FPS = 20

SECONDS_BEFORE = 180
SECONDS_AFTER = 60

# camera file name -> label shown in the viewer
CAMERAS = {
  "fcamera.hevc": "road",
  "ecamera.hevc": "wide",
  "dcamera.hevc": "driver",
}


def dashcam_root() -> str:
  from openpilot.system.hardware.hw import Paths
  return os.path.join(os.path.dirname(os.path.normpath(Paths.log_root())), "dashcam")


def log_roots() -> list[str]:
  from openpilot.system.hardware.hw import Paths
  return [Paths.log_root(), Paths.log_root_external()]


@dataclass
class Clip:
  id: str
  route: str
  created: float          # wall clock time of the press (may be wrong before the clock syncs)
  trigger_segment: int
  trigger_offset: float   # seconds into trigger_segment
  first_segment: int
  last_segment: int
  start_offset: float     # seconds into first_segment where the clip starts
  duration: float

  @property
  def segments(self) -> list[str]:
    return [f"{self.route}--{n}" for n in range(self.first_segment, self.last_segment + 1)]

  def to_json(self) -> str:
    return json.dumps(asdict(self), indent=2)

  @classmethod
  def from_dict(cls, d: dict) -> "Clip":
    return cls(**{k: d[k] for k in cls.__dataclass_fields__})


def clip_window(trigger_segment: int, trigger_offset: float,
                before: float = SECONDS_BEFORE, after: float = SECONDS_AFTER,
                segment_seconds: float = SEGMENT_SECONDS) -> tuple[int, int, float, float]:
  """Return (first_segment, last_segment, start_offset, duration) for a press at trigger_offset seconds into trigger_segment.

  Assumes earlier segments of the route are full length, which holds unless loggerd force-rotated one.
  A press early in a drive gets less than `before` seconds of history.
  """
  t = trigger_segment * segment_seconds + max(0.0, trigger_offset)
  start = max(0.0, t - before)
  end = t + after
  first = int(start // segment_seconds)
  last = max(first, math.ceil(end / segment_seconds) - 1)
  return first, last, start - first * segment_seconds, end - start


def make_clip(route: str, trigger_segment: int, trigger_offset: float, now: float | None = None) -> Clip:
  first, last, start_offset, duration = clip_window(trigger_segment, trigger_offset)
  return Clip(
    id=f"{route}--{trigger_segment}--{int(trigger_offset * 1000)}",
    route=route,
    created=time.time() if now is None else now,  # noqa: TID251  wall clock, shown to the user as the clip's date
    trigger_segment=trigger_segment,
    trigger_offset=trigger_offset,
    first_segment=first,
    last_segment=last,
    start_offset=start_offset,
    duration=duration,
  )


class ClipStore:
  def __init__(self, root: str):
    self.root = root
    self.clips_dir = os.path.join(root, "clips")
    self.exports_dir = os.path.join(root, "exports")
    self.thumbs_dir = os.path.join(root, "thumbs")

  def _path(self, clip_id: str) -> str:
    return os.path.join(self.clips_dir, f"{clip_id}.json")

  def save(self, clip: Clip) -> None:
    os.makedirs(self.clips_dir, exist_ok=True)
    tmp = self._path(clip.id) + ".tmp"
    with open(tmp, "w") as f:
      f.write(clip.to_json())
    os.replace(tmp, self._path(clip.id))

  def get(self, clip_id: str) -> Clip | None:
    if os.path.basename(clip_id) != clip_id:
      return None
    try:
      with open(self._path(clip_id)) as f:
        return Clip.from_dict(json.load(f))
    except (OSError, ValueError, KeyError, TypeError):
      return None

  def list(self) -> list[Clip]:
    """All saved clips, newest press first."""
    try:
      names = os.listdir(self.clips_dir)
    except OSError:
      return []
    clips = [self.get(n[:-5]) for n in names if n.endswith(".json")]
    return sorted((c for c in clips if c is not None), key=lambda c: c.created, reverse=True)

  def delete(self, clip_id: str) -> None:
    """Forget a clip. Its segments go back to normal rolling deletion unless another clip uses them."""
    for path in (self._path(clip_id), os.path.join(self.thumbs_dir, f"{clip_id}.jpg")):
      try:
        os.remove(path)
      except OSError:
        pass
    self.clear_exports(clip_id)

  def clear_exports(self, clip_id: str | None = None) -> None:
    if clip_id is None:
      shutil.rmtree(self.exports_dir, ignore_errors=True)
      return
    try:
      for name in os.listdir(self.exports_dir):
        if name.startswith(f"{clip_id}."):
          os.remove(os.path.join(self.exports_dir, name))
    except OSError:
      pass

  def saved_segments(self) -> set[str]:
    return {seg for clip in self.list() for seg in clip.segments}

  def oldest_clip_using(self, segment: str) -> Clip | None:
    using = [c for c in self.list() if segment in c.segments]
    return min(using, key=lambda c: c.created) if using else None


def current_segment(log_root: str, route: str) -> int | None:
  """Highest segment number of `route` that loggerd has created so far."""
  try:
    names = os.listdir(log_root)
  except OSError:
    return None
  prefix = f"{route}--"
  nums = [int(n[len(prefix):]) for n in names if n.startswith(prefix) and n[len(prefix):].isdigit()]
  return max(nums) if nums else None


def find_segment_dir(segment: str, roots: list[str]) -> str | None:
  for root in roots:
    path = os.path.join(root, segment)
    if os.path.isdir(path):
      return path
  return None


def clip_cameras(clip: Clip, roots: list[str]) -> list[str]:
  """Camera files that exist for at least one segment of the clip, in CAMERAS order."""
  found = set()
  for seg in clip.segments:
    d = find_segment_dir(seg, roots)
    if d is None:
      continue
    found.update(name for name in CAMERAS if os.path.isfile(os.path.join(d, name)))
  return [name for name in CAMERAS if name in found]


def clip_bytes(clip: Clip, roots: list[str]) -> int:
  total = 0
  for seg in clip.segments:
    d = find_segment_dir(seg, roots)
    if d is None:
      continue
    for name in os.listdir(d):
      try:
        total += os.path.getsize(os.path.join(d, name))
      except OSError:
        pass
  return total
