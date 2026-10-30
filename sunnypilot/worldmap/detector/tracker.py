"""
Keeping detections as tracked objects between runs.

Detections come a few times a second and from two cameras, so each object keeps a small state in the car frame:
position and velocity over the ground. Between runs, every track is moved by the car's own motion (it drove
forward and turned), then nudged toward its new detection with fixed gains. Detections are matched to tracks of
the same kind by distance; tracks unseen for a while are dropped.
"""
import math
from dataclasses import dataclass

POS_GAIN = 0.6
VEL_GAIN = 0.3
MAX_AGE_UNSEEN = 1.5      # seconds
STATIC_KINDS = ("trafficLight", "stopSign")


@dataclass
class Track:
  id: int
  kind: str
  x: float
  y: float
  vx: float = 0.0          # m/s over the ground, car frame axes
  vy: float = 0.0
  width: float = 0.5
  score: float = 0.0
  camera: str = "road"
  light: str = "none"
  born: float = 0.0
  seen: float = 0.0
  hits: int = 1


class Tracker:
  def __init__(self):
    self.tracks: list[Track] = []
    self._next_id = 1
    self.t: float | None = None

  def predict(self, t: float, v_ego: float, yaw_rate: float) -> None:
    """Carry every track through the car's motion since the last call. yaw_rate positive turning right."""
    if self.t is None:
      self.t = t
      return
    dt = min(max(t - self.t, 0.0), 1.0)
    self.t = t
    if dt == 0.0:
      return
    dpsi = yaw_rate * dt                  # the car turned right by this much
    c, s = math.cos(dpsi), math.sin(dpsi)
    for tr in self.tracks:
      # the object moves over the ground; then express it in the car's new pose
      x = tr.x + tr.vx * dt - v_ego * dt
      y = tr.y + tr.vy * dt
      tr.x, tr.y = c * x + s * y, -s * x + c * y
      tr.vx, tr.vy = c * tr.vx + s * tr.vy, -s * tr.vx + c * tr.vy
    self.tracks = [tr for tr in self.tracks if t - tr.seen < MAX_AGE_UNSEEN and tr.x > -10.0]

  def update(self, t: float, dets: list[dict]) -> None:
    """dets: dicts with kind, x, y, width, score, camera, light."""
    used: set[int] = set()
    for d in sorted(dets, key=lambda d: -d["score"]):
      best, best_cost = None, None
      gate = 2.0 + 0.1 * abs(d["x"])
      for i, tr in enumerate(self.tracks):
        if i in used or tr.kind != d["kind"]:
          continue
        cost = math.hypot(tr.x - d["x"], tr.y - d["y"])
        if cost < gate and (best_cost is None or cost < best_cost):
          best, best_cost = i, cost
      if best is None:
        self.tracks.append(Track(self._next_id, d["kind"], d["x"], d["y"], width=d["width"], score=d["score"],
                                 camera=d["camera"], light=d.get("light", "none"), born=t, seen=t))
        used.add(len(self.tracks) - 1)
        self._next_id += 1
        continue
      used.add(best)
      tr = self.tracks[best]
      dt = max(t - tr.seen, 1e-3)
      ex, ey = d["x"] - tr.x, d["y"] - tr.y
      tr.x += POS_GAIN * ex
      tr.y += POS_GAIN * ey
      if tr.kind not in STATIC_KINDS:
        tr.vx += VEL_GAIN * ex / dt
        tr.vy += VEL_GAIN * ey / dt
        speed = math.hypot(tr.vx, tr.vy)
        if speed > 40.0:  # nothing we track moves this fast relative to the ground
          tr.vx, tr.vy = tr.vx * 40.0 / speed, tr.vy * 40.0 / speed
      tr.width = 0.7 * tr.width + 0.3 * d["width"]
      tr.score = 0.7 * tr.score + 0.3 * d["score"]
      tr.camera = d["camera"]
      if d.get("light", "none") != "none":
        tr.light = d["light"]
      tr.seen = t
      tr.hits += 1

  def confirmed(self) -> list[Track]:
    """Tracks seen at least twice, or once with a high score."""
    return [tr for tr in self.tracks if tr.hits >= 2 or tr.score > 0.6]
