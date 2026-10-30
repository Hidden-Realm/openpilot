"""
When the map may help steer, and by how much.

Qualification (unlocks the "map steering" switch in device settings): the map path and the driving model's
path are compared on every mapped stretch where the model is sure of its lane. A stretch agrees when the paths
line up within AGREE_M and their curves match within AGREE_LAT_ACCEL. Distances are added up with a slow fade
(older kilometers count less), and the map qualifies once enough of that distance agrees. Agreement dropping
later takes the qualification away again.

Weight (only with the switch on and the map qualified): how much of the map's curvature the steering may use
right now. It needs a well-trodden stretch, a good fit to the live view, recent agreement with the model, normal
road speeds and no lane change. It grows when the model is unsure of the lines (faded or unmarked roads), and it
is rate limited both ways. controlsd caps what it can change on top of this (see map_steer.py).
"""
import math
from dataclasses import dataclass

QUALIFY_KM = 25.0
QUALIFY_RATIO = 0.90
UNQUALIFY_RATIO = 0.80
FADE_KM = 150.0            # compared distance fades with this length scale
AGREE_M = 0.5              # paths within this many meters count as agreeing
AGREE_LAT_ACCEL = 0.3      # and their curves within this much sideways acceleration (m/s^2)
MIN_MAP_CONF = 0.7
MIN_SPEED = 8.0            # m/s, about 18 mph
MAX_SPEED = 35.0
MAX_WEIGHT = 0.5
AGREE_HOLD_S = 2.0         # live agreement must have held this long
RAMP_UP = 0.25             # weight per second
RAMP_DOWN = 1.0


@dataclass
class QualifyStats:
  compared_km: float = 0.0
  agreed_km: float = 0.0
  qualified: bool = False

  @property
  def ratio(self) -> float:
    return self.agreed_km / self.compared_km if self.compared_km > 0 else 0.0

  def add(self, dist_m: float, agreed: bool) -> bool:
    """Add a compared stretch. Returns True when the qualified state changed."""
    km = dist_m / 1000.0
    fade = math.exp(-km / FADE_KM)
    self.compared_km = self.compared_km * fade + km
    self.agreed_km = self.agreed_km * fade + (km if agreed else 0.0)
    before = self.qualified
    if not self.qualified and self.compared_km >= QUALIFY_KM and self.ratio >= QUALIFY_RATIO:
      self.qualified = True
    elif self.qualified and self.compared_km >= QUALIFY_KM and self.ratio < UNQUALIFY_RATIO:
      self.qualified = False
    return before != self.qualified

  def km_to_go(self) -> float:
    return max(QUALIFY_KM - self.compared_km, 0.0)

  def to_dict(self) -> dict:
    return {"compared_km": round(self.compared_km, 3), "agreed_km": round(self.agreed_km, 3), "qualified": self.qualified}

  @classmethod
  def from_dict(cls, d: dict | None) -> "QualifyStats":
    if not isinstance(d, dict):
      return cls()
    return cls(float(d.get("compared_km", 0.0)), float(d.get("agreed_km", 0.0)), bool(d.get("qualified", False)))


class SteerWeight:
  def __init__(self):
    self.weight = 0.0
    self.agree_since: float | None = None
    self.reason = "off"

  def update(self, t: float, dt: float, *, enabled: bool, qualified: bool, map_valid: bool, map_conf: float,
             lateral_error: float | None, align_ok: bool, pose_ok: bool, v_ego: float, lane_change: bool,
             model_lane_conf: float) -> float:
    if lateral_error is not None and lateral_error < AGREE_M * 1.5:
      if self.agree_since is None:
        self.agree_since = t
    else:
      self.agree_since = None

    target = 0.0
    if not enabled:
      self.reason = "off"
    elif not qualified:
      self.reason = "not qualified"
    elif not (map_valid and map_conf >= MIN_MAP_CONF):
      self.reason = "road not mapped enough"
    elif not (pose_ok and align_ok):
      self.reason = "position not locked to map"
    elif not (MIN_SPEED <= v_ego <= MAX_SPEED):
      self.reason = "speed"
    elif lane_change:
      self.reason = "lane change"
    elif self.agree_since is None or t - self.agree_since < AGREE_HOLD_S:
      self.reason = "waiting for agreement"
    else:
      self.reason = "active"
      uncertainty = 1.0 - min(max(model_lane_conf, 0.0), 1.0)
      target = MAX_WEIGHT * min(map_conf, 1.0) * (0.4 + 0.6 * uncertainty)

    step = (RAMP_UP if target > self.weight else RAMP_DOWN) * dt
    self.weight += max(min(target - self.weight, step), -step)
    self.weight = max(self.weight, 0.0)
    return self.weight
