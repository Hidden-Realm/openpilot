"""
Where the car is, for mapping.

A small Kalman filter on (east, north, heading) in a local frame. Wheel speed and yaw rate carry it between
fixes; GPS position and course pull it back. When sunnypilot's own fused location (liveLocationKalman, which
already blends GPS with the IMU) is healthy it is used instead of the raw GPS.

GPS fixes arrive a little late, so a fix is moved forward along the current heading by a fixed latency
before it is applied. Whatever error is left is taken out later by aligning to the map (see align.py).
"""
import math

import numpy as np

from openpilot.sunnypilot.worldmap.geo import LocalFrame, heading_from_bearing, wrap

GPS_LATENCY = 0.1        # seconds
REANCHOR_M = 3000.0      # keep the local frame within this distance of the car
MIN_POS_STD = 0.7        # receivers report optimistic accuracy
GATE_SIGMA2 = 25.0       # Mahalanobis gate for a fix (5 sigma)
MAX_REJECTS = 10         # this many rejected fixes in a row means we are lost: restart from GPS
LLK_FRESH = 1.0          # seconds a fused location keeps raw GPS out


class PoseFilter:
  def __init__(self):
    self.frame: LocalFrame | None = None
    self.frame_id = 0
    self.x = np.zeros(3)                 # east, north, heading
    self.P = np.diag([100.0, 100.0, 10.0])
    self.t: float | None = None
    self.heading_ok = False
    self.rejects = 0
    self.last_fix_t = -1e9
    self.last_llk_t = -1e9
    self.speed = 0.0

  # ---- state
  @property
  def initialized(self) -> bool:
    return self.frame is not None

  @property
  def valid(self) -> bool:
    return self.initialized and self.heading_ok and self.pos_std < 10.0 and self.t is not None and self.t - self.last_fix_t < 5.0

  @property
  def pos_std(self) -> float:
    return float(math.sqrt(max(np.linalg.eigvalsh(self.P[:2, :2]).max(), 0.0)))

  @property
  def heading(self) -> float:
    return float(self.x[2])

  def latlon(self) -> tuple[float, float]:
    assert self.frame is not None
    lat, lon = self.frame.to_ll(self.x[0], self.x[1])
    return float(lat), float(lon)

  # ---- motion
  def predict(self, t: float, v: float, yaw_rate: float) -> None:
    """Advance to time t. yaw_rate is positive turning right (openpilot's z-down axis)."""
    self.speed = v
    if self.t is None:
      self.t = t
      return
    dt = min(max(t - self.t, 0.0), 1.0)
    self.t = t
    if not self.initialized or dt == 0.0:
      return
    e, n, h = self.x
    c, s = math.cos(h), math.sin(h)
    self.x = np.array([e + v * c * dt, n + v * s * dt, wrap(h - yaw_rate * dt)])
    F = np.array([[1.0, 0.0, -v * s * dt],
                  [0.0, 1.0, v * c * dt],
                  [0.0, 0.0, 1.0]])
    q_pos = (0.05 + 0.03 * abs(v)) ** 2 * dt
    q_h = (0.02 ** 2) * dt
    self.P = F @ self.P @ F.T + np.diag([q_pos, q_pos, q_h])
    if abs(self.x[0]) > REANCHOR_M or abs(self.x[1]) > REANCHOR_M:
      self._reanchor()

  def _reanchor(self) -> None:
    lat, lon = self.latlon()
    self.frame = LocalFrame(lat, lon)
    self.frame_id += 1
    self.x[0] = self.x[1] = 0.0

  def _start(self, lat: float, lon: float, std: float, heading: float | None) -> None:
    self.frame = LocalFrame(lat, lon)
    self.frame_id += 1
    self.x = np.array([0.0, 0.0, heading if heading is not None else 0.0])
    self.P = np.diag([std ** 2, std ** 2, 0.05 if heading is not None else 10.0])
    self.heading_ok = heading is not None
    self.rejects = 0

  # ---- measurements
  def _update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray, angle_rows: tuple[int, ...] = ()) -> bool:
    y = z - H @ self.x
    for i in angle_rows:
      y[i] = wrap(y[i])
    S = H @ self.P @ H.T + R
    d2 = float(y @ np.linalg.solve(S, y))
    if d2 > GATE_SIGMA2 * len(z):
      return False
    K = self.P @ H.T @ np.linalg.inv(S)
    self.x = self.x + K @ y
    self.x[2] = wrap(self.x[2])
    self.P = (np.eye(3) - K @ H) @ self.P
    return True

  def _position(self, lat: float, lon: float, std: float, heading: float | None, heading_std: float, t: float) -> None:
    std = max(std, MIN_POS_STD)
    if not self.initialized:
      self._start(lat, lon, std, heading)
      self.last_fix_t = t
      return
    assert self.frame is not None
    e, n = self.frame.to_en(lat, lon)
    if self.heading_ok:
      e = e + self.speed * GPS_LATENCY * math.cos(self.x[2])
      n = n + self.speed * GPS_LATENCY * math.sin(self.x[2])
    ok = self._update(np.array([float(e), float(n)]), np.array([[1.0, 0, 0], [0, 1.0, 0]]), np.eye(2) * std ** 2)
    if ok:
      self.rejects = 0
      self.last_fix_t = t
    else:
      self.rejects += 1
      if self.rejects >= MAX_REJECTS:
        self._start(lat, lon, std, heading if heading is not None else (self.x[2] if self.heading_ok else None))
        self.last_fix_t = t
        return
    if heading is not None:
      if not self.heading_ok:
        self.x[2] = heading
        self.P[2, :] = self.P[:, 2] = 0.0
        self.P[2, 2] = heading_std ** 2
        self.heading_ok = True
      else:
        self._update(np.array([heading]), np.array([[0.0, 0, 1.0]]), np.array([[heading_std ** 2]]), angle_rows=(0,))

  def update_gps(self, t: float, lat: float, lon: float, accuracy: float, bearing_deg: float, bearing_accuracy: float,
                 speed: float, has_fix: bool = True) -> None:
    if not has_fix or t - self.last_llk_t < LLK_FRESH or not (-90 < lat < 90) or (lat == 0.0 and lon == 0.0):
      return
    heading = None
    if speed > 4.0 and 0 < bearing_accuracy < 8.0:
      heading = heading_from_bearing(bearing_deg)
    self._position(lat, lon, accuracy * 1.5, heading, math.radians(max(bearing_accuracy, 0.5)) * 1.5, t)

  def update_llk(self, t: float, lat: float, lon: float, pos_std: float, yaw_ned: float, yaw_std: float) -> None:
    """sunnypilot's fused location. yaw_ned is clockwise from north."""
    self.last_llk_t = t
    heading = math.pi / 2 - yaw_ned if self.speed > 2.0 else None
    self._position(lat, lon, pos_std, heading, max(yaw_std, math.radians(0.5)), t)
