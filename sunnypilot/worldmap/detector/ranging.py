"""
Where a detection is, in the car frame (x forward, y right, meters).

Things standing on the road (people, bikes, vehicles) are placed where the bottom of their box meets the road:
the ray through that pixel is followed down to the ground, using the device's calibration and its height above
the road. Signs and lights are not on the ground, so their distance comes from their apparent size (a stop sign
is about 0.75 m across, a traffic light head about 0.3 m wide).

The wide camera is a fisheye that openpilot treats as a pinhole with a short focal length, as its own models
do. That holds near the middle of the image; distances to things at the far edges are rougher.
"""
import math

import numpy as np

# openpilot's view frame (x right, y down, z forward) from its device frame (x forward, y right, z down)
VIEW_FROM_DEVICE = np.array([[0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0],
                             [1.0, 0.0, 0.0]])

REAL_WIDTH = {"stopSign": 0.75, "trafficLight": 0.33, "person": 0.5, "bicycle": 0.6, "car": 1.8, "motorcycle": 0.8,
              "bus": 2.5, "truck": 2.5}
MAX_RANGE = 120.0


def rot_from_euler(rpy) -> np.ndarray:
  """Same convention as openpilot's euler2rot: R = Rz(yaw) Ry(pitch) Rx(roll)."""
  r, p, y = (float(v) for v in rpy)
  cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
  rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
  ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
  rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
  return rz @ ry @ rx


class CameraGeometry:
  def __init__(self, width: int, height: int, focal: float, rpy_calib, height_m: float, wide_from_device_euler=None):
    self.width, self.height, self.focal = width, height, focal
    self.cam_height = max(float(height_m), 0.5)
    K = np.array([[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]])
    device_from_calib = rot_from_euler(rpy_calib)
    if wide_from_device_euler is not None:
      device_from_calib = rot_from_euler(wide_from_device_euler) @ device_from_calib
    self.camera_from_calib = K @ VIEW_FROM_DEVICE @ device_from_calib
    self.calib_from_camera = np.linalg.inv(self.camera_from_calib)

  def ray(self, u: float, v: float) -> np.ndarray:
    """Direction in the calibrated car frame (x forward, y right, z down) through a pixel."""
    d = self.calib_from_camera @ np.array([u, v, 1.0])
    return d / np.linalg.norm(d)

  def ground_point(self, u: float, v: float) -> tuple[float, float] | None:
    d = self.ray(u, v)
    if d[2] <= 1e-3:
      return None  # at or above the horizon: never meets the road
    t = self.cam_height / d[2]
    x, y = t * d[0], t * d[1]
    if not (0.5 < x < MAX_RANGE):
      return None
    return float(x), float(y)

  def sized_point(self, u: float, v: float, pixel_width: float, real_width: float) -> tuple[float, float] | None:
    if pixel_width < 2:
      return None
    dist = self.focal * real_width / pixel_width
    d = self.ray(u, v)
    if d[0] <= 0.1:
      return None
    t = dist / math.hypot(d[0], d[1]) if math.hypot(d[0], d[1]) > 1e-6 else dist
    x, y = t * d[0], t * d[1]
    if not (0.5 < x < MAX_RANGE):
      return None
    return float(x), float(y)

  def locate(self, kind: str, box: np.ndarray) -> tuple[float, float, float] | None:
    """(x, y, width) in meters for a box in camera pixels."""
    x0, y0, x1, y1 = (float(v) for v in box)
    u = (x0 + x1) / 2
    pw = max(x1 - x0, 1.0)
    if kind in ("stopSign", "trafficLight"):
      p = self.sized_point(u, (y0 + y1) / 2, pw, REAL_WIDTH[kind])
      return None if p is None else (p[0], p[1], REAL_WIDTH[kind])
    if y1 >= self.height - 2:
      return None  # cut off by the bottom of the frame: its foot point isn't visible
    p = self.ground_point(u, y1)
    if p is None:
      return None
    width = pw * p[0] / self.focal
    return p[0], p[1], width
