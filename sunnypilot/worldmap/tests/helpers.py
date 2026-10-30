"""Synthetic drives for the world map tests: GPS fixes and driving-model frames on a known road."""
import math
from types import SimpleNamespace

import numpy as np

LAT0, LON0 = 32.7674, -117.2058
M_LAT = 111_132.92 - 559.82 * math.cos(2 * math.radians(LAT0))
M_LON = 111_412.84 * math.cos(math.radians(LAT0))
X = np.array([0.0, 3.0, 6.0, 10.0, 15.0, 20.0, 27.0, 35.0, 45.0, 55.0, 70.0, 85.0, 100.0])


def en_to_ll(e: float, n: float) -> tuple[float, float]:
  return LAT0 + n / M_LAT, LON0 + e / M_LON


def gps(e: float, n: float, heading: float, speed: float, accuracy: float = 1.0):
  lat, lon = en_to_ll(e, n)
  return SimpleNamespace(latitude=lat, longitude=lon, horizontalAccuracy=accuracy, bearingDeg=(90 - math.degrees(heading)) % 360,
                         bearingAccuracyDeg=1.0, speed=speed, hasFix=True, altitude=10.0)


def line(xs, ys):
  return SimpleNamespace(x=list(xs), y=list(ys))


def model_frame(curvature: float = 0.0, lane_half: float = 1.8, offset: float = 0.0, stop_at: float | None = None, v: float = 15.0):
  """A driving-model frame on a road of constant curvature (right positive), car `offset` meters right of center."""
  y = 0.5 * curvature * X ** 2 - offset
  vel = np.full(len(X), v)
  if stop_at is not None:
    vel = np.where(X >= stop_at, 0.0, v)
  return SimpleNamespace(
    laneLines=[line(X, y - 3 * lane_half), line(X, y - lane_half), line(X, y + lane_half), line(X, y + 3 * lane_half)],
    laneLineProbs=[0.3, 0.95, 0.95, 0.3],
    roadEdges=[line(X, y - 3 * lane_half), line(X, y + 3 * lane_half)],
    roadEdgeStds=[0.2, 0.2],
    position=line(X, y + offset * 0),
    velocity=SimpleNamespace(x=list(vel)),
    action=SimpleNamespace(desiredCurvature=curvature, shouldStop=stop_at is not None),
    meta=SimpleNamespace(laneChangeState="off"),
    leadsV3=[],
  )


def drive_straight(builder, t0: float, seconds: float, v: float = 15.0, heading: float = math.radians(80),
                   gps_offset: tuple[float, float] = (0.0, 0.0), rng=None, steer: bool = False):
  """Drive a straight road through the builder: 20 Hz motion, 10 Hz GPS and map updates."""
  rng = rng or np.random.default_rng(0)
  outs = []
  for k in range(int(seconds * 20)):
    t = t0 + k * 0.05
    d = v * (t - t0)
    e, n = d * math.cos(heading), d * math.sin(heading)
    builder.on_motion(t, v, 0.0)
    if k % 2 == 0:
      g = gps(e + gps_offset[0] + rng.normal(0, 0.3), n + gps_offset[1] + rng.normal(0, 0.3), heading, v)
      builder.on_gps(t, g)
      outs.append(builder.on_model(t, model_frame(v=v), v, steer))
  return outs
