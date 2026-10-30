import math

import numpy as np
import pytest

from openpilot.sunnypilot.worldmap import builder as B
from openpilot.sunnypilot.worldmap.mappath import curvature_ahead
from openpilot.sunnypilot.worldmap.pose import PoseFilter
from openpilot.sunnypilot.worldmap.qualify import QualifyStats, SteerWeight
from openpilot.sunnypilot.worldmap.tests.helpers import drive_straight, gps, model_frame
from openpilot.sunnypilot.worldmap.tiles import TileStore


@pytest.fixture
def clock(monkeypatch):
  wall = {"t": 1.8e9}
  monkeypatch.setattr(B, "wall_now", lambda: wall["t"])
  return wall


def test_pose_filter_follows_gps_and_rejects_a_jump():
  pf = PoseFilter()
  h = math.radians(45)
  for k in range(200):
    t = k * 0.05
    pf.predict(t, 10.0, 0.0)
    if k % 2 == 0:
      d = 10.0 * t
      pf.update_gps(t, *(lambda g: (g.latitude, g.longitude, g.horizontalAccuracy, g.bearingDeg, g.bearingAccuracyDeg, g.speed))(
        gps(d * math.cos(h), d * math.sin(h), h, 10.0)))
  assert pf.valid and abs(pf.heading - h) < 0.02
  e, n = pf.x[0], pf.x[1]
  pf.update_gps(10.0, *(lambda g: (g.latitude, g.longitude, 1.0, g.bearingDeg, 1.0, 10.0))(gps(e + 80.0, n, h, 10.0)))
  assert math.hypot(pf.x[0] - e, pf.x[1] - n) < 5.0  # an 80 m jump is not believed


def test_first_drive_builds_a_map_and_second_drive_follows_it(tmp_path, clock):
  store = TileStore(str(tmp_path))
  b1 = B.WorldBuilder(store)
  outs = drive_straight(b1, 0.0, 30.0)
  assert any(o["valid"] for o in outs)
  assert not any(o["map_valid"] for o in outs if o["valid"])          # nothing to follow on a new road
  assert np.allclose(b1.align.corr, 0.0)                              # and nothing to line up with
  store.flush()

  clock["t"] += 3600
  b2 = B.WorldBuilder(TileStore(str(tmp_path)))
  outs = drive_straight(b2, 100.0, 25.0, gps_offset=(1.5, -1.0), rng=np.random.default_rng(1))
  valid = [o for o in outs[60:] if o["valid"]]
  assert sum(o["map_valid"] for o in valid) > 0.8 * len(valid)
  errs = [o["lat_error"] for o in valid if o["lat_error"] >= 0]
  assert np.median(errs) < 0.5
  # the sideways part of the GPS offset was learnt: correction points back across the road
  h = math.radians(80)
  right = np.array([math.sin(h), -math.cos(h)])
  assert abs(float(b2.align.corr @ right) + float(np.array([1.5, -1.0]) @ right)) < 0.5


def test_planned_stop_is_remembered(tmp_path, clock):
  store = TileStore(str(tmp_path))
  b = B.WorldBuilder(store)
  drive_straight(b, 0.0, 5.0)
  t = 5.0
  for k in range(20):
    t += 0.05
    b.on_motion(t, 10.0, 0.0)
    if k % 2 == 0:
      b.on_model(t, model_frame(stop_at=35.0, v=10.0), 10.0, False)
  assert b.stop is not None and abs(b.stop["dist"] - 35.0) < 1.0
  stops = [s for tile in store.loaded() for s in tile.meta["stops"]]
  assert len(stops) == 1 and stops[0]["hits"] >= 5


def test_curvature_sign_matches_openpilot():
  x = np.arange(0.0, 60.0, 2.0)
  right = np.stack([x, 0.5 * 0.01 * x ** 2], axis=1)
  assert abs(curvature_ahead(right, 15.0) - 0.01) < 0.001
  assert curvature_ahead(right * [1, -1], 15.0) < 0


def test_qualification_needs_distance_and_agreement():
  q = QualifyStats()
  for _ in range(240):
    q.add(100.0, True)   # 24 km
  assert not q.qualified
  for _ in range(40):
    q.add(100.0, True)   # older kilometers fade, so it takes a little more than 25 km of driving
  assert q.qualified
  for _ in range(150):
    q.add(100.0, False)
  assert not q.qualified  # agreement fell away


def test_steer_weight_gates_and_ramps():
  sw = SteerWeight()
  kw = dict(enabled=True, qualified=True, map_valid=True, map_conf=1.0, lateral_error=0.1, align_ok=True, pose_ok=True,
            v_ego=20.0, lane_change=False, model_lane_conf=0.0)
  t = 0.0
  for _ in range(10):
    t += 0.1
    assert sw.update(t, 0.1, **kw) == 0.0   # agreement must hold for a while first
  for _ in range(100):
    t += 0.1
    w = sw.update(t, 0.1, **kw)
  assert 0.45 < w <= 0.5
  assert sw.update(t + 0.1, 0.1, **{**kw, "lane_change": True}) < w
  for flag in ("enabled", "qualified", "map_valid", "align_ok", "pose_ok"):
    s = SteerWeight()
    for i in range(100):
      out = s.update(i * 0.1, 0.1, **{**kw, flag: False})
    assert out == 0.0, flag
