from types import SimpleNamespace

import pytest

from openpilot.sunnypilot.selfdrive.controls.lib import map_steer as M


class FakeParams:
  def __init__(self, **values):
    self.values = values

  def get_bool(self, key):
    return self.values.get(key, False)


class FakeSM:
  def __init__(self, msg, recv_time, valid=True):
    self.msg = msg
    self.recv_time = {"worldMapSP": recv_time}
    self.seen = {"worldMapSP": True}
    self.valid = {"worldMapSP": valid}

  def update(self, timeout=0):
    pass

  def __getitem__(self, key):
    return self.msg


def make(monkeypatch, now=100.0, enabled=True, qualified=True, weight=0.5, delta=0.01, age=0.05, valid=True):
  monkeypatch.setattr(M.time, "monotonic", lambda: now)
  monkeypatch.setattr(M.messaging, "SubMaster", lambda services: None)
  ms = M.MapSteer(FakeParams(WorldMapSteer=enabled, WorldMapQualified=qualified))
  msg = SimpleNamespace(valid=valid, steerQualified=qualified, mapPathValid=True, steerWeight=weight, curvatureDelta=delta)
  ms.sm = FakeSM(msg, now - age)
  return ms


def run(ms, seconds=5.0, model=0.0, v=20.0, lane_change=False):
  out = model
  for _ in range(int(seconds / M.DT_CTRL)):
    out = ms.apply(model, v, lane_change)
  return out


@pytest.mark.parametrize("kw", [{"enabled": False}, {"qualified": False}, {"age": 1.0}, {"valid": False}])
def test_off_unless_everything_agrees(monkeypatch, kw):
  ms = make(monkeypatch, **kw)
  assert run(ms) == 0.0 and ms.weight == 0.0


def test_change_is_capped_by_sideways_acceleration(monkeypatch):
  ms = make(monkeypatch, delta=0.05)  # a huge map/model disagreement
  out = run(ms, v=25.0)
  assert ms.weight == pytest.approx(0.5)
  assert abs(out) <= M.MAX_LAT_ACCEL / 25.0 ** 2 + 1e-9
  ms = make(monkeypatch, delta=-0.05)
  assert run(ms, v=25.0) == pytest.approx(-M.MAX_LAT_ACCEL / 25.0 ** 2)


def test_small_nudge_passes_through_weighted(monkeypatch):
  ms = make(monkeypatch, weight=0.4, delta=0.0002)
  assert run(ms, model=0.001, v=20.0) == pytest.approx(0.001 + 0.4 * 0.0002)


def test_drops_out_for_lane_changes_and_low_speed(monkeypatch):
  ms = make(monkeypatch)
  run(ms)
  assert ms.weight > 0.4
  run(ms, seconds=0.5, lane_change=True)
  assert ms.weight == 0.0
  run(ms, seconds=3.0)
  run(ms, seconds=0.5, v=5.0)
  assert ms.weight == 0.0


def test_ramps_in_slowly(monkeypatch):
  ms = make(monkeypatch)
  run(ms, seconds=0.4)
  assert 0.05 < ms.weight < 0.15


def test_errors_fall_back_to_the_model(monkeypatch):
  ms = make(monkeypatch)
  ms.sm = None  # anything going wrong
  assert ms.apply(0.002, 20.0, False) == 0.002
