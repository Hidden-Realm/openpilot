from cereal import log
from openpilot.selfdrive.controls.lib.longitudinal_planner import get_max_accel, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS

Personality = log.LongitudinalPersonality


def test_relaxed_is_stock():
  for v, a in zip(A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS, strict=True):
    assert get_max_accel(v, Personality.relaxed) == a


def test_faster_with_more_aggressive_personality():
  for v in (5., 15., 25., 35.):
    relaxed, standard, aggressive = (get_max_accel(v, p) for p in (Personality.relaxed, Personality.standard, Personality.aggressive))
    assert relaxed < standard < aggressive


def test_same_from_a_standstill():
  # the MPC plans with CRUISE_MAX_ACCEL = 1.6, so the launch limit stays where it was
  assert {get_max_accel(0., p) for p in (Personality.relaxed, Personality.standard, Personality.aggressive)} == {1.6}
