from types import SimpleNamespace

import numpy as np

from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, CarControllerParams
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP
from openpilot.sunnypilot.selfdrive.car.honda_integra import apply_integra_steering, OLD_MAX, NEW_MAX


def make_ci(fingerprint=CAR.ACURA_INTEGRA, sp_flags=0):
  CP = CarInterface.get_non_essential_params(fingerprint)
  CP_SP = CarInterface.get_non_essential_params_sp(CP, fingerprint)
  CP_SP.flags = sp_flags
  return SimpleNamespace(CP=CP, CP_SP=CP_SP, CC=SimpleNamespace(params=CarControllerParams(CP)))


def raw_torque(CP, normalized):
  ccp = CarControllerParams(CP)
  return np.interp(normalized * ccp.STEER_MAX, ccp.STEER_LOOKUP_BP, ccp.STEER_LOOKUP_V)


def test_integra_gets_more_headroom():
  ci = make_ci()
  assert list(ci.CP.lateralParams.torqueBP) == [0, OLD_MAX]
  old_pid = ci.CP.lateralTuning.pid
  old = (list(old_pid.kpV), list(old_pid.kiV), old_pid.kf)

  assert apply_integra_steering(ci)
  assert list(ci.CP.lateralParams.torqueBP) == [0, NEW_MAX]
  assert ci.CC.params.STEER_MAX == NEW_MAX

  # same raw torque for the same error below the old limit: gain * max is unchanged
  pid = ci.CP.lateralTuning.pid
  assert np.isclose(pid.kpV[0] * NEW_MAX, old[0][0] * OLD_MAX)
  assert np.isclose(pid.kiV[0] * NEW_MAX, old[1][0] * OLD_MAX)
  assert np.isclose(pid.kf * NEW_MAX, old[2] * OLD_MAX)
  assert np.isclose(raw_torque(ci.CP, 0.5 * OLD_MAX / NEW_MAX), 0.5 * OLD_MAX)
  assert raw_torque(ci.CP, 1.0) == NEW_MAX


def test_applies_once():
  ci = make_ci()
  assert apply_integra_steering(ci)
  assert not apply_integra_steering(ci)
  assert list(ci.CP.lateralParams.torqueBP) == [0, NEW_MAX]


def test_other_cars_untouched():
  ci = make_ci(CAR.HONDA_CIVIC_2022)
  before = list(ci.CP.lateralParams.torqueBP)
  assert not apply_integra_steering(ci)
  assert list(ci.CP.lateralParams.torqueBP) == before


def test_modified_eps_untouched():
  ci = make_ci(sp_flags=HondaFlagsSP.EPS_MODIFIED.value)
  assert not apply_integra_steering(ci)
  assert list(ci.CP.lateralParams.torqueBP) == [0, OLD_MAX]


def test_torque_control_untouched():
  ci = make_ci()
  CarInterface.configure_torque_tune(CAR.ACURA_INTEGRA, ci.CP.lateralTuning)
  assert not apply_integra_steering(ci)
  assert list(ci.CP.lateralParams.torqueBP) == [0, OLD_MAX]
