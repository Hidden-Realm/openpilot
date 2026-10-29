"""
Acura Integra steering headroom.

The Integra reports the same EPS firmware as the 2022 Civic (39990-T38), which this
branch already drives at up to 5120. The Integra was left at 4096, and in sharp
corners it runs out of torque and drifts wide. This raises the ceiling to 5120 and
scales the PID gains by 4096/5120, so any request below the old limit sends the
same torque as before; only corners that used to saturate get more.

Lives here instead of opendbc so the fork does not need its own opendbc repo.
"""
from opendbc.car import structs
from opendbc.car.honda.values import CAR, CarControllerParams
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP

OLD_MAX = 4096
NEW_MAX = 5120


def apply_integra_steering(CI) -> bool:
  CP: structs.CarParams = CI.CP
  if CP.carFingerprint != CAR.ACURA_INTEGRA or CI.CP_SP.flags & HondaFlagsSP.EPS_MODIFIED:
    return False
  # only the stock PID tune; torque control and NNLC learn their own scaling
  if CP.lateralTuning.which() != 'pid' or list(CP.lateralParams.torqueBP) != [0, OLD_MAX] or list(CP.lateralParams.torqueV) != [0, OLD_MAX]:
    return False

  scale = OLD_MAX / NEW_MAX
  pid = CP.lateralTuning.pid
  CP.lateralParams.torqueBP = [0, NEW_MAX]
  CP.lateralParams.torqueV = [0, NEW_MAX]
  pid.kpV = [v * scale for v in pid.kpV]
  pid.kiV = [v * scale for v in pid.kiV]
  pid.kf = pid.kf * scale

  # the car controller read the old limit when it was built
  if CI.CC is not None:
    CI.CC.params = CarControllerParams(CP)
  return True
