"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Map steering: lets the world map (sunnypilot/worldmap) nudge the driving model's desired curvature.

It only ever adds the difference between the road shape the map knows and the shape of the model's own path,
times a weight the world map earns (see worldmap/qualify.py), and the result is capped at a small sideways
acceleration. It is off unless the "map steering" switch in device settings is on AND the map has passed its
confidence check, and it drops out immediately when the map message stops, during lane changes and below
road speeds. Everything after this (curvature rate and acceleration limits, torque limits, the panda's safety
checks, driver override and monitoring) is unchanged and applies on top.
"""
import time

import cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.common.realtime import DT_CTRL
from openpilot.common.swaglog import cloudlog

MAX_WEIGHT = 0.5
MAX_LAT_ACCEL = 0.3       # m/s^2: the most the map may change the requested sideways acceleration
MIN_SPEED = 8.0           # m/s
MAX_AGE = 0.3             # seconds: older map messages are ignored
RAMP_UP = 0.25            # weight per second
RAMP_DOWN = 2.0
PARAMS_PERIOD = 1.0


class MapSteer:
  def __init__(self, params: Params | None = None):
    self.params = params or Params()
    self.sm = messaging.SubMaster(['worldMapSP'])  # its own, so a missing world map never affects controlsd's checks
    self.enabled = False
    self.qualified = False
    self.weight = 0.0
    self.delta = 0.0
    self._param_t = 0.0
    self._errored = False

  def _read_params(self, now: float) -> None:
    if now - self._param_t > PARAMS_PERIOD:
      self.enabled = self.params.get_bool("WorldMapSteer")
      self.qualified = self.params.get_bool("WorldMapQualified")
      self._param_t = now

  def target_weight(self, now: float, v_ego: float, lane_change: bool) -> float:
    self.sm.update(0)
    if not (self.enabled and self.qualified) or lane_change or v_ego < MIN_SPEED:
      return 0.0
    if not self.sm.seen['worldMapSP'] or now - self.sm.recv_time['worldMapSP'] > MAX_AGE:
      return 0.0
    msg = self.sm['worldMapSP']
    if not (self.sm.valid['worldMapSP'] and msg.valid and msg.steerQualified and msg.mapPathValid):
      return 0.0
    return min(max(float(msg.steerWeight), 0.0), MAX_WEIGHT)

  def apply(self, model_curvature: float, v_ego: float, lane_change: bool) -> float:
    try:
      now = time.monotonic()
      self._read_params(now)
      target = self.target_weight(now, v_ego, lane_change)
      step = (RAMP_UP if target > self.weight else RAMP_DOWN) * DT_CTRL
      self.weight = max(min(self.weight + max(min(target - self.weight, step), -step), MAX_WEIGHT), 0.0)
      if self.weight <= 0.0:
        self.delta = 0.0
        return model_curvature
      limit = MAX_LAT_ACCEL / max(v_ego, MIN_SPEED) ** 2
      self.delta = min(max(self.weight * float(self.sm['worldMapSP'].curvatureDelta), -limit), limit)
      return model_curvature + self.delta
    except Exception:
      if not self._errored:
        cloudlog.exception("map steer failed, using the driving model alone")
        self._errored = True
      self.weight = 0.0
      self.delta = 0.0
      return model_curvature
