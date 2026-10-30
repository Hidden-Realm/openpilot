"""
Lining up the live view with the stored map.

GPS is off by a meter or more, and that error drifts slowly. On a road that is already mapped, the lane lines
and road edges the model sees right now should land on the lines in the map. We try small east/north shifts
of the live points and keep the one where they fit best (smallest distance to stored structure).

A straight road only pins down the sideways error; along the road every shift looks the same. So the fit is
judged separately across and along the car's heading, and each direction only updates the correction when its
fit has a clear minimum. The correction itself is a slow random walk with its own uncertainty.
"""
import math
from dataclasses import dataclass

import numpy as np

from openpilot.sunnypilot.worldmap.window import MapWindow

SEARCH_M = 2.5
STEP_M = 0.25
MIN_POINTS = 40        # about two lines' worth at one point per meter
CLEAR_MIN = 0.12       # meters of cost rise one meter away from the best shift
CLEAR_MIN_ALONG = 0.45  # along the road it has to be much clearer: straight lines fit anywhere along themselves
ALONG_MIN_QUALITY = 0.7
DRIFT_PER_S = 0.05     # how fast the GPS error is allowed to wander
GOOD_FIT_M = 0.35


@dataclass
class AlignResult:
  ok: bool = False
  quality: float = 0.0        # share of live points that sit on stored structure after the fit
  lateral: bool = False
  longitudinal: bool = False
  shift: tuple[float, float] = (0.0, 0.0)


class Aligner:
  def __init__(self):
    self.corr = np.zeros(2)                 # east, north meters added to the filter's position
    self.P = np.eye(2) * 4.0
    self.t: float | None = None
    self.quality = 0.0
    self.matched_t = -1e9

  def reset(self) -> None:
    self.corr[:] = 0.0
    self.P = np.eye(2) * 4.0
    self.quality = 0.0

  def lateral_std(self, heading: float) -> float:
    right = np.array([math.sin(heading), -math.cos(heading)])
    return math.sqrt(max(float(right @ self.P @ right), 0.0))

  def converged(self, heading: float, max_std: float = 0.6) -> bool:
    """Sure of the sideways correction. Along the road it may stay unsure: that only shifts things along the road."""
    return self.lateral_std(heading) < max_std

  def predict(self, t: float) -> None:
    if self.t is not None:
      dt = min(max(t - self.t, 0.0), 5.0)
      self.P += np.eye(2) * (DRIFT_PER_S ** 2) * dt
      np.minimum(self.P, 9.0, out=self.P)
    self.t = t

  def update(self, window: MapWindow, e: np.ndarray, n: np.ndarray, heading: float) -> AlignResult:
    """e/n: live structure points (lane lines, edges) already shifted by the current correction."""
    res = AlignResult()
    if window.has_structure < 200 or len(e) < MIN_POINTS:
      return res
    inside = window.inside(e, n, SEARCH_M + 1.0)
    e, n = e[inside], n[inside]
    if len(e) < MIN_POINTS:
      return res

    steps = np.arange(-SEARCH_M, SEARCH_M + 1e-6, STEP_M)
    k = len(steps)
    de = steps[:, None, None]
    dn = steps[None, :, None]
    ge, gn = np.broadcast_arrays(e[None, None, :] + de, n[None, None, :] + dn)
    cost = window.lookup(window.dist, ge, gn, default=8.0).mean(axis=2)   # [east shift, north shift]
    i, j = np.unravel_index(np.argmin(cost), cost.shape)
    best = float(cost[i, j])
    shift = np.array([steps[i], steps[j]])

    # sharpness of the minimum across and along the heading, one meter out
    c, s = math.cos(heading), math.sin(heading)
    fwd = np.array([c, s])
    right = np.array([s, -c])

    def cost_at(v):
      p = shift + v
      ii = int(np.clip(round((p[0] + SEARCH_M) / STEP_M), 0, k - 1))
      jj = int(np.clip(round((p[1] + SEARCH_M) / STEP_M), 0, k - 1))
      return float(cost[ii, jj])

    lat_rise = min(cost_at(right), cost_at(-right)) - best
    long_rise = min(cost_at(fwd), cost_at(-fwd)) - best
    on_edge = np.any(np.abs(shift) >= SEARCH_M - 1e-6)
    after = window.lookup(window.dist, e + shift[0], n + shift[1], default=8.0)
    res.quality = float(np.mean(after < GOOD_FIT_M))
    res.lateral = lat_rise > CLEAR_MIN and not on_edge
    res.longitudinal = long_rise > CLEAR_MIN_ALONG and res.quality > ALONG_MIN_QUALITY and not on_edge
    res.shift = (float(shift[0]), float(shift[1]))

    # Each fit only says something about its own axis. A full Kalman update would let a sideways fit move the
    # along-road correction too (through the large along-road uncertainty) and drift as the road bends, so the
    # update is kept on the measured axis.
    axes = ([right] if res.lateral else []) + ([fwd] if res.longitudinal else [])
    for axis in axes:
      z = float(shift @ axis)                     # extra correction along this axis
      p_axis = float(axis @ self.P @ axis)
      R = 0.3 ** 2 + best ** 2
      gain = p_axis / (p_axis + R)
      self.corr = self.corr + axis * (gain * z)
      self.P = self.P - np.outer(axis, axis) * (gain * p_axis)
      w, V = np.linalg.eigh(0.5 * (self.P + self.P.T))
      self.P = V @ np.diag(np.maximum(w, 0.01)) @ V.T
    res.ok = bool(axes)
    if res.ok:
      self.matched_t = self.t if self.t is not None else 0.0
    self.quality = 0.8 * self.quality + 0.2 * res.quality
    return res
