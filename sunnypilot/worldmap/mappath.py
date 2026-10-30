"""
The road ahead according to the map.

On a road we have driven before, the cells the car drove over form a ridge along the middle of the lane, with
a driving direction stored for each cell. Starting at the car, we step forward two meters at a time and, at
each step, slide sideways onto the top of that ridge (only cells driven in our direction count). The result is
the path the map expects, in the car frame, plus how far ahead it reaches and how well-trodden it is.

That path is compared with the driving model's planned path. The comparison feeds the live agreement score,
and over many kilometers decides whether map steering may be offered at all (see qualify.py).
"""
import math
from dataclasses import dataclass, field

import numpy as np

from openpilot.sunnypilot.worldmap.geo import en_to_car
from openpilot.sunnypilot.worldmap.window import MapWindow

STEP_M = 2.0
MAX_AHEAD = 80.0
SEARCH = np.arange(-1.25, 1.2501, 0.25)
MIN_HEAT = 0.5          # ridge height (in passes) where it is still followed
TRUSTED_PASSES = 3.0    # this many passes over a cell makes it fully trusted
MAX_TURN = math.radians(12)  # per step
COMPARE_X = np.array([10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0])


@dataclass
class MapPath:
  valid: bool = False
  points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))   # car frame x, y
  length: float = 0.0
  confidence: float = 0.0
  curvature: float = 0.0


def follow(window: MapWindow, e: float, n: float, heading: float) -> MapPath:
  out = MapPath()
  if not window.inside(np.array([e]), np.array([n]))[0]:
    return out
  pts_e, pts_n, heats = [e], [n], []
  h = heading
  pe, pn = e, n
  for _ in range(int(MAX_AHEAD / STEP_M)):
    ce, cn = pe + STEP_M * math.cos(h), pn + STEP_M * math.sin(h)
    re, rn = math.sin(h), -math.cos(h)          # right of the current heading
    se = ce + SEARCH * re
    sn = cn + SEARCH * rn
    ridge = window.ridge_at(se, sn)
    heat, ue, un, ok = ridge[:, 0], ridge[:, 1], ridge[:, 2], ridge[:, 3] > 0.5
    same_way = (ue * math.cos(h) + un * math.sin(h)) > 0.6
    heat = np.where(ok & same_way, heat, 0.0)
    k = int(np.argmax(heat))
    if heat[k] < MIN_HEAT:
      break
    off = SEARCH[k]
    if 0 < k < len(SEARCH) - 1:                  # parabola through the top three for sub-cell precision
      a, b, c = heat[k - 1], heat[k], heat[k + 1]
      den = a - 2 * b + c
      if den < 0:
        off += 0.25 * 0.5 * (a - c) / den
    ne, nn = ce + off * re, cn + off * rn
    new_h = math.atan2(nn - pn, ne - pe)
    turn = (new_h - h + math.pi) % (2 * math.pi) - math.pi
    h = h + float(np.clip(turn, -MAX_TURN, MAX_TURN))
    pe, pn = ne, nn
    pts_e.append(pe)
    pts_n.append(pn)
    heats.append(float(heat[k]))
  if len(pts_e) < 6:
    return out
  x, y = en_to_car(np.array(pts_e), np.array(pts_n), e, n, heading)
  out.points = np.stack([x, y], axis=1)
  out.length = float(x[-1])
  out.confidence = float(np.clip(np.mean(heats) / TRUSTED_PASSES, 0.0, 1.0) * np.clip(out.length / 40.0, 0.0, 1.0))
  out.valid = out.length >= 10.0
  return out


def curvature_ahead(points: np.ndarray, v_ego: float) -> float:
  """Curvature (1/m, positive to the right like openpilot's) of the map path over the next stretch."""
  if len(points) < 4:
    return 0.0
  reach = float(np.clip(1.5 * v_ego + 10.0, 20.0, 60.0))
  # the first few meters say more about where the car sits in its lane than about the road's shape
  keep = (points[:, 0] >= 6.0) & (points[:, 0] <= reach)
  if keep.sum() < 4:
    return 0.0
  x, y = points[keep, 0], points[keep, 1]
  a, b, _ = np.polyfit(x, y, 2)
  return float(2 * a / (1 + b * b) ** 1.5)


def lateral_error(map_points: np.ndarray, model_path: np.ndarray) -> float | None:
  """Largest sideways gap between the map path and the model path over the compared stretch."""
  if len(map_points) < 3 or len(model_path) < 3:
    return None
  xs = COMPARE_X[COMPARE_X <= min(map_points[-1, 0], model_path[-1, 0])]
  if len(xs) < 3:
    return None
  ym = np.interp(xs, map_points[:, 0], map_points[:, 1])
  yp = np.interp(xs, model_path[:, 0], model_path[:, 1])
  return float(np.max(np.abs(ym - yp)))
