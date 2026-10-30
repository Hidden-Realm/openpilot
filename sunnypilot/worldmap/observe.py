"""
What the driving model sees, in a form the map can use.

Everything is in openpilot's calibrated car frame: x forward, y right, meters, origin at the camera (which is
also where the device's GPS is). Lines are cut to the stretch where the model is confident: lane lines fade
with their probability, road edges with their standard deviation.

The model also tells us when it plans to stop (stop signs, red lights, a stopped car ahead): it does not label
what it is stopping for, but its plan ends short and `action.shouldStop` turns on. That stop point is what we
record; the camera detector adds a label when it sees a sign or a light there.
"""
from dataclasses import dataclass, field

import numpy as np

MIN_X = 3.0           # the hood hides the first few meters
LANE_MIN_PROB = 0.5
EDGE_MAX_STD = 0.6
STOP_MAX_DIST = 80.0


@dataclass
class Observation:
  lanes: list[np.ndarray] = field(default_factory=list)     # each (N, 2) x, y
  lane_probs: list[float] = field(default_factory=list)
  edges: list[np.ndarray] = field(default_factory=list)     # left, right
  path: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
  plan_v: np.ndarray = field(default_factory=lambda: np.zeros(0))
  curvature: float = 0.0
  should_stop: bool = False
  stop_dist: float | None = None
  lane_change: bool = False
  ego_lane_conf: float = 0.0   # how sure the model is about the lines of our own lane, 0..1


def _xy(line, max_x: float) -> np.ndarray:
  x = np.asarray(line.x, dtype=np.float64)
  y = np.asarray(line.y, dtype=np.float64)
  keep = (x >= MIN_X) & (x <= max_x)
  return np.stack([x[keep], y[keep]], axis=1)


def lane_range(prob: float) -> float:
  """How far ahead to trust a lane line of this probability."""
  return float(np.interp(prob, [LANE_MIN_PROB, 0.8, 1.0], [25.0, 50.0, 70.0]))


def observe(model, night: bool = False) -> Observation:
  obs = Observation()
  probs = list(model.laneLineProbs)
  for i, line in enumerate(model.laneLines):
    p = probs[i] if i < len(probs) else 0.0
    if p < LANE_MIN_PROB:
      continue
    reach = lane_range(p) * (0.8 if night else 1.0)
    pts = _xy(line, reach)
    if len(pts) >= 3:
      obs.lanes.append(pts)
      obs.lane_probs.append(float(p))
  stds = list(model.roadEdgeStds)
  for i, edge in enumerate(model.roadEdges):
    std = stds[i] if i < len(stds) else 9.0
    if std > EDGE_MAX_STD * (0.8 if night else 1.0):
      obs.edges.append(np.zeros((0, 2)))
      continue
    obs.edges.append(_xy(edge, 60.0 if std < 0.3 else 40.0))
  obs.path = _xy(model.position, 100.0)
  obs.plan_v = np.asarray(model.velocity.x, dtype=np.float64)
  obs.curvature = float(model.action.desiredCurvature)
  obs.should_stop = bool(model.action.shouldStop)
  obs.lane_change = str(model.meta.laneChangeState) != "off"
  if len(probs) >= 3:
    obs.ego_lane_conf = float(min(probs[1], probs[2]))

  # where the plan comes to rest: the first point whose planned speed is near zero
  px = np.asarray(model.position.x, dtype=np.float64)
  if obs.should_stop and len(obs.plan_v) == len(px) and len(px):
    slow = np.nonzero(obs.plan_v < 0.5)[0]
    if len(slow):
      d = float(px[slow[0]])
      if 2.0 < d < STOP_MAX_DIST:
        obs.stop_dist = d
  return obs


def path_y_at(path: np.ndarray, x: np.ndarray) -> np.ndarray:
  """The path's lateral offset at the given distances ahead (NaN past its end)."""
  if len(path) < 2:
    return np.full(len(x), np.nan)
  out = np.interp(x, path[:, 0], path[:, 1], left=np.nan, right=np.nan)
  return out
