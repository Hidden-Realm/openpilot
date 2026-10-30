"""
Builds and uses the world map, one driving-model frame at a time.

Each update:
  1. moves the pose filter on and applies the map-alignment correction,
  2. refreshes the working window of the stored map around the car,
  3. lines the live lane lines / road edges up with that window (align.py),
  4. follows the stored path ahead (mappath.py) and compares it with the model's plan,
  5. adds this frame's view to the tiles: where the camera looked, road, edges, lanes, where we drove,
     and where the model planned to stop,
  6. keeps a snapshot for the live viewer and the numbers that go out on worldMapSP.

Nothing here talks to the car. The only thing that can reach steering is the published weight, and controlsd
decides what to do with it.
"""
import math
import time
from collections import deque

import numpy as np

from openpilot.sunnypilot.worldmap import geo
from openpilot.sunnypilot.worldmap.imgops import draw_mask
from openpilot.sunnypilot.worldmap.align import Aligner, AlignResult
from openpilot.sunnypilot.worldmap.geo import car_to_en
from openpilot.sunnypilot.worldmap.mappath import MapPath, curvature_ahead, follow, lateral_error
from openpilot.sunnypilot.worldmap.observe import Observation, observe
from openpilot.sunnypilot.worldmap.pose import PoseFilter
from openpilot.sunnypilot.worldmap.qualify import AGREE_LAT_ACCEL, AGREE_M, QualifyStats, SteerWeight
from openpilot.sunnypilot.worldmap.tiles import TileStore, now as wall_now
from openpilot.sunnypilot.worldmap.window import MapWindow

WINDOW_REFRESH_S = 10.0
WINDOW_AHEAD_M = 40.0
SEEN_RANGE = 45.0
SEEN_HALF_WIDTH = 10.0
TRAIL_WIDTH_PX = 1  # a thin trail blurs into a ridge with a clear top (a wide one has a flat top the path wobbles on)
TRAIL_DELAY = 1.0   # seconds; the trail is smoothed over this much before and after each point
CURV_SMOOTH_S = 1.0
STOP_MERGE_M = 8.0
OBJECT_FRESH_S = 1.0


def _rounded(points: np.ndarray, step: int = 1) -> list:
  return np.round(points[::step], 2).tolist() if len(points) else []


class WorldBuilder:
  def __init__(self, store: TileStore, stats: QualifyStats | None = None):
    self.store = store
    self.pose = PoseFilter()
    self.align = Aligner()
    self.stats = stats or QualifyStats()
    self.steer = SteerWeight()
    self.window: MapWindow | None = None
    self.window_t = -1e9
    self.prev_ego: tuple[int, float, float] | None = None
    self.prev_trail: tuple[int, float, float] | None = None
    self.trail_hist: deque = deque()
    self.t: float | None = None
    self.altitude = 0.0
    self.night = False
    self.objects: list[dict] = []
    self.objects_t = -1e9
    self.detector_info: dict = {}
    self.map_path = MapPath()
    self.align_last: AlignResult | None = None
    self.lat_error: float | None = None
    self.curvature_delta = 0.0
    self.curvature_lp = 0.0
    self.stop: dict | None = None
    self.stats_changed = False
    self.frames = 0
    self.update_ms = 0.0
    self.live: dict = {"valid": False}
    self.out: dict = {"valid": False}
    self.software: dict = {}

  # ---- inputs between model frames
  def on_motion(self, t: float, v_ego: float, yaw_rate: float) -> None:
    self.pose.predict(t, v_ego, yaw_rate)

  def on_gps(self, t: float, gps) -> None:
    self.pose.update_gps(t, gps.latitude, gps.longitude, gps.horizontalAccuracy, gps.bearingDeg, gps.bearingAccuracyDeg,
                         gps.speed, bool(gps.hasFix))
    if gps.hasFix:
      self.altitude = float(gps.altitude)

  def on_llk(self, t: float, llk) -> None:
    pg = llk.positionGeodetic
    yaw = llk.calibratedOrientationNED
    if not (pg.valid and llk.gpsOK and yaw.valid and len(pg.value) >= 3 and len(pg.std) >= 2):
      return
    self.pose.update_llk(t, pg.value[0], pg.value[1], max(pg.std[0], pg.std[1]), yaw.value[2], yaw.std[2] if len(yaw.std) > 2 else 0.05)
    self.altitude = float(pg.value[2])

  def on_objects(self, t: float, msg) -> None:
    self.objects_t = t
    self.night = bool(msg.night)
    self.detector_info = {"ready": bool(msg.detectorReady), "ms": round(float(msg.inferenceTime) * 1000, 1),
                          "hz": round(float(msg.rateHz), 1), "skipped": int(msg.skipped)}
    self.objects = [{
      "id": int(o.id), "kind": str(o.kind), "conf": round(float(o.confidence), 2),
      "x": round(float(o.x), 2), "y": round(float(o.y), 2), "vx": round(float(o.vx), 2), "vy": round(float(o.vy), 2),
      "w": round(float(o.width), 2), "cam": str(o.camera), "light": str(o.lightColor), "age": round(float(o.age), 1),
    } for o in msg.objects]

  # ---- the per-frame update
  def corrected(self) -> tuple[float, float, float]:
    return float(self.pose.x[0] + self.align.corr[0]), float(self.pose.x[1] + self.align.corr[1]), self.pose.heading

  def on_model(self, t: float, model, v_ego: float, steer_enabled: bool) -> dict:
    t0 = time.perf_counter()
    dt = 0.1 if self.t is None else min(max(t - self.t, 0.0), 1.0)
    self.t = t
    self.frames += 1
    obs = observe(model, self.night)
    self.align.predict(t)

    if not self.pose.valid:
      self.window = None
      self.prev_ego = None
      self.prev_trail = None
      self.trail_hist.clear()
      self.steer.update(t, dt, enabled=steer_enabled, qualified=self.stats.qualified, map_valid=False, map_conf=0.0,
                        lateral_error=None, align_ok=False, pose_ok=False, v_ego=v_ego, lane_change=obs.lane_change,
                        model_lane_conf=obs.ego_lane_conf)
      self._snapshot(obs, v_ego)
      self.update_ms = (time.perf_counter() - t0) * 1000
      return self.out

    frame = self.pose.frame
    assert frame is not None
    e, n, h = self.corrected()
    if self.window is None or self.window.stale(self.pose.frame_id, e, n) or t - self.window_t > WINDOW_REFRESH_S:
      ce, cn = car_to_en(WINDOW_AHEAD_M, 0.0, e, n, h)
      self.window = MapWindow(self.store, frame, self.pose.frame_id, float(ce), float(cn), wall_now(), e, n)
      self.window_t = t

    # line the live structure up with the stored map
    # the model gives each line at a handful of uneven distances; resample to one point a meter so every line
    # counts the same in the fit
    struct = [_resample(p, 5.0, 40.0, 1.0) for p in obs.lanes + obs.edges if len(p) >= 2]
    struct = [p for p in struct if len(p)]
    if struct and self.frames % 2 == 0:
      pts = np.concatenate(struct)
      if len(pts):
        pe, pn = car_to_en(pts[:, 0], pts[:, 1], e, n, h)
        self.align_last = self.align.update(self.window, pe, pn, h)
        e, n, h = self.corrected()

    # the stored road ahead, and how it compares with the model's plan
    self.map_path = follow(self.window, e, n, h)
    self.lat_error = lateral_error(self.map_path.points, obs.path) if self.map_path.valid else None
    # road shape by the same fit for both paths; steering only ever gets the difference between them
    # The map's shape carries the slow GPS wander of the drives it was built from (tenths of a meter over tens
    # of meters). That is noise on gentle curves, so the curvature is smoothed over about a second, and a
    # stretch only counts as agreeing when the curves match too, not just the positions.
    raw_curv = curvature_ahead(self.map_path.points, v_ego) if self.map_path.valid else 0.0
    model_curvature = curvature_ahead(obs.path, v_ego) if len(obs.path) >= 4 else 0.0
    alpha = min(dt / CURV_SMOOTH_S, 1.0)
    if self.map_path.valid:
      self.curvature_lp += alpha * (raw_curv - self.curvature_lp)
      self.curvature_delta += alpha * ((raw_curv - model_curvature) - self.curvature_delta)
    else:
      self.curvature_lp = model_curvature
      self.curvature_delta = 0.0
    curvature = self.curvature_lp
    if (self.map_path.valid and self.map_path.confidence >= 0.5 and self.lat_error is not None and obs.ego_lane_conf >= 0.6
        and not obs.lane_change and v_ego > 5.0):
      curves_agree = abs(self.curvature_delta) * v_ego ** 2 < AGREE_LAT_ACCEL
      self.stats_changed |= self.stats.add(v_ego * dt, self.lat_error < AGREE_M and curves_agree)

    align_ok = self.align.converged(h) and self.align.quality > 0.4
    self.steer.update(t, dt, enabled=steer_enabled, qualified=self.stats.qualified, map_valid=self.map_path.valid,
                      map_conf=self.map_path.confidence, lateral_error=self.lat_error, align_ok=align_ok,
                      pose_ok=self.pose.pos_std < 3.0, v_ego=v_ego, lane_change=obs.lane_change, model_lane_conf=obs.ego_lane_conf)

    self.trail_hist.append((t, self.pose.frame_id, e, n))
    while self.trail_hist and self.trail_hist[0][0] < t - 3 * TRAIL_DELAY:
      self.trail_hist.popleft()
    new_area = self.window.has_structure < 200
    if self.pose.pos_std < 3.0 and (new_area or align_ok):
      self._fuse(obs, e, n, h, model)
    self.prev_ego = (self.pose.frame_id, e, n)

    self.out = {
      "valid": True, "lat": 0.0, "lon": 0.0, "bearing": geo.bearing_from_heading(h), "pos_std": self.pose.pos_std,
      "corr_lat": 0.0, "corr_long": 0.0, "align_quality": self.align.quality,
      "map_valid": self.map_path.valid, "map_x": self.map_path.points[:, 0].tolist() if self.map_path.valid else [],
      "map_y": self.map_path.points[:, 1].tolist() if self.map_path.valid else [],
      "curvature": curvature, "curvature_delta": self.curvature_delta, "model_curvature": model_curvature,
      "map_conf": self.map_path.confidence,
      "agreement": 0.0 if self.lat_error is None else math.exp(-(self.lat_error / AGREE_M) ** 2),
      "lat_error": -1.0 if self.lat_error is None else self.lat_error,
      "qualified": self.stats.qualified, "weight": self.steer.weight,
      "stop": self.stop is not None, "stop_dist": self.stop["dist"] if self.stop else 0.0,
      "stop_kind": self.stop["kind"] if self.stop else "unknown",
      "compared_km": self.stats.compared_km, "agreed_km": self.stats.agreed_km, "night": self.night,
    }
    lat, lon = frame.to_ll(e, n)
    self.out["lat"], self.out["lon"] = float(lat), float(lon)
    ce, cn = self.align.corr
    self.out["corr_long"] = float(ce * math.cos(h) + cn * math.sin(h))
    self.out["corr_lat"] = float(ce * math.sin(h) - cn * math.cos(h))
    self._snapshot(obs, v_ego)
    self.update_ms = (time.perf_counter() - t0) * 1000
    self.out["ms"] = self.update_ms
    return self.out

  # ---- writing into the tiles
  def _fuse(self, obs: Observation, e: float, n: float, h: float, model) -> None:
    lat_lon = self.pose.frame.to_ll if self.pose.frame is not None else None
    if lat_lon is None:
      return
    reach = max([float(p[-1, 0]) for p in obs.lanes + obs.edges if len(p)] + [20.0])
    reach = min(reach, SEEN_RANGE)
    seen = np.array([[3.0, -SEEN_HALF_WIDTH * 0.6], [reach, -SEEN_HALF_WIDTH], [reach, SEEN_HALF_WIDTH], [3.0, SEEN_HALF_WIDTH * 0.6]])
    shapes: list[tuple[str, np.ndarray, str, float]] = [("seen", seen, "fill", 1)]

    left, right = (obs.edges + [np.zeros((0, 2)), np.zeros((0, 2))])[:2]
    if len(left) >= 3 and len(right) >= 3:
      reach_r = min(left[-1, 0], right[-1, 0])
      road = np.concatenate([left[left[:, 0] <= reach_r], right[right[:, 0] <= reach_r][::-1]])
      shapes.append(("road", road, "fill", 1))
    elif len(obs.path) >= 3:
      p = obs.path[obs.path[:, 0] <= 40.0]
      if len(p) >= 3:
        road = np.concatenate([p + [0.0, -1.6], (p + [0.0, 1.6])[::-1]])
        shapes.append(("road", road, "fill", 1))
    for edge in obs.edges:
      if len(edge) >= 2:
        shapes.append(("edge", edge, "line", 1))
    for lane in obs.lanes:
      if len(lane) >= 2:
        shapes.append(("lane", lane, "line", 1))

    ll_shapes = []
    all_lat, all_lon = [], []
    for layer, pts, kind, width in shapes:
      pe, pn = car_to_en(pts[:, 0], pts[:, 1], e, n, h)
      la, lo = lat_lon(pe, pn)
      ll_shapes.append((layer, np.asarray(la), np.asarray(lo), kind, width))
      probe = _probe_points(pts, 4.0, closed=(kind == "fill"))  # extra points so every tile a shape crosses is found
      pe, pn = car_to_en(probe[:, 0], probe[:, 1], e, n, h)
      la, lo = lat_lon(pe, pn)
      all_lat.append(np.asarray(la))
      all_lon.append(np.asarray(lo))

    # the trail of the car itself: where we actually drove, and which way. Drawn a second late from a smoothed
    # track, because each GPS fix nudges the raw position a little and those kinks would read as tight curves.
    trail = None
    trail_h = h
    smooth = self._smoothed_trail_point()
    if smooth is not None:
      se, sn, trail_h = smooth
      if self.prev_trail is not None and self.prev_trail[0] == self.pose.frame_id:
        pe0, pn0 = self.prev_trail[1], self.prev_trail[2]
        if 0.05 < math.hypot(se - pe0, sn - pn0) < 10.0:
          la, lo = lat_lon(np.array([pe0, se]), np.array([pn0, sn]))
          trail = (np.asarray(la), np.asarray(lo))
          all_lat.append(trail[0])
          all_lon.append(trail[1])
      self.prev_trail = (self.pose.frame_id, se, sn)

    stamp = wall_now()
    groups = geo.group_by_tile(np.concatenate(all_lat), np.concatenate(all_lon))
    ego_lat, ego_lon = lat_lon(e, n)
    ego_key = geo.tile_key(float(ego_lat), float(ego_lon))
    for key in set(groups) | {ego_key}:
      tile = self.store.get(key, create=True)
      if tile is None:
        continue
      tile.prior(stamp)  # freeze what this tile held before this pass, before drawing into it
      tile.last_written = stamp
      if key == ego_key:
        tile.note_visit(stamp)
      for layer, la, lo, kind, width in ll_shapes:
        _stamp(tile, layer, *geo.ll_to_tile_px(key, la, lo), kind, int(width))
      if trail is not None:
        cols, rows = geo.ll_to_tile_px(key, trail[0], trail[1])
        mask_rc = _stamp(tile, "drive", cols, rows, "line", TRAIL_WIDTH_PX)
        if mask_rc is not None:
          rs, cs, mask = mask_rc
          m = mask.astype(np.float32)
          tile.dir_e[rs, cs] += m * math.cos(trail_h)
          tile.dir_n[rs, cs] += m * math.sin(trail_h)
          tile.alt[rs, cs] += m * self.altitude

    self._record_stop(obs, e, n, h, model, stamp)

  def _smoothed_trail_point(self) -> tuple[float, float, float] | None:
    """Position and heading TRAIL_DELAY seconds ago, from a quadratic fit over +-TRAIL_DELAY of the track.
    A quadratic follows a curve exactly, so smoothing does not cut corners like an average would."""
    hist = self.trail_hist
    if len(hist) < 5 or self.t is None:
      return None
    tc = self.t - TRAIL_DELAY
    ts = np.array([p[0] for p in hist])
    if ts[0] > tc - TRAIL_DELAY * 0.8:
      return None
    sel = np.abs(ts - tc) <= TRAIL_DELAY
    if sel.sum() < 5 or len({hist[i][1] for i in np.nonzero(sel)[0]}) != 1:
      return None
    tt = ts[sel] - tc
    ee = np.array([hist[i][2] for i in np.nonzero(sel)[0]])
    nn = np.array([hist[i][3] for i in np.nonzero(sel)[0]])
    ce = np.polyfit(tt, ee, 2)
    cn = np.polyfit(tt, nn, 2)
    ve, vn = ce[1], cn[1]
    if math.hypot(ve, vn) < 0.5:
      return None  # standing still: nothing to draw
    return float(ce[2]), float(cn[2]), math.atan2(vn, ve)

  def _record_stop(self, obs: Observation, e: float, n: float, h: float, model, stamp: float) -> None:
    self.stop = None
    if obs.stop_dist is None:
      return
    kind = "unknown"
    leads = list(model.leadsV3)
    if leads and leads[0].prob > 0.5 and len(leads[0].x) and leads[0].x[0] < obs.stop_dist + 4.0:
      kind = "lead"
    elif self.t is not None and self.t - self.objects_t < OBJECT_FRESH_S:
      near = [o for o in self.objects if o["kind"] in ("stopSign", "trafficLight") and abs(o["x"] - obs.stop_dist) < 20.0]
      if near:
        kind = min(near, key=lambda o: abs(o["x"] - obs.stop_dist))["kind"]
    self.stop = {"dist": round(obs.stop_dist, 1), "kind": kind}
    if kind == "lead" or self.pose.speed < 1.0:
      return  # only fixed stopping places go in the map
    y = float(np.interp(obs.stop_dist, obs.path[:, 0], obs.path[:, 1])) if len(obs.path) >= 2 else 0.0
    se, sn = car_to_en(obs.stop_dist, y, e, n, h)
    lat, lon = self.pose.frame.to_ll(float(se), float(sn))
    lat, lon = float(lat), float(lon)
    tile = self.store.get(geo.tile_key(lat, lon), create=True)
    if tile is None:
      return
    stops = tile.meta.setdefault("stops", [])
    m_lat, m_lon = geo.m_per_deg(lat)
    for s in stops:
      if math.hypot((s["lat"] - lat) * m_lat, (s["lon"] - lon) * m_lon) < STOP_MERGE_M:
        w = s["hits"]
        s["lat"] = (s["lat"] * w + lat) / (w + 1)
        s["lon"] = (s["lon"] * w + lon) / (w + 1)
        s["hits"] = min(w + 1, 1000)
        s["kinds"][kind] = s["kinds"].get(kind, 0) + 1
        break
    else:
      stops.append({"lat": lat, "lon": lon, "bearing": round(geo.bearing_from_heading(h), 1), "hits": 1, "kinds": {kind: 1}})
    tile.touch()

  # ---- for the viewer
  def stops_near(self, e: float, n: float, radius: float = 200.0) -> list[dict]:
    frame = self.pose.frame
    if frame is None:
      return []
    out = []
    for tile in self.store.loaded():
      for s in tile.meta.get("stops", []):
        se, sn = frame.to_en(s["lat"], s["lon"])
        if math.hypot(float(se) - e, float(sn) - n) < radius and s["hits"] >= 2:
          kinds = s.get("kinds", {})
          kind = max((k for k in kinds if k != "unknown"), key=lambda k: kinds[k], default="unknown")
          out.append({"e": round(float(se), 2), "n": round(float(sn), 2), "bearing": s.get("bearing", 0.0), "kind": kind, "hits": s["hits"]})
    return out

  def _snapshot(self, obs: Observation, v_ego: float) -> None:
    frame = self.pose.frame
    live: dict = {
      "t": round(self.t or 0.0, 2), "valid": self.pose.valid, "speed": round(v_ego, 2), "night": self.night,
      "model": {
        "lanes": [{"pts": _rounded(p, 2), "prob": round(pr, 2)} for p, pr in zip(obs.lanes, obs.lane_probs, strict=True)],
        "edges": [_rounded(p, 2) for p in obs.edges],
        "path": _rounded(obs.path, 2),
        "curvature": obs.curvature, "laneConf": round(obs.ego_lane_conf, 2), "laneChange": obs.lane_change,
      },
      "stop": self.stop,
      "objects": self.objects if self.t is not None and self.t - self.objects_t < 2.0 else [],
      "detector": self.detector_info,
      "steer": {"qualified": self.stats.qualified, "weight": round(self.steer.weight, 3), "reason": self.steer.reason,
                "comparedKm": round(self.stats.compared_km, 2), "agreedKm": round(self.stats.agreed_km, 2),
                "kmToGo": round(self.stats.km_to_go(), 2), "ratio": round(self.stats.ratio, 3)},
      "map": {"valid": self.map_path.valid, "path": _rounded(self.map_path.points), "conf": round(self.map_path.confidence, 2),
              "latError": None if self.lat_error is None else round(self.lat_error, 2),
              "passes": self.window.passes if self.window is not None else 0},
      "perf": {"worldMs": round(self.update_ms, 1), "tiles": len(self.store.keys()), "mapBytes": self.store.total_bytes},
    }
    if frame is not None:
      e, n, h = self.corrected()
      live["origin"] = {"id": self.pose.frame_id, "lat": frame.lat0, "lon": frame.lon0}
      live["pose"] = {"e": round(e, 2), "n": round(n, 2), "h": round(h, 4), "std": round(self.pose.pos_std, 2)}
      live["align"] = {"quality": round(self.align.quality, 2), "converged": self.align.converged(h),
                       "e": round(float(self.align.corr[0]), 2), "n": round(float(self.align.corr[1]), 2)}
      live["stops"] = self.stops_near(e, n)
    live["software"] = self.software
    self.live = live

  def refresh_idle_snapshot(self) -> None:
    """With no driving model running there are no frames to snapshot; still publish stats and software status."""
    live = dict(self.live)
    live["software"] = self.software
    live["perf"] = {**live.get("perf", {}), "tiles": len(self.store.keys()), "mapBytes": self.store.total_bytes}
    live["steer"] = {**live.get("steer", {}), "qualified": self.stats.qualified, "comparedKm": round(self.stats.compared_km, 2),
                     "agreedKm": round(self.stats.agreed_km, 2), "kmToGo": round(self.stats.km_to_go(), 2), "ratio": round(self.stats.ratio, 3)}
    self.live = live


def _resample(line: np.ndarray, x0: float, x1: float, step: float) -> np.ndarray:
  lo, hi = max(x0, line[0, 0]), min(x1, line[-1, 0])
  if hi - lo < step:
    return np.zeros((0, 2))
  xs = np.arange(lo, hi + 1e-6, step)
  return np.stack([xs, np.interp(xs, line[:, 0], line[:, 1])], axis=1)


def _probe_points(pts: np.ndarray, step: float, closed: bool) -> np.ndarray:
  """Points along (and for filled shapes, inside) a shape, close enough that no tile it touches is missed."""
  if len(pts) < 2:
    return pts
  ring = np.concatenate([pts, pts[:1]]) if closed else pts
  seg = np.diff(ring, axis=0)
  k = np.maximum((np.hypot(seg[:, 0], seg[:, 1]) // step).astype(np.int64), 1)
  idx = np.repeat(np.arange(len(seg)), k)
  frac = (np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k) + 1) / np.repeat(k, k)
  dense = np.concatenate([ring[:1], ring[idx] + seg[idx] * frac[:, None]])
  if closed:  # interior points too, for shapes bigger than a tile corner
    c = pts.mean(axis=0)
    dense = np.concatenate([dense, [c], (dense + c) / 2])
  return dense


def _stamp(tile, layer: str, cols: np.ndarray, rows: np.ndarray, kind: str, width: int):
  """Rasterize one shape (tile pixel coords) into a tile layer. Returns (rows, cols, mask) of what changed."""
  size = tile.layers[layer].shape[0]
  pad = width + 2
  c0 = int(max(np.floor(cols.min()) - pad, 0))
  r0 = int(max(np.floor(rows.min()) - pad, 0))
  c1 = int(min(np.ceil(cols.max()) + pad, size))
  r1 = int(min(np.ceil(rows.max()) + pad, size))
  if c1 <= c0 or r1 <= r0:
    return None
  mask = draw_mask(r1 - r0, c1 - c0, cols - c0, rows - r0, kind, width)
  if not mask.any():
    return None
  rs, cs = slice(r0, r1), slice(c0, c1)
  tile.add(layer, rs, cs, mask)
  return rs, cs, mask
