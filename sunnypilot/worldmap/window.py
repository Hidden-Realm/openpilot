"""
A working copy of the map around the car.

Alignment and path following need many lookups near the car, and tiles have their own frames. So every couple
of seconds the stored map around the car is resampled into one grid in the live local frame (east/north,
0.5 m cells, row 0 at the north edge), and the slow products are derived once:

  structure  how often lane lines or road edges were seen in a cell, 0..1
  dist       meters to the nearest strong structure, for alignment
  heat       smoothed count of passes the car drove over a cell, for path following
  dir_e/n    the average driving direction there (unit vector)
"""
import cv2
import numpy as np

from openpilot.sunnypilot.worldmap.geo import CELL_M, LocalFrame, group_by_tile, ll_to_tile_px, DLAT
from openpilot.sunnypilot.worldmap.tiles import TileStore

WINDOW_M = 192.0
WINDOW_PX = int(WINDOW_M / CELL_M)
STRUCT_MIN = 0.2
DIST_CAP = 8.0
HEAT_SIGMA = 1.5                                     # cells
PASS_HEAT = 1.0 / (np.sqrt(2 * np.pi) * HEAT_SIGMA)  # ridge height left by one pass of a one-cell trail


class MapWindow:
  def __init__(self, store: TileStore, frame: LocalFrame, frame_id: int, center_e: float, center_n: float, now: float,
               car_e: float, car_n: float):
    self.now = now
    self.frame = frame
    self.frame_id = frame_id
    self.e0 = center_e - WINDOW_M / 2   # west edge
    self.n0 = center_n + WINDOW_M / 2   # north edge
    self.built_at = (car_e, car_n)       # the window sits ahead of the car; it is stale once the car has moved on
    shape = (WINDOW_PX, WINDOW_PX)
    self.structure = np.zeros(shape, np.float32)
    self.seen = np.zeros(shape, np.float32)
    self.drive = np.zeros(shape, np.float32)
    self.dir_e = np.zeros(shape, np.float32)
    self.dir_n = np.zeros(shape, np.float32)
    self.passes = 0
    self.tiles = 0
    self._fill(store)
    self._derive()

  def _fill(self, store: TileStore) -> None:
    cols_e = self.e0 + (np.arange(WINDOW_PX) + 0.5) * CELL_M
    rows_n = self.n0 - (np.arange(WINDOW_PX) + 0.5) * CELL_M
    lat_rows, _ = self.frame.to_ll(np.zeros_like(rows_n), rows_n)
    _, lon_cols = self.frame.to_ll(cols_e, np.zeros_like(cols_e))
    lat_rows = np.asarray(lat_rows)
    lon_cols = np.asarray(lon_cols)

    # which tiles touch the window: sample a coarse grid of the window
    step = 16
    glat = np.repeat(lat_rows[::step], len(lon_cols[::step]))
    glon = np.tile(lon_cols[::step], len(lat_rows[::step]))
    keys = list(group_by_tile(glat, glon).keys())
    passes = []
    for key in keys:
      tile = store.get(key)
      if tile is None:
        continue
      self.tiles += 1
      passes.append(tile.passes)
      # window rows/cols that fall inside this tile
      row_in = np.nonzero(np.floor(lat_rows / DLAT).astype(np.int64) == key[0])[0]
      if not len(row_in):
        continue
      tcol, _ = ll_to_tile_px(key, np.full(len(lon_cols), lat_rows[row_in[0]]), lon_cols)
      _, trow = ll_to_tile_px(key, lat_rows[row_in], np.full(len(row_in), lon_cols[0]))
      col_in = np.nonzero((tcol >= -0.5) & (tcol < tile.layers["seen"].shape[1] - 0.5))[0]
      if not len(col_in):
        continue
      r0, r1, c0, c1 = row_in[0], row_in[-1] + 1, col_in[0], col_in[-1] + 1
      # only the part of the tile under the window is converted and resampled
      size = tile.layers["seen"].shape[0]
      sc0 = int(max(np.floor(tcol[c0:c1].min()) - 1, 0))
      sc1 = int(min(np.ceil(tcol[c0:c1].max()) + 2, size))
      sr0 = int(max(np.floor(trow.min()) - 1, 0))
      sr1 = int(min(np.ceil(trow.max()) + 2, size))
      map_x = np.broadcast_to(tcol[c0:c1][None, :] - 0.5 - sc0, (r1 - r0, c1 - c0)).astype(np.float32)
      map_y = np.broadcast_to(trow[:, None] - 0.5 - sr0, (r1 - r0, c1 - c0)).astype(np.float32)
      src = (slice(sr0, sr1), slice(sc0, sc1))

      def sample(arr, src=src, map_x=map_x, map_y=map_y):
        return cv2.remap(arr[src].astype(np.float32), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

      prior = tile.prior(self.now)
      seen = sample(prior["seen"])
      self.seen[r0:r1, c0:c1] = seen
      self.structure[r0:r1, c0:c1] = np.where(seen >= 2, np.minimum(sample(prior["struct"]) / np.maximum(seen, 1), 1.0), 0)
      self.drive[r0:r1, c0:c1] = sample(prior["drive"])
      self.dir_e[r0:r1, c0:c1] = sample(prior["dir_e"])
      self.dir_n[r0:r1, c0:c1] = sample(prior["dir_n"])
    self.passes = max(passes) if passes else 0

  def _derive(self) -> None:
    strong = (self.structure >= STRUCT_MIN).astype(np.uint8)
    self.has_structure = int(strong.sum())
    if self.has_structure:
      self.dist = cv2.distanceTransform(1 - strong, cv2.DIST_L2, 3).astype(np.float32) * CELL_M
      np.minimum(self.dist, DIST_CAP, out=self.dist)
    else:
      self.dist = np.full(self.structure.shape, DIST_CAP, np.float32)
    self.heat = cv2.GaussianBlur(self.drive, (0, 0), HEAT_SIGMA) / PASS_HEAT   # about 1.0 per pass on the ridge top
    de = cv2.GaussianBlur(self.dir_e, (0, 0), HEAT_SIGMA)
    dn = cv2.GaussianBlur(self.dir_n, (0, 0), HEAT_SIGMA)
    norm = np.hypot(de, dn)
    dir_ok = norm > 0.3 * PASS_HEAT   # a consistent direction from at least about a third of a pass
    norm = np.maximum(norm, 1e-6)
    # heat, direction and direction-ok in one array so path following does a single lookup per step
    self.ridge = np.stack([self.heat, de / norm, dn / norm, dir_ok.astype(np.float32)], axis=-1)

  # ---- lookups
  def px(self, e, n):
    """Fractional (col, row) of east/north points."""
    return (np.asarray(e) - self.e0) / CELL_M - 0.5, (self.n0 - np.asarray(n)) / CELL_M - 0.5

  def inside(self, e, n, margin_m: float = 0.0) -> np.ndarray:
    c, r = self.px(e, n)
    m = margin_m / CELL_M
    return (c >= m) & (r >= m) & (c < WINDOW_PX - 1 - m) & (r < WINDOW_PX - 1 - m)

  def ridge_at(self, e: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Bilinear lookup of (heat, dir_e, dir_n, dir_ok) for 1-D points; zeros outside the window."""
    c, r = self.px(e, n)
    ok = (c >= 0) & (r >= 0) & (c < WINDOW_PX - 1) & (r < WINDOW_PX - 1)
    out = np.zeros((len(c), 4), np.float32)
    c0 = np.floor(c[ok]).astype(np.int64)
    r0 = np.floor(r[ok]).astype(np.int64)
    fc = (c[ok] - c0)[:, None]
    fr = (r[ok] - r0)[:, None]
    g = self.ridge
    out[ok] = (g[r0, c0] * (1 - fc) * (1 - fr) + g[r0, c0 + 1] * fc * (1 - fr) +
               g[r0 + 1, c0] * (1 - fc) * fr + g[r0 + 1, c0 + 1] * fc * fr)
    return out

  def lookup(self, arr: np.ndarray, e, n, default: float = 0.0) -> np.ndarray:
    c, r = self.px(e, n)
    ci = np.rint(c).astype(np.int64)
    ri = np.rint(r).astype(np.int64)
    ok = (ci >= 0) & (ri >= 0) & (ci < WINDOW_PX) & (ri < WINDOW_PX)
    out = np.full(np.shape(ci), default, np.float32)
    out[ok] = arr[ri[ok], ci[ok]]
    return out

  def stale(self, frame_id: int, e: float, n: float, max_offset: float = 32.0) -> bool:
    return frame_id != self.frame_id or abs(e - self.built_at[0]) > max_offset or abs(n - self.built_at[1]) > max_offset
