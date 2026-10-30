import math
import os

import numpy as np

from openpilot.sunnypilot.worldmap import geo
from openpilot.sunnypilot.worldmap.tiles import Tile, TileStore


def test_tile_pixels_round_trip():
  for lat, lon in ((32.7674, -117.2058), (47.61, -122.33), (-33.86, 151.21), (64.84, -147.72)):
    key = geo.tile_key(lat, lon)
    col, row = geo.ll_to_tile_px(key, lat, lon)
    assert 0 <= col < geo.TILE_PX and 0 <= row < geo.TILE_PX
    lat2, lon2 = geo.tile_px_to_ll(key, col, row)
    assert abs(lat2 - lat) < 1e-9 and abs(lon2 - lon) < 1e-9


def test_tiles_are_about_256_meters():
  key = geo.tile_key(32.7674, -117.2058)
  sw, se, ne, _ = geo.tile_corners_ll(key)
  m_lat, m_lon = geo.m_per_deg(32.7674)
  assert abs((se[1] - sw[1]) * m_lon - geo.TILE_M) < 1.0
  assert geo.TILE_M - 3.0 < (ne[0] - se[0]) * m_lat <= geo.TILE_M  # rows are sized to never exceed TILE_M


def test_car_frame_round_trip_and_handedness():
  h = math.radians(30)
  e, n = geo.car_to_en(np.array([10.0]), np.array([2.0]), 5.0, -3.0, h)
  x, y = geo.en_to_car(e, n, 5.0, -3.0, h)
  assert np.allclose([x[0], y[0]], [10.0, 2.0])
  # facing north, a point to the right (y > 0) is to the east
  e, n = geo.car_to_en(0.0, 1.0, 0.0, 0.0, math.pi / 2)
  assert e > 0.99 and abs(n) < 1e-9


def test_group_by_tile_splits_points():
  lat = np.array([32.7674, 32.7674, 32.80])
  lon = np.array([-117.2058, -117.2057, -117.2058])
  groups = geo.group_by_tile(lat, lon)
  assert sum(len(v) for v in groups.values()) == 3
  assert len(groups) == 2


def test_tile_save_load_round_trip(tmp_path):
  tile = Tile((10, 20))
  mask = np.ones((4, 4), np.uint8)
  tile.add("road", slice(5, 9), slice(5, 9), mask, 3)
  tile.dir_e[6, 6] = 0.5
  tile.meta["stops"].append({"lat": 1.0, "lon": 2.0, "hits": 1, "kinds": {"stopSign": 1}})
  path = str(tmp_path / "t.npz")
  tile.save(path)
  back = Tile.load((10, 20), path)
  assert back.layers["road"][6, 6] == 3
  assert float(back.dir_e[6, 6]) == 0.5
  assert back.meta["stops"][0]["kinds"] == {"stopSign": 1}


def test_saturation_keeps_ratios():
  tile = Tile((0, 0))
  m = np.ones((1, 1), np.uint8)
  for _ in range(100):
    tile.add("edge", slice(0, 1), slice(0, 1), m, 1000)
    tile.add("seen", slice(0, 1), slice(0, 1), m, 2000)
  assert tile.layers["seen"][0, 0] <= 60_000
  assert abs(tile.ratio("edge")[0, 0] - 0.5) < 0.01


def test_prior_is_frozen_while_a_tile_is_being_drawn():
  tile = Tile((0, 0))
  m = np.ones((1, 1), np.uint8)
  tile.add("drive", slice(0, 1), slice(0, 1), m)
  assert tile.prior(1000.0)["drive"][0, 0] == 1
  tile.last_written = 1000.0
  tile.add("drive", slice(0, 1), slice(0, 1), m)
  assert tile.prior(1001.0)["drive"][0, 0] == 1       # drawn this pass: not used yet
  assert tile.prior(1000.0 + 3600)["drive"][0, 0] == 2  # a later pass sees it


def test_store_cleans_half_written_files_and_keeps_under_limit(tmp_path):
  store = TileStore(str(tmp_path), limit_bytes=10**9)
  for i in range(4):
    t = store.get((0, i), create=True)
    t.add("road", slice(0, 50), slice(0, 50), np.ones((50, 50), np.uint8))
  store.flush()
  assert len(os.listdir(store.dir)) == 4
  open(os.path.join(store.dir, "0_9.npz.tmp"), "w").write("partial")
  sizes = TileStore(str(tmp_path))._sizes
  assert len(sizes) == 4 and not os.path.exists(os.path.join(store.dir, "0_9.npz.tmp"))

  store = TileStore(str(tmp_path), limit_bytes=int(sum(sizes.values()) * 0.6), max_loaded=1)
  removed = store.enforce_limit()
  assert removed and store.total_bytes <= store.limit_bytes


def test_erase_all(tmp_path):
  store = TileStore(str(tmp_path))
  store.get((1, 1), create=True).touch()
  store.flush()
  store.erase_all()
  assert store.keys() == [] and os.listdir(store.dir) == []
