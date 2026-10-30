"""
Coordinates for the world map.

Two kinds of flat frames are used, both east/north in meters:

* a local frame around a moving origin, for live work (pose filter, viewer). It is re-anchored when the car
  gets far from the origin, so the flat-earth error stays tiny.
* a tile frame for storage. The world is cut into rows of constant latitude span and, within a row, columns
  of constant longitude span, about 256 m on a side. A tile's frame starts at its south-west corner and uses
  the scale at the middle of its row, so a stored tile never depends on where the car was when it was saved.

The car frame matches openpilot's calibrated frame: x forward, y right. A heading `h` is the angle of the car's
forward axis, counter-clockwise from east.
"""
import math

import numpy as np

TILE_M = 256.0             # nominal tile size
CELL_M = 0.5               # map resolution
TILE_PX = int(TILE_M / CELL_M)
DLAT = TILE_M / 111_700.0  # rows are this many degrees tall: never more than TILE_M at any latitude


def m_per_deg(lat: float) -> tuple[float, float]:
  """Meters per degree of latitude and of longitude at `lat` (WGS84 series)."""
  p = math.radians(lat)
  m_lat = 111_132.92 - 559.82 * math.cos(2 * p) + 1.175 * math.cos(4 * p)
  m_lon = 111_412.84 * math.cos(p) - 93.5 * math.cos(3 * p)
  return m_lat, m_lon


def heading_from_bearing(bearing_deg: float) -> float:
  return math.radians(90.0 - bearing_deg)


def bearing_from_heading(h: float) -> float:
  return (90.0 - math.degrees(h)) % 360.0


def wrap(a):
  return (a + np.pi) % (2 * np.pi) - np.pi


class LocalFrame:
  """Flat east/north frame around an origin."""

  def __init__(self, lat: float, lon: float):
    self.lat0, self.lon0 = lat, lon
    self.m_lat, self.m_lon = m_per_deg(lat)

  def to_en(self, lat, lon):
    return (np.asarray(lon) - self.lon0) * self.m_lon, (np.asarray(lat) - self.lat0) * self.m_lat

  def to_ll(self, e, n):
    return self.lat0 + np.asarray(n) / self.m_lat, self.lon0 + np.asarray(e) / self.m_lon


def car_to_en(x, y, e0: float, n0: float, h: float):
  """Car frame points (x forward, y right) to east/north around a pose."""
  c, s = math.cos(h), math.sin(h)
  x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
  return e0 + x * c + y * s, n0 + x * s - y * c


def en_to_car(e, n, e0: float, n0: float, h: float):
  c, s = math.cos(h), math.sin(h)
  de, dn = np.asarray(e) - e0, np.asarray(n) - n0
  return de * c + dn * s, de * s - dn * c


def _row_scale(row: int) -> tuple[float, float, float]:
  """(m per deg lat, m per deg lon, deg lon per column) for a tile row."""
  m_lat, m_lon = m_per_deg((row + 0.5) * DLAT)
  return m_lat, m_lon, TILE_M / m_lon


def tile_key(lat: float, lon: float) -> tuple[int, int]:
  row = math.floor(lat / DLAT)
  dlon = _row_scale(row)[2]
  return row, math.floor(lon / dlon)


def tile_origin(key: tuple[int, int]) -> tuple[float, float]:
  row, col = key
  return row * DLAT, col * _row_scale(row)[2]


def tile_corners_ll(key: tuple[int, int]) -> list[tuple[float, float]]:
  """South-west, south-east, north-east, north-west corners as (lat, lon)."""
  lat0, lon0 = tile_origin(key)
  dlon = _row_scale(key[0])[2]
  return [(lat0, lon0), (lat0, lon0 + dlon), (lat0 + DLAT, lon0 + dlon), (lat0 + DLAT, lon0)]


def ll_to_tile_px(key: tuple[int, int], lat, lon):
  """Fractional (col, row) pixel of lat/lon inside a tile image. Image row 0 is the tile's north edge."""
  lat0, lon0 = tile_origin(key)
  m_lat, m_lon, _ = _row_scale(key[0])
  e = (np.asarray(lon) - lon0) * m_lon
  n = (np.asarray(lat) - lat0) * m_lat
  return e / CELL_M, (TILE_M - n) / CELL_M


def tile_px_to_ll(key: tuple[int, int], col, row):
  lat0, lon0 = tile_origin(key)
  m_lat, m_lon, _ = _row_scale(key[0])
  e = np.asarray(col) * CELL_M
  n = TILE_M - np.asarray(row) * CELL_M
  return lat0 + n / m_lat, lon0 + e / m_lon


def group_by_tile(lat: np.ndarray, lon: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
  """Indices of points per tile key."""
  rows = np.floor(lat / DLAT).astype(np.int64)
  groups: dict[tuple[int, int], np.ndarray] = {}
  for row in np.unique(rows):
    in_row = np.nonzero(rows == row)[0]
    dlon = _row_scale(int(row))[2]
    cols = np.floor(lon[in_row] / dlon).astype(np.int64)
    for col in np.unique(cols):
      groups[(int(row), int(col))] = in_row[cols == col]
  return groups
