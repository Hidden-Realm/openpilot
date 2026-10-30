"""
Map tiles on disk.

Each tile is a 512x512 grid of 0.5 m cells holding evidence counts:

  seen   times the cell was inside the camera's view while mapping
  road   times it was drivable road
  edge   times a road edge crossed it
  lane   times a lane line crossed it
  drive  times the car drove over it; dir_e/dir_n sum its driving direction
  alt    summed altitude where the car drove (for hills in the viewer)

Ratios like edge/seen are what matter, so a cell that keeps being seen without an edge fades on its own. When
counts near the uint16 limit, the whole tile is halved, which keeps the ratios.

The device loses power the moment the car turns off, so tiles are written often, to a temp file, then
renamed into place: a tile on disk is always a whole one. Old tiles are removed first when the map passes its
size limit.
"""
import json
import os
import threading
import time
from collections import OrderedDict

import numpy as np

from openpilot.sunnypilot.worldmap.geo import TILE_PX

LAYERS = ("seen", "road", "edge", "lane", "drive")
FORMAT_VERSION = 1
SATURATE = 60_000
DEFAULT_LIMIT_BYTES = 10 * 1024 ** 3
NEW_PASS_SECONDS = 600  # coming back to a tile after this long counts as another pass


def worldmap_root() -> str:
  from openpilot.system.hardware.hw import Paths
  return os.path.join(os.path.dirname(os.path.normpath(Paths.log_root())), "worldmap")


class Tile:
  def __init__(self, key: tuple[int, int]):
    self.key = key
    self.layers = {name: np.zeros((TILE_PX, TILE_PX), np.uint16) for name in LAYERS}
    self.dir_e = np.zeros((TILE_PX, TILE_PX), np.float16)  # half precision: direction only needs to be roughly right
    self.dir_n = np.zeros((TILE_PX, TILE_PX), np.float16)
    self.alt = np.zeros((TILE_PX, TILE_PX), np.float32)
    self.meta: dict = {"passes": 0, "last_seen": 0.0, "stops": [], "version": FORMAT_VERSION}
    self.revision = 0  # bumped on every change; caches key off it
    self.dirty = False
    self._prior: dict[str, np.ndarray] | None = None
    self._prior_rev = -1
    self.last_written = 0.0  # wall clock of the last fusion into this tile (memory only)

  def prior(self, now: float) -> dict[str, np.ndarray]:
    """The tile as it was before this pass. Alignment and path following use this copy, so a drive never
    lines up with, or follows, what it drew itself a moment ago. The copy is only refreshed once nothing
    has been written to the tile for a while, so everything in it comes from earlier passes."""
    if self._prior is None or (self._prior_rev != self.revision and now - self.last_written > NEW_PASS_SECONDS):
      self.refresh_prior()
    assert self._prior is not None
    return self._prior

  def refresh_prior(self) -> None:
    self._prior_rev = self.revision
    self._prior = {
      "seen": self.layers["seen"].copy(),
      "struct": np.maximum(self.layers["edge"], self.layers["lane"]),
      "drive": self.layers["drive"].copy(),
      "dir_e": self.dir_e.copy(),
      "dir_n": self.dir_n.copy(),
    }

  def add(self, name: str, rows: slice, cols: slice, mask: np.ndarray, weight: int = 1) -> None:
    layer = self.layers[name][rows, cols]
    total = layer.astype(np.int32) + mask.astype(np.int32) * weight
    np.minimum(total, np.iinfo(np.uint16).max, out=total)  # never wrap around
    layer[:] = total
    if total.max(initial=0) > SATURATE:
      self._halve()
    self.touch()

  def touch(self) -> None:
    self.revision += 1
    self.dirty = True

  def _halve(self) -> None:
    for layer in self.layers.values():
      layer //= 2
    self.dir_e *= 0.5
    self.dir_n *= 0.5
    self.alt *= 0.5

  def ratio(self, name: str, min_seen: int = 2) -> np.ndarray:
    seen = self.layers["seen"].astype(np.float32)
    out = self.layers[name].astype(np.float32) / np.maximum(seen, 1.0)
    out[seen < min_seen] = 0.0
    return np.minimum(out, 1.0)

  def note_visit(self, now: float) -> None:
    if now - self.meta.get("last_seen", 0.0) > NEW_PASS_SECONDS:
      self.meta["passes"] = self.meta.get("passes", 0) + 1
    self.meta["last_seen"] = now
    self.dirty = True

  @property
  def passes(self) -> int:
    return int(self.meta.get("passes", 0))

  def save(self, path: str) -> int:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
      np.savez_compressed(f, dir_e=self.dir_e, dir_n=self.dir_n, alt=self.alt,
                          meta=np.array(json.dumps(self.meta)), **self.layers)
      f.flush()
      os.fsync(f.fileno())
    os.replace(tmp, path)
    self.dirty = False
    return os.path.getsize(path)

  @classmethod
  def load(cls, key: tuple[int, int], path: str) -> "Tile":
    tile = cls(key)
    with np.load(path) as data:
      for name in LAYERS:
        tile.layers[name][:] = data[name]
      tile.dir_e[:] = data["dir_e"]
      tile.dir_n[:] = data["dir_n"]
      tile.alt[:] = data["alt"]
      tile.meta.update(json.loads(str(data["meta"])))
    return tile


class TileStore:
  """Tiles in memory (a small LRU) backed by files under `root`."""

  def __init__(self, root: str, limit_bytes: int = DEFAULT_LIMIT_BYTES, max_loaded: int = 9):
    self.root = root
    self.dir = os.path.join(root, "tiles")
    os.makedirs(self.dir, exist_ok=True)
    self.limit_bytes = limit_bytes
    self.max_loaded = max_loaded
    self._tiles: OrderedDict[tuple[int, int], Tile] = OrderedDict()
    self._lock = threading.RLock()
    self._sizes: dict[tuple[int, int], int] = {}
    self._scan()

  def path(self, key: tuple[int, int]) -> str:
    return os.path.join(self.dir, f"{key[0]}_{key[1]}.npz")

  def _scan(self) -> None:
    for name in os.listdir(self.dir):
      if name.endswith(".tmp"):
        try:
          os.remove(os.path.join(self.dir, name))  # half-written when power was cut
        except OSError:
          pass
        continue
      key = parse_key(name)
      if key is not None:
        try:
          self._sizes[key] = os.path.getsize(os.path.join(self.dir, name))
        except OSError:
          pass

  @property
  def total_bytes(self) -> int:
    return sum(self._sizes.values())

  def keys(self) -> list[tuple[int, int]]:
    with self._lock:
      return list(set(self._sizes) | set(self._tiles))

  def get(self, key: tuple[int, int], create: bool = False) -> Tile | None:
    with self._lock:
      tile = self._tiles.get(key)
      if tile is not None:
        self._tiles.move_to_end(key)
        return tile
      on_disk = key in self._sizes
    # read the file without holding the lock, so a prefetch never blocks the main loop
    loaded = None
    if on_disk:
      try:
        loaded = Tile.load(key, self.path(key))
      except (OSError, ValueError, KeyError):
        loaded = None  # unreadable file: start the tile over if asked to
    with self._lock:
      tile = self._tiles.get(key)  # someone else may have loaded it meanwhile
      if tile is None:
        tile = loaded if loaded is not None else (Tile(key) if create else None)
        if tile is None:
          return None
        self._tiles[key] = tile
        while len(self._tiles) > self.max_loaded:
          _, old = self._tiles.popitem(last=False)
          if old.dirty:
            self._save(old)
      self._tiles.move_to_end(key)
      return tile

  def cached(self, key: tuple[int, int]) -> Tile | None:
    with self._lock:
      return self._tiles.get(key)

  def loaded(self) -> list[Tile]:
    with self._lock:
      return list(self._tiles.values())

  def _save(self, tile: Tile) -> None:
    try:
      self._sizes[tile.key] = tile.save(self.path(tile.key))
    except OSError:
      pass

  def flush(self) -> int:
    """Write every changed tile. Returns how many were written."""
    with self._lock:
      dirty = [t for t in self._tiles.values() if t.dirty]
      for tile in dirty:
        self._save(tile)
    self.enforce_limit()
    return len(dirty)

  def enforce_limit(self) -> list[tuple[int, int]]:
    """Remove the least recently driven tiles until the map fits its limit."""
    removed: list[tuple[int, int]] = []
    with self._lock:
      if self.total_bytes <= self.limit_bytes:
        return removed
      by_age = sorted(self._sizes, key=lambda k: self._mtime(k))
      for key in by_age:
        if self.total_bytes <= self.limit_bytes * 0.95:
          break
        if key in self._tiles:
          continue  # in use right now
        try:
          os.remove(self.path(key))
        except OSError:
          pass
        self._sizes.pop(key, None)
        removed.append(key)
    return removed

  def _mtime(self, key: tuple[int, int]) -> float:
    try:
      return os.path.getmtime(self.path(key))
    except OSError:
      return 0.0

  def erase_all(self) -> None:
    with self._lock:
      self._tiles.clear()
      for key in list(self._sizes):
        try:
          os.remove(self.path(key))
        except OSError:
          pass
      self._sizes.clear()


def parse_key(name: str) -> tuple[int, int] | None:
  if not name.endswith(".npz"):
    return None
  try:
    row, col = name[:-4].split("_")
    return int(row), int(col)
  except ValueError:
    return None


def now() -> float:
  return time.time()  # noqa: TID251  wall clock, stored with tiles to age them
