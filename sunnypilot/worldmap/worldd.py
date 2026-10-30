#!/usr/bin/env python3
"""
World map daemon.

Builds a map of the roads the car drives (see builder.py), publishes worldMapSP for controlsd and the UI, saves
the map every few seconds, and serves the live 3D viewer on the device's Wi-Fi (see server.py).

It runs whenever the device is on. With the car off (or in always-offroad mode) there is no driving model, so
it only serves the map that is already saved.
"""
import os
import threading
import time

import cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.worldmap.builder import WorldBuilder
from openpilot.sunnypilot.worldmap.qualify import QualifyStats
from openpilot.sunnypilot.worldmap.tiles import TileStore, worldmap_root

PARAMS_PERIOD = 2.0
STATS_PERIOD = 30.0
FLUSH_PERIOD = 10.0        # the device loses power with the car, so at most this much mapping is lost
SOFTWARE_PERIOD = 5.0
MODEL_FRAMES_PER_UPDATE = 4   # 5 map updates a second: plenty for mapping, and light on the CPU
SLOW_UPDATE_MS = 60.0         # above this (averaged), halve again to leave the CPU to driving


def publish(pm: messaging.PubMaster, builder: WorldBuilder, out: dict) -> None:
  msg = messaging.new_message('worldMapSP', valid=bool(out.get("valid", False)))
  w = msg.worldMapSP
  w.valid = bool(out.get("valid", False))
  if w.valid:
    w.latitude = out["lat"]
    w.longitude = out["lon"]
    w.bearingDeg = out["bearing"]
    w.positionStd = out["pos_std"]
    w.correctionLateral = out["corr_lat"]
    w.correctionLongitudinal = out["corr_long"]
    w.alignQuality = out["align_quality"]
    w.mapPathValid = out["map_valid"]
    w.mapPathX = out["map_x"]
    w.mapPathY = out["map_y"]
    w.mapCurvature = out["curvature"]
    w.curvatureDelta = out["curvature_delta"]
    w.mapConfidence = out["map_conf"]
    w.agreement = out["agreement"]
    w.lateralError = out["lat_error"]
    w.stopAhead = out["stop"]
    w.stopDistance = out["stop_dist"]
    w.stopKind = out["stop_kind"]
  w.steerQualified = builder.stats.qualified
  w.steerWeight = builder.steer.weight
  w.comparedKm = builder.stats.compared_km
  w.agreedKm = builder.stats.agreed_km
  w.night = builder.night
  w.processingTime = builder.update_ms / 1000.0
  w.tiles = len(builder.store.keys())
  w.mapBytes = builder.store.total_bytes
  pm.send('worldMapSP', msg)


def software_status(params: Params, sm: messaging.SubMaster) -> dict:
  """Branch and updater health for the viewer's Info tab, so a stuck software update can be seen from the iPad."""
  updated = None
  if sm.seen['managerState']:
    updated = next((p for p in sm['managerState'].processes if p.name == "updated"), None)
  last = params.get("LastUpdateTime")
  return {
    "branch": params.get("GitBranch") or "",
    "commit": (params.get("GitCommit") or "")[:7],
    "target": params.get("UpdaterTargetBranch") or "",
    "updaterRunning": None if updated is None else bool(updated.running),
    "updaterExit": None if updated is None or updated.running else int(updated.exitCode),
    "state": params.get("UpdaterState") or "",
    "disabled": params.get_bool("DisableUpdates"),
    "failed": int(params.get("UpdateFailedCount") or 0),
    "error": (params.get("LastUpdateException") or "")[-400:],
    "downloadReady": params.get_bool("UpdaterFetchAvailable"),
    "installReady": params.get_bool("UpdateAvailable"),
    "lastUpdate": last.isoformat(timespec="minutes") if last is not None else "",
  }


def save_stats(params: Params, stats: QualifyStats) -> None:
  params.put("WorldMapStats", stats.to_dict())
  params.put_bool("WorldMapQualified", stats.qualified)


def main():
  try:
    os.nice(10)  # never compete with driving processes
  except (AttributeError, OSError):
    pass

  params = Params()
  store = TileStore(worldmap_root())
  stats = QualifyStats.from_dict(params.get("WorldMapStats"))
  builder = WorldBuilder(store, stats)

  from openpilot.sunnypilot.worldmap.server import start_server
  start_server(builder, store, on_erase=lambda: save_stats(params, builder.stats))

  stop = threading.Event()

  def flusher():
    while not stop.wait(FLUSH_PERIOD):
      try:
        store.flush()
      except Exception:
        cloudlog.exception("worldmap: flush failed")
  threading.Thread(target=flusher, daemon=True).start()

  pm = messaging.PubMaster(['worldMapSP'])
  sm = messaging.SubMaster(['modelV2', 'carState', 'livePose', 'gpsLocationExternal', 'liveLocationKalman', 'worldObjectsSP',
                            'managerState'], poll='modelV2')
  steer_enabled = False
  last_params = 0.0
  last_stats = time.monotonic()
  last_software = 0.0
  last_model = -1e9
  v_ego = 0.0
  every = MODEL_FRAMES_PER_UPDATE
  slow_ms = 0.0

  cloudlog.warning(f"worldmap: {len(store.keys())} tiles, {store.total_bytes / 1e6:.0f} MB")
  try:
    while True:
      sm.update(1000)
      now = time.monotonic()
      if now - last_params > PARAMS_PERIOD:
        steer_enabled = params.get_bool("WorldMapSteer")
        last_params = now
      if now - last_software > SOFTWARE_PERIOD:
        builder.software = software_status(params, sm)
        if now - last_model > 2.0:
          builder.refresh_idle_snapshot()  # parked or always-offroad: keep the viewer's Info tab current
        last_software = now

      if sm.updated['carState']:
        v_ego = sm['carState'].vEgo
      if sm.updated['gpsLocationExternal']:
        builder.on_gps(sm.logMonoTime['gpsLocationExternal'] * 1e-9, sm['gpsLocationExternal'])
      if sm.updated['liveLocationKalman']:
        builder.on_llk(sm.logMonoTime['liveLocationKalman'] * 1e-9, sm['liveLocationKalman'])
      if sm.updated['worldObjectsSP']:
        builder.on_objects(sm.logMonoTime['worldObjectsSP'] * 1e-9, sm['worldObjectsSP'])
      if not sm.updated['modelV2']:
        continue
      last_model = now

      t = sm.logMonoTime['modelV2'] * 1e-9
      yaw_rate = sm['livePose'].angularVelocityDevice.z if sm.seen['livePose'] else 0.0
      builder.on_motion(t, v_ego, yaw_rate)
      if sm['modelV2'].frameId % every:
        continue

      try:
        out = builder.on_model(t, sm['modelV2'], v_ego, steer_enabled)
      except Exception:
        cloudlog.exception("worldmap: update failed")
        out = {"valid": False}
      publish(pm, builder, out)

      slow_ms = 0.95 * slow_ms + 0.05 * builder.update_ms
      every = 2 * MODEL_FRAMES_PER_UPDATE if slow_ms > SLOW_UPDATE_MS else MODEL_FRAMES_PER_UPDATE

      if builder.stats_changed or now - last_stats > STATS_PERIOD:
        save_stats(params, builder.stats)
        builder.stats_changed = False
        last_stats = now
  finally:
    stop.set()
    store.flush()
    save_stats(params, builder.stats)


if __name__ == "__main__":
  main()
