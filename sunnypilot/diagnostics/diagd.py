#!/usr/bin/env python3
"""
Diagnostics recorder: a running record of what went wrong and how hard the device was working.

One JSON-lines file per boot (see logs.py), downloadable from the Wi-Fi viewers. It records:

  header     version, device, and the settings that matter for troubleshooting
  events     whenever the set of active onroad events changes (commIssue, locationdTemporaryError, ...)
  alert      whenever the on-screen alert text changes
  error      every error-level log line from any process (crashes, "commIssue" with the late services, ...)
  process    a managed process that stopped while it should be running, with its exit code
  updater    updater state and last error, when they change
  snapshot   every couple of seconds while driving: CPU per core, temperatures, memory, GPU, driving and
             driver-monitoring model timing, locationd input health, world map and detector timing, and
             the processes using the most CPU

No GPS positions are recorded, so a log can be shared without giving away where the car goes.

It reads only the latest of each message, twice a second, so it costs next to nothing.
"""
import json
import os
import time

import cereal.messaging as messaging
from openpilot.common.params import Params

LOOP_S = 0.5
SNAPSHOT_ONROAD_S = 2.0
SNAPSHOT_OFFROAD_S = 15.0
UPDATER_CHECK_S = 10.0
TOP_PROCESSES = 8
WATCHED = ("modeld", "dmonitoringmodeld", "locationd", "calibrationd", "plannerd", "dmonitoringd", "selfdrived", "controlsd",
           "paramsd", "camerad", "ui", "worldd", "worldobjectsd", "diagd")
SETTINGS = ("WorldMapEnabled", "WorldMapDetector", "WorldMapSteer", "WorldMapQualified", "DisableUpdates", "AlphaLongitudinalEnabled",
            "DriveSightLink", "OffroadMode")
SERVICES = ['deviceState', 'managerState', 'procLog', 'onroadEvents', 'selfdriveState', 'carState', 'modelV2', 'driverStateV2',
            'livePose', 'worldMapSP', 'worldObjectsSP']
DROP_KEYS = ("ctx", "dongle_id", "serial", "imei", "latitude", "longitude", "lat", "lon", "gps")


def _short(s, n: int) -> str:
  s = str(s)
  return s if len(s) <= n else s[:n] + "..."


def sanitize_error(raw: str) -> dict:
  """An error log line, minus identifying context and anything location-like."""
  try:
    rec = json.loads(raw)
  except (TypeError, ValueError):
    return {"msg": _short(raw, 2000)}
  if not isinstance(rec, dict):
    return {"msg": _short(rec, 2000)}
  out = {k: rec[k] for k in ("levelname", "name", "filename", "funcName", "lineno") if k in rec}
  msg = rec.get("msg")
  if isinstance(msg, dict):
    msg = {k: v for k, v in msg.items() if not any(d in k.lower() for d in DROP_KEYS)}
    text = json.dumps(msg, default=str)
    out["msg"] = msg if len(text) <= 4000 else _short(text, 4000)
  elif msg is not None:
    out["msg"] = _short(msg, 2000)
  if rec.get("exc_info"):
    out["exc_info"] = _short(rec["exc_info"], 6000)
  return out


class ProcCpu:
  """Per-process CPU use (percent of one core) from consecutive procLog messages."""

  def __init__(self):
    self.last: dict[int, tuple[float, float]] = {}

  def update(self, t: float, procs) -> list[dict]:
    rows = []
    seen = {}
    for p in procs:
      name = p.name
      if len(p.cmdline) and p.cmdline[0].startswith("python"):
        mod = next((c for c in p.cmdline[1:] if not c.startswith("-")), "")
        name = mod.rsplit(".", 1)[-1] or name
      total = float(p.cpuUser) + float(p.cpuSystem)
      seen[p.pid] = (total, t)
      prev = self.last.get(p.pid)
      if prev is not None and t > prev[1]:
        rows.append({"name": _short(name, 32), "cpu": round(100.0 * (total - prev[0]) / (t - prev[1]), 1), "rssMB": round(p.memRss / 1e6, 1)})
    self.last = seen
    return rows


class DiagRecorder:
  def __init__(self, path: str):
    self.path = path
    self.f = open(path, "a", buffering=1, encoding="utf-8")
    self.events: tuple[str, ...] = ()
    self.alert: tuple[str, str] = ("", "")
    self.running: dict[str, bool] = {}
    self.updater: tuple[str, str] = ("", "")
    self.proc_rows: list[dict] = []

  def close(self) -> None:
    self.f.close()

  def write(self, t: float, kind: str, **data) -> None:
    self.f.write(json.dumps({"t": round(t, 2), "wall": round(time.time(), 1), "type": kind, **data}, default=str) + "\n")  # noqa: TID251

  def header(self, t: float, version: dict, settings: dict) -> None:
    self.write(t, "header", version=version, settings=settings)

  def on_events(self, t: float, events) -> None:
    names = tuple(sorted(str(e.name) for e in events))
    if names != self.events:
      self.write(t, "events", names=list(names), added=sorted(set(names) - set(self.events)), removed=sorted(set(self.events) - set(names)))
      self.events = names

  def on_alert(self, t: float, text1: str, text2: str, status: str) -> None:
    if (text1, text2) != self.alert:
      if text1 or text2:
        self.write(t, "alert", text1=text1, text2=text2, status=status)
      self.alert = (text1, text2)

  def on_error(self, t: float, raw: str) -> None:
    self.write(t, "error", **sanitize_error(raw))

  def on_manager(self, t: float, processes) -> None:
    for p in processes:
      was = self.running.get(p.name)
      if was and not p.running and p.shouldBeRunning:
        self.write(t, "process", name=p.name, event="stopped", exitCode=int(p.exitCode))
      elif was is False and p.running and p.shouldBeRunning:
        self.write(t, "process", name=p.name, event="started")
      self.running[p.name] = bool(p.running)

  def on_updater(self, t: float, state: str, error: str) -> None:
    if (state, error) != self.updater:
      self.write(t, "updater", state=state, error=_short(error, 2000))
      self.updater = (state, error)

  def snapshot(self, t: float, sm) -> None:
    ds = sm['deviceState']
    snap: dict = {
      "started": bool(ds.started),
      "cpu": list(ds.cpuUsagePercent),
      "cpuTempMax": round(max(ds.cpuTempC), 1) if len(ds.cpuTempC) else None,
      "gpuTempMax": round(max(ds.gpuTempC), 1) if len(ds.gpuTempC) else None,
      "maxTemp": round(ds.maxTempC, 1),
      "thermal": str(ds.thermalStatus),
      "memory": int(ds.memoryUsagePercent),
      "gpu": int(ds.gpuUsagePercent),
      "freeSpace": round(ds.freeSpacePercent, 1),
    }
    if sm.seen['carState']:
      snap["vEgo"] = round(sm['carState'].vEgo, 2)
    if sm.seen['selfdriveState']:
      snap["enabled"] = bool(sm['selfdriveState'].enabled)
    if sm.seen['modelV2']:
      m = sm['modelV2']
      snap["modelMs"] = round(m.modelExecutionTime * 1000, 1)
      snap["modelDrop"] = round(m.frameDropPerc, 2)
      snap["modelAlive"] = bool(sm.alive['modelV2'])
    if sm.seen['driverStateV2']:
      snap["dmMs"] = round(sm['driverStateV2'].modelExecutionTime * 1000, 1)
    if sm.seen['livePose']:
      lp = sm['livePose']
      snap["locationd"] = {"inputsOK": bool(lp.inputsOK), "posenetOK": bool(lp.posenetOK), "sensorsOK": bool(lp.sensorsOK)}
    if sm.seen['worldMapSP']:
      snap["worldMs"] = round(sm['worldMapSP'].processingTime * 1000, 1)
    if sm.seen['worldObjectsSP']:
      w = sm['worldObjectsSP']
      snap["detector"] = {"ready": bool(w.detectorReady), "ms": round(w.inferenceTime * 1000, 1), "hz": round(w.rateHz, 1), "skipped": int(w.skipped)}
    if self.proc_rows:
      top = sorted(self.proc_rows, key=lambda r: -r["cpu"])
      watched = [r for r in top if r["name"] in WATCHED]
      snap["procs"] = top[:TOP_PROCESSES] + [r for r in watched if r not in top[:TOP_PROCESSES]]
    self.write(t, "snapshot", **snap)


def version_info(params: Params) -> dict:
  info = {"branch": params.get("GitBranch") or "", "commit": (params.get("GitCommit") or "")[:12]}
  try:
    from openpilot.system.hardware import HARDWARE
    info["device"] = HARDWARE.get_device_type()
    info["os"] = HARDWARE.get_os_version()
  except Exception:
    pass
  return info


def main():
  try:
    os.nice(10)
  except (AttributeError, OSError):
    pass
  from openpilot.sunnypilot.diagnostics.logs import diag_dir, trim

  params = Params()
  root = diag_dir()
  os.makedirs(root, exist_ok=True)
  path = os.path.join(root, f"diag_{int(time.time())}.jsonl")  # noqa: TID251  a unique name per boot
  trim(root, keep=os.path.basename(path))
  rec = DiagRecorder(path)
  rec.header(time.monotonic(), version_info(params), {k: params.get_bool(k) for k in SETTINGS})

  sm = messaging.SubMaster(SERVICES)
  errors = messaging.sub_sock('errorLogMessage', conflate=False)
  procs = ProcCpu()
  last_snap = last_updater = 0.0
  try:
    while True:
      sm.update(0)
      t = time.monotonic()
      for ev in messaging.drain_sock(errors):
        rec.on_error(t, ev.errorLogMessage)
      if sm.updated['onroadEvents']:
        rec.on_events(t, sm['onroadEvents'])
      if sm.updated['selfdriveState']:
        ss = sm['selfdriveState']
        rec.on_alert(t, ss.alertText1, ss.alertText2, str(ss.alertStatus))
      if sm.updated['managerState']:
        rec.on_manager(t, sm['managerState'].processes)
      if sm.updated['procLog']:
        rec.proc_rows = procs.update(t, sm['procLog'].procs)
      if t - last_updater > UPDATER_CHECK_S:
        rec.on_updater(t, params.get("UpdaterState") or "", params.get("LastUpdateException") or "")
        last_updater = t
      started = sm.seen['deviceState'] and sm['deviceState'].started
      if sm.seen['deviceState'] and t - last_snap > (SNAPSHOT_ONROAD_S if started else SNAPSHOT_OFFROAD_S):
        rec.snapshot(t, sm)
        last_snap = t
      time.sleep(LOOP_S)
  finally:
    rec.close()


if __name__ == "__main__":
  main()
