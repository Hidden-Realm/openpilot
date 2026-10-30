import io
import json
import os
import zipfile
from types import SimpleNamespace as NS

from openpilot.sunnypilot.diagnostics import logs
from openpilot.sunnypilot.diagnostics.diagd import DiagRecorder, ProcCpu, sanitize_error


def read(path):
  with open(path, encoding="utf-8") as f:
    return [json.loads(line) for line in f]


def test_bundle_and_trim(tmp_path, monkeypatch):
  root = str(tmp_path)
  for i in range(25):
    p = os.path.join(root, f"diag_{i}.jsonl")
    open(p, "w").write("x" * 100)
    os.utime(p, (1000 + i, 1000 + i))
  logs.trim(root, keep="diag_0.jsonl")
  names = [n for n, _, _ in logs.list_logs(root)]
  assert len(names) == logs.MAX_FILES + 1 and "diag_0.jsonl" in names and "diag_24.jsonl" == names[0]
  z = zipfile.ZipFile(io.BytesIO(logs.bundle(root, max_files=3)))
  assert z.namelist() == ["diag_24.jsonl", "diag_23.jsonl", "diag_22.jsonl"]
  empty = zipfile.ZipFile(io.BytesIO(logs.bundle(str(tmp_path / "none"))))
  assert empty.namelist() == ["README.txt"]


def test_errors_lose_identifying_and_location_data():
  raw = json.dumps({"levelname": "ERROR", "filename": "selfdrived.py", "ctx": {"dongle_id": "abc", "version": "1"},
                    "msg": {"event": "commIssue", "invalid": ["liveCalibration"], "latitude": 32.7, "lon": -117.2}})
  out = sanitize_error(raw)
  assert out["msg"] == {"event": "commIssue", "invalid": ["liveCalibration"]}
  assert "ctx" not in out and "abc" not in json.dumps(out)
  assert sanitize_error("not json")["msg"] == "not json"


def test_recorder_writes_changes_only(tmp_path):
  path = str(tmp_path / "d.jsonl")
  rec = DiagRecorder(path)
  rec.header(0.0, {"commit": "abc"}, {"WorldMapDetector": False})
  ev = lambda *names: [NS(name=n) for n in names]  # noqa: E731
  rec.on_events(1.0, ev("commIssue"))
  rec.on_events(1.5, ev("commIssue"))                      # unchanged: not written again
  rec.on_events(2.0, ev("commIssue", "locationdTemporaryError"))
  rec.on_alert(2.0, "Communication Issue", "liveCalibration, ...", "normal")
  rec.on_alert(2.5, "Communication Issue", "liveCalibration, ...", "normal")
  proc = lambda name, running, should=True, code=0: NS(name=name, running=running, shouldBeRunning=should, exitCode=code)  # noqa: E731
  rec.on_manager(3.0, [proc("worldd", True)])
  rec.on_manager(4.0, [proc("worldd", False, code=1)])
  rec.on_manager(5.0, [proc("worldd", True)])
  rec.close()
  rows = read(path)
  kinds = [r["type"] for r in rows]
  assert kinds == ["header", "events", "events", "alert", "process", "process"]
  assert rows[2]["added"] == ["locationdTemporaryError"]
  assert rows[4] == {**rows[4], "name": "worldd", "event": "stopped", "exitCode": 1}
  assert rows[5]["event"] == "started"


def test_process_cpu_from_proclog():
  pc = ProcCpu()
  p = lambda pid, user, name="x", cmd=(): NS(pid=pid, name=name, cmdline=list(cmd), cpuUser=user, cpuSystem=0.0, memRss=50e6)  # noqa: E731
  assert pc.update(0.0, [p(1, 10.0, cmd=("python3", "-m", "openpilot.sunnypilot.worldmap.worldd"))]) == []
  rows = pc.update(2.0, [p(1, 11.0, cmd=("python3", "-m", "openpilot.sunnypilot.worldmap.worldd"))])
  assert rows == [{"name": "worldd", "cpu": 50.0, "rssMB": 50.0}]


def test_snapshot(tmp_path):
  path = str(tmp_path / "s.jsonl")
  rec = DiagRecorder(path)
  rec.proc_rows = [{"name": "modeld", "cpu": 80.0, "rssMB": 1}, {"name": "worldd", "cpu": 3.0, "rssMB": 1}]
  data = {
    "deviceState": NS(started=True, cpuUsagePercent=[10, 20], cpuTempC=[50.0, 61.5], gpuTempC=[55.0], maxTempC=62.0,
                      thermalStatus="green", memoryUsagePercent=40, gpuUsagePercent=70, freeSpacePercent=80.0),
    "carState": NS(vEgo=12.0), "selfdriveState": NS(enabled=False),
    "modelV2": NS(modelExecutionTime=0.031, frameDropPerc=1.5), "driverStateV2": NS(modelExecutionTime=0.02),
    "livePose": NS(inputsOK=False, posenetOK=True, sensorsOK=True),
  }
  class FakeSM:
    seen = {k: k in data for k in ("carState", "selfdriveState", "modelV2", "driverStateV2", "livePose", "worldMapSP", "worldObjectsSP")}
    alive = {"modelV2": True}

    def __getitem__(self, k):
      return data[k]

  rec.snapshot(10.0, FakeSM())
  rec.close()
  snap = read(path)[0]
  assert snap["type"] == "snapshot" and snap["cpuTempMax"] == 61.5 and snap["modelMs"] == 31.0
  assert snap["locationd"]["inputsOK"] is False and snap["procs"][0]["name"] == "modeld"
  assert "latitude" not in json.dumps(snap)
