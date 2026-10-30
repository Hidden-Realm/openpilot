#!/usr/bin/env python3
"""
Camera detector for the world map: people, bikes, vehicles, traffic lights and stop signs.

A small YOLO11n network (compiled for the GPU at build time, like the driver monitoring model) looks at the
road and wide cameras in turn, about twice a second. It is off unless turned on in device settings: it shares
the GPU with the driving and driver monitoring models, and a run holds the GPU until it finishes, so it can
make their frames late. When it does run, they always come first:

* a run only starts right after a driving-model frame, so they don't overlap on the GPU, and
* if either model reports slow frames, or the driving model drops frames, runs are skipped and the pause grows
  until they recover.

Detections are placed in the car frame (ranging.py), tracked between runs (tracker.py) and published as
worldObjectsSP. At night the frames are brightened first (yolo.py). Nothing here reaches the car's controls.
"""
import os
import pickle
import time
from pathlib import Path

import numpy as np

import cereal.messaging as messaging
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.worldmap.detector import lights, yolo
from openpilot.sunnypilot.worldmap.detector.ranging import CameraGeometry
from openpilot.sunnypilot.worldmap.detector.tracker import Tracker

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PKL = MODEL_DIR / "yolo11n_448x256_tinygrad.pkl"

TARGET_HZ = 2.0              # runs per second over both cameras
MAX_MODEL_EXEC = 0.035       # seconds: slower driving-model frames mean the GPU is busy enough
MAX_DM_EXEC = 0.03           # same for the driver monitoring model
MAX_FRAME_DROP = 0.5         # percent
MAX_BACKOFF = 2.0            # seconds between runs at most when backing off
HEARTBEAT = 0.5              # publish at least this often, so the viewer knows we're alive


class Network:
  def __init__(self, path: Path):
    from tinygrad.tensor import Tensor
    self._tensor = Tensor
    with open(path, "rb") as f:
      self.run = pickle.load(f)

  def __call__(self, x: np.ndarray) -> np.ndarray:
    return self.run(images=self._tensor(x, device="NPY")).numpy()


def camera_geometry(sm, wide: bool) -> CameraGeometry | None:
  from openpilot.common.transformations.camera import DEVICE_CAMERAS
  from openpilot.system.hardware import HARDWARE
  state = sm['wideRoadCameraState' if wide else 'roadCameraState']
  calib = sm['liveCalibration']
  if not sm.seen['liveCalibration'] or len(calib.rpyCalib) != 3:
    return None
  try:
    cams = DEVICE_CAMERAS[(str(HARDWARE.get_device_type()), str(state.sensor))]
  except KeyError:
    return None
  cam = cams.ecam if wide else cams.fcam
  height = calib.height[0] if len(calib.height) else 1.3
  wide_euler = list(calib.wideFromDeviceEuler) if wide and len(calib.wideFromDeviceEuler) == 3 else None
  return CameraGeometry(cam.width, cam.height, cam.focal_length, list(calib.rpyCalib), height, wide_euler)


def publish(pm, tracker: Tracker, ready: bool, night: bool, infer_s: float, rate: float, skipped: int) -> None:
  msg = messaging.new_message('worldObjectsSP', valid=ready)
  w = msg.worldObjectsSP
  w.valid = ready
  w.detectorReady = ready
  w.night = night
  w.inferenceTime = infer_s
  w.rateHz = rate
  w.skipped = skipped
  confirmed = tracker.confirmed()
  objs = w.init('objects', len(confirmed))
  now = tracker.t or 0.0
  for o, tr in zip(objs, confirmed, strict=True):
    o.id = tr.id
    o.kind = tr.kind
    o.confidence = tr.score
    o.x, o.y = tr.x, tr.y
    o.vx, o.vy = tr.vx, tr.vy
    o.width = tr.width
    o.camera = tr.camera
    o.lightColor = tr.light if tr.kind == "trafficLight" else "none"
    o.age = now - tr.born
  pm.send('worldObjectsSP', msg)


def main():
  try:
    os.nice(10)
  except (AttributeError, OSError):
    pass

  from msgq.visionipc import VisionIpcClient, VisionStreamType

  pm = messaging.PubMaster(['worldObjectsSP'])
  sm = messaging.SubMaster(['modelV2', 'carState', 'livePose', 'liveCalibration', 'roadCameraState', 'wideRoadCameraState',
                            'driverStateV2'], poll='modelV2')
  tracker = Tracker()

  net = None
  if MODEL_PKL.is_file():
    try:
      net = Network(MODEL_PKL)
    except Exception:
      cloudlog.exception("worldmap detector: model failed to load")
  else:
    cloudlog.warning("worldmap detector: no compiled model, detection off")

  clients = {
    "road": VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True),
    "wide": VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, True),
  }
  order = ["road", "wide"]
  turn = 0
  interval = 1.0 / TARGET_HZ
  next_run = 0.0
  last_pub = 0.0
  skipped = 0
  infer_s = 0.0
  rate = 0.0
  last_run: float | None = None
  night = False
  v_ego = 0.0

  while True:
    sm.update(200)
    now = time.monotonic()
    if sm.updated['carState']:
      v_ego = sm['carState'].vEgo
    yaw_rate = sm['livePose'].angularVelocityDevice.z if sm.seen['livePose'] else 0.0
    tracker.predict(now, v_ego, yaw_rate)

    ran = False
    if net is not None and sm.updated['modelV2'] and now >= next_run:
      model = sm['modelV2']
      dm_slow = sm.seen['driverStateV2'] and sm['driverStateV2'].modelExecutionTime > MAX_DM_EXEC
      if model.modelExecutionTime > MAX_MODEL_EXEC or model.frameDropPerc > MAX_FRAME_DROP or dm_slow:
        skipped += 1
        interval = min(interval * 1.5, MAX_BACKOFF)
        next_run = now + interval
      else:
        cam = order[turn % len(order)]
        turn += 1
        client = clients[cam]
        if not client.is_connected():
          client.connect(False)
        buf = client.recv(5) if client.is_connected() else None
        geom = camera_geometry(sm, cam == "wide") if buf is not None else None
        if buf is not None and geom is not None:
          t0 = time.monotonic()
          prep = yolo.from_nv12(np.asarray(buf.data), buf.width, buf.height, buf.stride, buf.uv_offset)
          out = net(prep.tensor)
          dets = []
          for d in yolo.decode(out, prep.scale, prep.night):
            loc = geom.locate(d.kind, d.box)
            if loc is None:
              continue
            light = lights.light_color(prep.rgb, d.box, prep.scale) if d.kind == "trafficLight" else "none"
            dets.append({"kind": d.kind, "x": loc[0], "y": loc[1], "width": loc[2], "score": d.score, "camera": cam, "light": light})
          tracker.update(now, dets)
          infer_s = 0.9 * infer_s + 0.1 * (time.monotonic() - t0) if infer_s else time.monotonic() - t0
          night = prep.night
          if last_run is not None:
            rate = 0.8 * rate + 0.2 / max(now - last_run, 1e-3)
          last_run = now
          interval = max(interval / 1.2, 1.0 / TARGET_HZ)
          ran = True
        next_run = now + interval

    if ran or now - last_pub > HEARTBEAT:
      publish(pm, tracker, net is not None, night, infer_s, rate, skipped)
      last_pub = now


if __name__ == "__main__":
  main()
