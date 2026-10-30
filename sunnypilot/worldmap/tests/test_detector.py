import math

import numpy as np

from openpilot.sunnypilot.worldmap.detector import lights, yolo
from openpilot.sunnypilot.worldmap.detector.ranging import CameraGeometry
from openpilot.sunnypilot.worldmap.detector.tracker import Tracker


def fake_output(boxes):
  """boxes: (cx, cy, w, h, class, score) in network input pixels."""
  out = np.zeros((1, 84, 2352), np.float32)
  for i, (cx, cy, w, h, cls, score) in enumerate(boxes):
    out[0, :4, i] = (cx, cy, w, h)
    out[0, 4 + cls, i] = score
  return out


def test_decode_keeps_road_classes_and_suppresses_duplicates():
  out = fake_output([(100, 100, 20, 40, 0, 0.9), (101, 100, 20, 40, 0, 0.8),   # same person twice
                     (300, 120, 60, 40, 2, 0.7), (50, 50, 10, 10, 14, 0.99)])  # a car; a bird (ignored)
  dets = yolo.decode(out, 3.0, night=False)
  assert sorted(d.kind for d in dets) == ["car", "person"]
  person = next(d for d in dets if d.kind == "person")
  assert np.allclose(person.box, np.array([90, 80, 110, 120]) * 3.0)


def test_night_frames_are_brightened():
  dark = np.full((760, 1344, 3), 20, np.uint8)
  dark[300:400, 600:700] = 60
  prep = yolo.from_rgb(dark)
  assert prep.night and prep.tensor.shape == (1, 3, yolo.INPUT_H, yolo.INPUT_W)
  assert not yolo.from_rgb(np.full((760, 1344, 3), 140, np.uint8)).night


def test_ground_ranging():
  geom = CameraGeometry(1344, 760, 1141.5, [0.0, 0.0, 0.0], 1.3)
  # a point 20 m ahead on the road sits f*h/d pixels below the image center
  v = 380 + 1141.5 * 1.3 / 20.0
  x, y = geom.ground_point(672, v)
  assert abs(x - 20.0) < 0.05 and abs(y) < 0.05
  x, y = geom.ground_point(672 + 1141.5 * 2.0 / 20.0, v)
  assert abs(y - 2.0) < 0.05  # to the right of the image is to the right of the car
  assert geom.ground_point(672, 300) is None  # above the horizon


def test_signs_are_ranged_by_size():
  geom = CameraGeometry(1344, 760, 1141.5, [0.0, 0.0, 0.0], 1.3)
  pw = 1141.5 * 0.75 / 30.0
  x, y, _ = geom.locate("stopSign", np.array([700, 300, 700 + pw, 300 + pw]))
  assert abs(math.hypot(x, y) - 30.0) < 0.5


def test_light_color():
  img = np.zeros((256, 448, 3), np.uint8)
  img[50:60, 100:108] = (255, 30, 30)
  assert lights.light_color(img, np.array([96, 40, 112, 90]) * 2.0, 2.0) == "red"
  img[50:60, 100:108] = (40, 255, 120)
  assert lights.light_color(img, np.array([96, 40, 112, 90]) * 2.0, 2.0) == "green"
  img[50:60, 100:108] = (20, 20, 20)
  assert lights.light_color(img, np.array([96, 40, 112, 90]) * 2.0, 2.0) == "none"


def test_tracker_carries_parked_things_through_our_motion():
  tr = Tracker()
  tr.predict(0.0, 10.0, 0.0)
  tr.update(0.0, [{"kind": "person", "x": 30.0, "y": 2.0, "width": 0.5, "score": 0.8, "camera": "road"}])
  for k in range(1, 11):
    tr.predict(k * 0.1, 10.0, 0.0)   # we drive 10 m toward a standing person
  t = tr.tracks[0]
  assert abs(t.x - 20.0) < 0.01 and abs(t.y - 2.0) < 0.01
  tr.update(1.0, [{"kind": "person", "x": 20.2, "y": 2.0, "width": 0.5, "score": 0.8, "camera": "road"}])
  assert len(tr.tracks) == 1 and tr.tracks[0].hits == 2
  tr.predict(1.0 + 0.5, 0.0, math.radians(20))  # turning right in place: things ahead swing left
  assert tr.tracks[0].y < 2.0
  tr.predict(5.0, 0.0, 0.0)
  assert tr.tracks == []  # unseen too long
