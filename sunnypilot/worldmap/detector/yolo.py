"""
Getting camera frames in and detections out of the small YOLO11n detector.

The network takes a 448x256 RGB image (the whole 16:9 camera frame, scaled, with a thin gray band at the
bottom) and returns, for 2352 candidate boxes, the box (center x, center y, width, height in input pixels) and
a score for each of the 80 COCO classes. Only the classes that matter on the road are kept.

At night the frame is brightened before detection: each pixel is lifted by how dark its surroundings are
(a local gain from a heavily blurred copy), which brings people and signs out of the shadows without blowing
out headlights the way a plain gain would.

Only numpy and Pillow are used: the device has no OpenCV.
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image

from openpilot.sunnypilot.worldmap.imgops import resize_area

INPUT_W, INPUT_H = 448, 256
PAD_VALUE = 114
NIGHT_LUMA = 60.0              # mean luma (0-255) below which the night preprocessing kicks in
NIGHT_TARGET = 110.0           # local brightness the night gain aims for
NIGHT_MAX_GAIN = 5.0

# COCO class index -> our kind name (cereal WorldObjectsSP.Kind)
CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 9: "trafficLight", 11: "stopSign"}
MIN_SCORE = {"person": 0.35, "bicycle": 0.35, "car": 0.4, "motorcycle": 0.35, "bus": 0.4, "truck": 0.4,
             "trafficLight": 0.35, "stopSign": 0.45}
NIGHT_SCORE_FACTOR = 0.85      # be a little more willing at night, when scores run lower
NMS_IOU = 0.5


@dataclass
class Prepared:
  rgb: np.ndarray        # INPUT_H x INPUT_W x 3, uint8, what the network saw
  tensor: np.ndarray     # 1 x 3 x INPUT_H x INPUT_W, float32 0..1
  scale: float           # camera pixels per input pixel
  night: bool
  luma: float


def night_gain(luma: np.ndarray) -> np.ndarray:
  """Per-pixel gain lifting dark surroundings toward NIGHT_TARGET (a stand-in for CLAHE)."""
  h, w = luma.shape
  small = Image.fromarray(luma).resize((max(w // 16, 1), max(h // 16, 1)), Image.BOX)
  surround = np.asarray(small.resize((w, h), Image.BILINEAR), np.float32)
  return np.clip(NIGHT_TARGET / (surround + 12.0), 1.0, NIGHT_MAX_GAIN)


def nv12_to_rgb(y: np.ndarray, uv: np.ndarray) -> np.ndarray:
  """BT.601 limited range, as the camera encodes it. y: (H, W) uint8; uv: (H/2, W/2, 2) uint8."""
  yf = (y.astype(np.float32) - 16.0) * 1.164
  u = np.repeat(np.repeat(uv[:, :, 0].astype(np.float32) - 128.0, 2, axis=0), 2, axis=1)[:y.shape[0], :y.shape[1]]
  v = np.repeat(np.repeat(uv[:, :, 1].astype(np.float32) - 128.0, 2, axis=0), 2, axis=1)[:y.shape[0], :y.shape[1]]
  rgb = np.stack([yf + 1.596 * v, yf - 0.813 * v - 0.391 * u, yf + 2.018 * u], axis=-1)
  return np.clip(rgb, 0, 255).astype(np.uint8)


def from_nv12(buf: np.ndarray, width: int, height: int, stride: int, uv_offset: int) -> Prepared:
  """Scale a camera NV12 buffer down to the network input. Luma and chroma are scaled separately and only
  then converted, which is far cheaper than converting the full frame first."""
  scale = width / INPUT_W
  out_h = int(round(height / scale)) & ~1
  y = np.ascontiguousarray(buf[:stride * height].reshape(height, stride)[:, :width])
  uv = buf[uv_offset:uv_offset + stride * (height // 2)].reshape(height // 2, stride)[:, :width].reshape(height // 2, width // 2, 2)
  y_small = resize_area(y, INPUT_W, out_h)
  u_small = resize_area(np.ascontiguousarray(uv[:, :, 0]), INPUT_W // 2, out_h // 2)
  v_small = resize_area(np.ascontiguousarray(uv[:, :, 1]), INPUT_W // 2, out_h // 2)
  luma = float(y_small.mean())
  night = luma < NIGHT_LUMA
  rgb = nv12_to_rgb(y_small, np.stack([u_small, v_small], axis=-1))
  if night:
    rgb = np.clip(rgb.astype(np.float32) * night_gain(y_small)[:, :, None], 0, 255).astype(np.uint8)
  return _finish(rgb, scale, night, luma)


def from_rgb(rgb: np.ndarray) -> Prepared:
  """Same as from_nv12 for an RGB frame (tests and recorded video)."""
  h, w = rgb.shape[:2]
  scale = w / INPUT_W
  out_h = int(round(h / scale)) & ~1
  small = resize_area(np.ascontiguousarray(rgb), INPUT_W, out_h)
  luma_img = (small[:, :, 0] * 0.299 + small[:, :, 1] * 0.587 + small[:, :, 2] * 0.114).astype(np.uint8)
  luma = float(luma_img.mean())
  night = luma < NIGHT_LUMA
  if night:
    small = np.clip(small.astype(np.float32) * night_gain(luma_img)[:, :, None], 0, 255).astype(np.uint8)
  return _finish(small, scale, night, luma)


def _finish(rgb: np.ndarray, scale: float, night: bool, luma: float) -> Prepared:
  canvas = np.full((INPUT_H, INPUT_W, 3), PAD_VALUE, np.uint8)
  h = min(rgb.shape[0], INPUT_H)
  canvas[:h] = rgb[:h]
  tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
  return Prepared(canvas, tensor, scale, night, luma)


@dataclass
class Detection:
  kind: str
  score: float
  box: np.ndarray        # x0, y0, x1, y1 in camera pixels


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
  """Greedy non-max suppression on x0, y0, x1, y1 boxes. Returns kept indices, best first."""
  order = np.argsort(-scores)
  area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
  keep: list[int] = []
  while len(order):
    i = int(order[0])
    keep.append(i)
    rest = order[1:]
    w = np.clip(np.minimum(boxes[i, 2], boxes[rest, 2]) - np.maximum(boxes[i, 0], boxes[rest, 0]), 0, None)
    h = np.clip(np.minimum(boxes[i, 3], boxes[rest, 3]) - np.maximum(boxes[i, 1], boxes[rest, 1]), 0, None)
    inter = w * h
    order = rest[inter / np.maximum(area[i] + area[rest] - inter, 1e-6) <= iou]
  return keep


def decode(output: np.ndarray, scale: float, night: bool) -> list[Detection]:
  """Raw network output (1 x 84 x N) to scored boxes in camera pixels, after per-class non-max suppression."""
  out = output.reshape(84, -1)
  boxes_c = out[:4].T
  scores = out[4:]
  dets: list[Detection] = []
  factor = NIGHT_SCORE_FACTOR if night else 1.0
  for cls, kind in CLASSES.items():
    s = scores[cls]
    keep = np.nonzero(s >= MIN_SCORE[kind] * factor)[0]
    if not len(keep):
      continue
    b = boxes_c[keep]
    xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], axis=1)
    for i in nms(xyxy, s[keep], NMS_IOU):
      dets.append(Detection(kind, float(s[keep][i]), xyxy[i] * scale))
  return dets
