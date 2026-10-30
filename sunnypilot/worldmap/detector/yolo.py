"""
Getting camera frames in and detections out of the small YOLO11n detector.

The network takes a 448x256 RGB image (the whole 16:9 camera frame, scaled, with a thin gray band at the
bottom) and returns, for 2352 candidate boxes, the box (center x, center y, width, height in input pixels) and
a score for each of the 80 COCO classes. Only the classes that matter on the road are kept.

At night the frame is brightened before detection: local contrast on the luma channel (CLAHE) brings people
and signs out of the dark without blowing out headlights the way a plain gain would.
"""
from dataclasses import dataclass

import cv2
import numpy as np

INPUT_W, INPUT_H = 448, 256
PAD_VALUE = 114
NIGHT_LUMA = 60.0              # mean luma (0-255) below which the night preprocessing kicks in

# COCO class index -> our kind name (cereal WorldObjectsSP.Kind)
CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 9: "trafficLight", 11: "stopSign"}
MIN_SCORE = {"person": 0.35, "bicycle": 0.35, "car": 0.4, "motorcycle": 0.35, "bus": 0.4, "truck": 0.4,
             "trafficLight": 0.35, "stopSign": 0.45}
NIGHT_SCORE_FACTOR = 0.85      # be a little more willing at night, when scores run lower
NMS_IOU = 0.5

_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


@dataclass
class Prepared:
  rgb: np.ndarray        # INPUT_H x INPUT_W x 3, uint8, what the network saw
  tensor: np.ndarray     # 1 x 3 x INPUT_H x INPUT_W, float32 0..1
  scale: float           # camera pixels per input pixel
  night: bool
  luma: float


def from_nv12(buf: np.ndarray, width: int, height: int, stride: int, uv_offset: int) -> Prepared:
  """Scale a camera NV12 buffer down to the network input. Luma and chroma are scaled separately and only
  then converted, which is several times cheaper than converting the full frame first."""
  scale = width / INPUT_W
  out_h = int(round(height / scale)) & ~1
  y = buf[:stride * height].reshape(height, stride)[:, :width]
  uv = buf[uv_offset:uv_offset + stride * (height // 2)].reshape(height // 2, stride)[:, :width].reshape(height // 2, width // 2, 2)
  y_small = cv2.resize(y, (INPUT_W, out_h), interpolation=cv2.INTER_AREA)
  uv_small = cv2.resize(uv, (INPUT_W // 2, out_h // 2), interpolation=cv2.INTER_AREA)
  return _finish(y_small, uv_small, scale)


def from_rgb(rgb: np.ndarray) -> Prepared:
  """Same as from_nv12 for an RGB frame (tests and recorded video)."""
  h, w = rgb.shape[:2]
  scale = w / INPUT_W
  out_h = int(round(h / scale)) & ~1
  small = cv2.resize(rgb, (INPUT_W, out_h), interpolation=cv2.INTER_AREA)
  yuv = cv2.cvtColor(small, cv2.COLOR_RGB2YUV)
  return _finish(yuv[:, :, 0].copy(), None, scale, small_rgb=small)


def _finish(y_small: np.ndarray, uv_small: np.ndarray | None, scale: float, small_rgb: np.ndarray | None = None) -> Prepared:
  luma = float(y_small.mean())
  night = luma < NIGHT_LUMA
  if night:
    y_small = _clahe.apply(y_small)
  if small_rgb is not None:
    if night:  # put the brightened luma back under the original color
      yuv = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2YUV)
      yuv[:, :, 0] = y_small
      small_rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    rgb = small_rgb
  else:
    assert uv_small is not None
    nv12 = np.concatenate([y_small, uv_small.reshape(uv_small.shape[0], -1)], axis=0)
    rgb = cv2.cvtColor(nv12, cv2.COLOR_YUV2RGB_NV12)
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
    xywh = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 2], b[:, 3]], axis=1)
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), s[keep].tolist(), MIN_SCORE[kind] * factor, NMS_IOU)
    for i in np.asarray(idx).reshape(-1):
      x0, y0, w, h = xywh[i]
      dets.append(Detection(kind, float(s[keep][i]), np.array([x0, y0, x0 + w, y0 + h]) * scale))
  return dets
