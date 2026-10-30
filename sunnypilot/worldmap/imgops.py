"""
Small image operations for the world map, with numpy and Pillow only.

The device's Python has numpy and Pillow but not OpenCV (OpenCV is only a development dependency), so
everything here is written to need nothing else. Each function is a plain stand-in for the OpenCV call
named in its docstring.
"""
import io

import numpy as np
from PIL import Image, ImageDraw


def draw_mask(height: int, width: int, cols: np.ndarray, rows: np.ndarray, kind: str, line_width: int = 1) -> np.ndarray:
  """Rasterize one shape into a uint8 0/1 mask (cv2.fillPoly / cv2.polylines). kind is "fill" or "line"."""
  img = Image.new("L", (width, height), 0)
  draw = ImageDraw.Draw(img)
  pts = list(zip(cols.tolist(), rows.tolist(), strict=True))
  if kind == "fill":
    if len(pts) >= 3:
      draw.polygon(pts, fill=1, outline=1)
  elif len(pts) >= 2:
    draw.line(pts, fill=1, width=max(int(line_width), 1))
  return np.asarray(img, dtype=np.uint8)


def resample(arr: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
  """Bilinear samples of `arr` on the grid rows x cols (1-D fractional pixel positions, centers at integers),
  zero outside (cv2.remap with separable maps and a constant border)."""
  a = np.pad(arr.astype(np.float32), 1)
  y = np.asarray(rows, np.float64) + 1.0
  x = np.asarray(cols, np.float64) + 1.0
  y0 = np.clip(np.floor(y).astype(np.int64), 0, a.shape[0] - 2)
  x0 = np.clip(np.floor(x).astype(np.int64), 0, a.shape[1] - 2)
  fy = np.clip(y - y0, 0.0, 1.0).astype(np.float32)[:, None]
  fx = np.clip(x - x0, 0.0, 1.0).astype(np.float32)[None, :]
  top = a[y0][:, x0] * (1 - fx) + a[y0][:, x0 + 1] * fx
  bottom = a[y0 + 1][:, x0] * (1 - fx) + a[y0 + 1][:, x0 + 1] * fx
  return top * (1 - fy) + bottom * fy


def distance_to(mask: np.ndarray, cap: int) -> np.ndarray:
  """Distance in cells from each cell to the nearest True cell, capped at `cap` (cv2.distanceTransform).
  Exact Euclidean within the cap: squared distances along rows, then along columns."""
  inf = float((cap + 1) ** 2)
  g = np.where(mask, 0.0, inf).astype(np.float32)
  rows_pass = g.copy()
  for d in range(1, cap + 1):
    np.minimum(rows_pass[:, d:], g[:, :-d] + d * d, out=rows_pass[:, d:])
    np.minimum(rows_pass[:, :-d], g[:, d:] + d * d, out=rows_pass[:, :-d])
  out = rows_pass.copy()
  for d in range(1, cap + 1):
    np.minimum(out[d:], rows_pass[:-d] + d * d, out=out[d:])
    np.minimum(out[:-d], rows_pass[d:] + d * d, out=out[:-d])
  return np.minimum(np.sqrt(out), float(cap))


def gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
  """Separable Gaussian blur with mirrored edges (cv2.GaussianBlur)."""
  r = max(int(np.ceil(3 * sigma)), 1)
  k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2).astype(np.float32)
  k /= k.sum()
  a = arr.astype(np.float32)
  h, w = a.shape
  p = np.pad(a, ((r, r), (0, 0)), mode="reflect")
  tmp = sum(k[i] * p[i:i + h] for i in range(2 * r + 1))
  p = np.pad(tmp, ((0, 0), (r, r)), mode="reflect")
  return sum(k[i] * p[:, i:i + w] for i in range(2 * r + 1))


def resize_area(img: np.ndarray, width: int, height: int) -> np.ndarray:
  """Shrink by averaging (cv2.resize with INTER_AREA). Works on gray (H, W) and color (H, W, 3) uint8."""
  return np.asarray(Image.fromarray(img).resize((width, height), Image.BOX))


def png_rgba(rgba: np.ndarray) -> bytes:
  """PNG bytes of an (H, W, 4) uint8 RGBA image (cv2.imencode)."""
  buf = io.BytesIO()
  Image.fromarray(rgba).save(buf, format="PNG", compress_level=3)
  return buf.getvalue()


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
  """Pillow's HSV: all three channels 0..255 (hue too, unlike OpenCV's 0..179)."""
  return np.asarray(Image.fromarray(np.ascontiguousarray(rgb)).convert("HSV"))
