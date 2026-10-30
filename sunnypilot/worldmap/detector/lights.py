"""
Traffic light color from the pixels inside its box.

A lit lamp is a bright, saturated blob. Bright saturated pixels are counted by hue (red wraps around 0, then
amber, then green/cyan for LED greens), and the color with the clear majority wins. Few lit pixels, or no
clear winner, means unknown: better to show nothing than the wrong color.
"""
import numpy as np

from openpilot.sunnypilot.worldmap.imgops import rgb_to_hsv

MIN_LIT_FRACTION = 0.03
MIN_MARGIN = 1.5


def light_color(rgb: np.ndarray, box: np.ndarray, scale: float) -> str:
  """rgb: the network input image; box: camera pixels; scale: camera pixels per input pixel."""
  x0, y0, x1, y1 = (int(round(v / scale)) for v in box)
  h, w = rgb.shape[:2]
  x0, x1 = max(x0, 0), min(x1, w)
  y0, y1 = max(y0, 0), min(y1, h)
  if x1 - x0 < 2 or y1 - y0 < 3:
    return "none"
  hsv = rgb_to_hsv(rgb[y0:y1, x0:x1])
  hue, sat, val = hsv[:, :, 0].astype(np.int32), hsv[:, :, 1], hsv[:, :, 2]
  lit = (val > 150) & (sat > 90)
  if lit.mean() < MIN_LIT_FRACTION:
    return "none"
  hl = hue[lit]  # Pillow hue is 0..255 for the full circle
  counts = {
    "red": int(np.sum((hl <= 11) | (hl >= 234))),
    "yellow": int(np.sum((hl > 17) & (hl <= 45))),
    "green": int(np.sum((hl > 64) & (hl <= 142))),
  }
  best = max(counts, key=lambda k: counts[k])
  rest = max(v for k, v in counts.items() if k != best)
  if counts[best] < 3 or counts[best] < MIN_MARGIN * max(rest, 1):
    return "none"
  return best
