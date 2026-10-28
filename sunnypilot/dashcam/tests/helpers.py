import os
from fractions import Fraction

import av
import numpy as np
import pytest

requires_hevc_encoder = pytest.mark.skipif("libx265" not in av.codecs_available, reason="needs an HEVC encoder")


def make_hevc(path: str, frames: int, shade: int = 0, fps: int = 20, gop: int = 30) -> None:
  """Raw annex-B HEVC like fcamera.hevc: no B-frames, a keyframe every `gop` frames."""
  ctx = av.CodecContext.create("libx265", "w")
  ctx.width, ctx.height, ctx.pix_fmt = 160, 96, "yuv420p"
  ctx.time_base = Fraction(1, fps)
  ctx.framerate = Fraction(fps, 1)
  ctx.gop_size = gop
  ctx.max_b_frames = 0
  ctx.options = {"x265-params": f"keyint={gop}:min-keyint={gop}:bframes=0:scenecut=0:log-level=error"}
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "wb") as f:
    for i in range(frames):
      img = np.full((96, 160, 3), (shade + i) % 256, dtype=np.uint8)
      frame = av.VideoFrame.from_ndarray(img, format="rgb24").reformat(format="yuv420p")
      frame.pts = i
      for pkt in ctx.encode(frame):
        f.write(bytes(pkt))
    for pkt in ctx.encode(None):
      f.write(bytes(pkt))
