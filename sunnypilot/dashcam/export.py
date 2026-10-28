"""
Turns a saved clip into a single MP4 that plays on an iPhone.

The camera files are raw HEVC, one per minute. Exporting copies the compressed
frames into an MP4 without re-encoding, so it is fast and costs almost no CPU.
The cut starts at the keyframe at or before the clip start (keyframes are at
most 1.5 s apart). The stream is tagged hvc1, which Safari and Photos need.

The encoder never uses B-frames, so presentation order is file order and each
frame gets the next timestamp.
"""
import os
from fractions import Fraction

import av

from openpilot.sunnypilot.dashcam.clips import CAMERA_FPS, SEGMENT_SECONDS, Clip, find_segment_dir

THUMB_WIDTH = 480


def _segment_files(clip: Clip, cam_file: str, roots: list[str]):
  for i, seg in enumerate(range(clip.first_segment, clip.last_segment + 1)):
    d = find_segment_dir(f"{clip.route}--{seg}", roots)
    path = os.path.join(d, cam_file) if d else None
    if path and os.path.isfile(path) and os.path.getsize(path) > 0:
      yield i, path


def export_clip(clip: Clip, cam_file: str, roots: list[str], dst: str,
                segment_seconds: float = SEGMENT_SECONDS, fps: int = CAMERA_FPS) -> int:
  """Write the clip to dst as MP4. Returns the number of frames written (0 means nothing to export)."""
  start = clip.start_offset
  end = clip.start_offset + clip.duration
  time_base = Fraction(1, fps)
  tmp = dst + ".part"

  out = None
  out_stream = None
  frames = 0
  pending: list[av.Packet] = []  # packets from the last keyframe up to the clip start
  started = False

  def write(pkt: av.Packet) -> None:
    nonlocal frames
    pkt.stream = out_stream
    pkt.pts = frames
    pkt.dts = frames
    pkt.time_base = time_base
    pkt.duration = 1
    out.mux(pkt)
    frames += 1

  try:
    for index, path in _segment_files(clip, cam_file, roots):
      base = index * segment_seconds
      if base >= end:
        break
      try:
        inp = av.open(path, format="hevc")
      except av.FFmpegError:
        continue  # unreadable segment, skip it
      with inp:
        in_stream = inp.streams.video[0]
        if out is None:
          out = av.open(tmp, "w", format="mp4", options={"movflags": "+faststart"})
          out_stream = out.add_stream_from_template(in_stream)
          out_stream.codec_tag = "hvc1"
          out_stream.time_base = time_base

        n = 0

        def handle(pkt: av.Packet, base: float = base) -> bool:
          """Add one frame to the clip. Returns False once past the clip end."""
          nonlocal n, started, pending
          t = base + n / fps
          n += 1
          if t >= end:
            return False
          if started:
            write(pkt)
          elif pkt.is_keyframe or pending:
            pending = [pkt] if pkt.is_keyframe else pending + [pkt]
            if t >= start:
              started = True
              for p in pending:
                write(p)
              pending = []
          return True

        # Hold back each frame until the next arrives: if power was cut mid-write, the last
        # frame of a short segment is incomplete and would show as corruption, so drop it.
        last = None
        complete = False
        try:
          for pkt in inp.demux(in_stream):
            if pkt.size == 0:
              continue
            if last is not None and not handle(last):
              last = None
              break
            last = pkt
          complete = n + 1 >= segment_seconds * fps
        except av.FFmpegError:
          pass  # unreadable tail; keep what we have
        if last is not None and complete:
          handle(last)
  finally:
    if out is not None:
      out.close()

  if frames == 0:
    if os.path.exists(tmp):
      os.remove(tmp)
    return 0
  os.replace(tmp, dst)
  return frames


def make_thumbnail(clip: Clip, roots: list[str], dst: str) -> bool:
  """Save a JPEG of the first frame of the minute the button was pressed in."""
  seg_name = f"{clip.route}--{clip.trigger_segment}"
  d = find_segment_dir(seg_name, roots)
  if d is None:
    return False
  for name, fmt in (("qcamera.ts", None), ("fcamera.hevc", "hevc"), ("ecamera.hevc", "hevc")):
    path = os.path.join(d, name)
    if not os.path.isfile(path):
      continue
    try:
      with av.open(path, format=fmt) as inp:
        for frame in inp.decode(video=0):
          img = frame.to_image()
          img.thumbnail((THUMB_WIDTH, THUMB_WIDTH))
          os.makedirs(os.path.dirname(dst), exist_ok=True)
          img.save(dst, "JPEG", quality=80)
          return True
    except (av.FFmpegError, OSError):
      continue
  return False
