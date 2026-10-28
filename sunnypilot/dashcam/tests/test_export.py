import os

import av

from openpilot.sunnypilot.dashcam.clips import Clip
from openpilot.sunnypilot.dashcam.export import export_clip, make_thumbnail
from openpilot.sunnypilot.dashcam.tests.helpers import make_hevc, requires_hevc_encoder

ROUTE = "0000002a--6c3b9f3e1d"
FPS = 20
SEG = 3  # short segments keep the test fast: 60 frames each


def make_route(root: str, segments: int) -> None:
  for n in range(segments):
    make_hevc(os.path.join(root, f"{ROUTE}--{n}", "fcamera.hevc"), SEG * FPS, shade=n * 60)


def clip(first: int, last: int, start: float, duration: float) -> Clip:
  return Clip(id="c", route=ROUTE, created=0.0, trigger_segment=last, trigger_offset=0.0,
              first_segment=first, last_segment=last, start_offset=start, duration=duration)


def probe(path: str) -> tuple[str, int, float]:
  with av.open(path) as f:
    v = f.streams.video[0]
    decoded = sum(1 for _ in f.decode(v))
    return v.codec_context.codec_tag, decoded, float(f.duration / av.time_base)


@requires_hevc_encoder
def test_exports_across_segments(tmp_path):
  make_route(str(tmp_path), 4)
  dst = str(tmp_path / "out.mp4")

  # start on a keyframe (frame 30 of segment 1) and run 6 s into segment 3
  frames = export_clip(clip(1, 3, 1.5, 6.0), "fcamera.hevc", [str(tmp_path)], dst, segment_seconds=SEG, fps=FPS)

  assert frames == 6 * FPS
  tag, decoded, duration = probe(dst)
  assert tag == "hvc1"
  assert decoded == frames
  assert abs(duration - 6.0) < 0.1


@requires_hevc_encoder
def test_start_backs_up_to_keyframe(tmp_path):
  make_route(str(tmp_path), 2)
  dst = str(tmp_path / "out.mp4")

  # 1.0 s in is frame 20; the keyframe before it is frame 0, so the clip gains 20 frames
  frames = export_clip(clip(0, 1, 1.0, 2.0), "fcamera.hevc", [str(tmp_path)], dst, segment_seconds=SEG, fps=FPS)

  assert frames == 20 + 2 * FPS
  assert probe(dst)[1] == frames


@requires_hevc_encoder
def test_missing_segment_is_skipped(tmp_path):
  make_route(str(tmp_path), 3)
  os.remove(tmp_path / f"{ROUTE}--1" / "fcamera.hevc")
  dst = str(tmp_path / "out.mp4")

  frames = export_clip(clip(0, 2, 0.0, 9.0), "fcamera.hevc", [str(tmp_path)], dst, segment_seconds=SEG, fps=FPS)

  assert frames == 2 * SEG * FPS
  assert probe(dst)[1] == frames


@requires_hevc_encoder
def test_truncated_segment_keeps_what_it_can(tmp_path):
  make_route(str(tmp_path), 2)
  path = tmp_path / f"{ROUTE}--1" / "fcamera.hevc"
  data = path.read_bytes()
  path.write_bytes(data[:len(data) // 2])  # power cut mid-write
  dst = str(tmp_path / "out.mp4")

  frames = export_clip(clip(0, 1, 0.0, 6.0), "fcamera.hevc", [str(tmp_path)], dst, segment_seconds=SEG, fps=FPS)

  assert SEG * FPS < frames < 2 * SEG * FPS
  assert probe(dst)[1] >= frames - 1


def test_nothing_to_export(tmp_path):
  dst = str(tmp_path / "out.mp4")
  assert export_clip(clip(0, 1, 0.0, 6.0), "fcamera.hevc", [str(tmp_path)], dst) == 0
  assert not os.path.exists(dst)
  assert not os.path.exists(dst + ".part")


@requires_hevc_encoder
def test_thumbnail(tmp_path):
  make_route(str(tmp_path), 2)
  dst = str(tmp_path / "thumbs" / "c.jpg")
  assert make_thumbnail(clip(0, 1, 0.0, 6.0), [str(tmp_path)], dst)
  with open(dst, "rb") as f:
    assert f.read(2) == b"\xff\xd8"
