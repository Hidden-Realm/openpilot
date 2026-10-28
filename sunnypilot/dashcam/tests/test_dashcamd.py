import pytest

from openpilot.sunnypilot.dashcam.dashcamd import SegmentClock


class TestSegmentClock:
  def test_uses_encoder_frame_for_segment_open_at_start(self):
    clock = SegmentClock()
    clock.update("route", 4, now=100.0)
    assert clock.offset(now=100.0, encoder_frame=600) == pytest.approx(30.0)

  def test_times_segments_it_saw_start(self):
    clock = SegmentClock()
    clock.update("route", 4, now=100.0)
    clock.update("route", 5, now=130.0)
    # encoder frame is ignored once we know when the segment began
    assert clock.offset(now=142.5, encoder_frame=1190) == pytest.approx(12.5)

  def test_new_route_resets(self):
    clock = SegmentClock()
    clock.update("a", 4, now=100.0)
    clock.update("a", 5, now=130.0)
    clock.update("b", 0, now=200.0)
    assert clock.offset(now=205.0, encoder_frame=None) == 0.0
    assert clock.offset(now=205.0, encoder_frame=100) == pytest.approx(5.0)
