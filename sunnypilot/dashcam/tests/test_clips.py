import os

import pytest

from openpilot.sunnypilot.dashcam.clips import ClipStore, clip_cameras, clip_window, current_segment, make_clip
from openpilot.sunnypilot.dashcam.hold import DEFAULT_WHEEL_BUTTON, ButtonHold, ButtonTap, set_wheel_button, wheel_button

ROUTE = "0000002a--6c3b9f3e1d"


class TestClipWindow:
  def test_three_minutes_before_one_after(self):
    # pressed 30 s into segment 5: from 2:30 into the drive until 6:30
    first, last, start, duration = clip_window(5, 30.0)
    assert (first, last) == (2, 6)
    assert start == pytest.approx(30.0)
    assert duration == pytest.approx(240.0)

  def test_press_on_segment_boundary(self):
    # pressed at exactly 5:00: from 2:00 to 6:00, segment 6 isn't needed
    assert clip_window(5, 0.0) == (2, 5, 0.0, 240.0)

  def test_press_early_in_drive(self):
    # only 70 s of history exists
    first, last, start, duration = clip_window(1, 10.0)
    assert (first, last, start) == (0, 2, 0.0)
    assert duration == pytest.approx(130.0)

  def test_segments(self):
    clip = make_clip(ROUTE, 5, 30.0, now=1000.0)
    assert clip.segments == [f"{ROUTE}--{n}" for n in range(2, 7)]


class TestClipStore:
  def test_save_list_delete(self, tmp_path):
    store = ClipStore(str(tmp_path))
    older = make_clip(ROUTE, 3, 10.0, now=100.0)
    newer = make_clip(ROUTE, 9, 10.0, now=200.0)
    store.save(older)
    store.save(newer)

    assert [c.id for c in store.list()] == [newer.id, older.id]
    assert store.get(older.id) == older
    assert f"{ROUTE}--0" in store.saved_segments()
    assert f"{ROUTE}--10" in store.saved_segments()

    store.delete(newer.id)
    assert [c.id for c in store.list()] == [older.id]
    assert f"{ROUTE}--10" not in store.saved_segments()

  def test_oldest_clip_using_shared_segment(self, tmp_path):
    store = ClipStore(str(tmp_path))
    a = make_clip(ROUTE, 5, 0.0, now=100.0)   # segments 2-5
    b = make_clip(ROUTE, 6, 0.0, now=160.0)   # segments 3-6
    store.save(b)
    store.save(a)
    assert store.oldest_clip_using(f"{ROUTE}--4").id == a.id
    assert store.oldest_clip_using(f"{ROUTE}--6").id == b.id
    assert store.oldest_clip_using(f"{ROUTE}--9") is None

  def test_rejects_path_ids(self, tmp_path):
    assert ClipStore(str(tmp_path)).get("../../etc/passwd") is None

  def test_ignores_corrupt_records(self, tmp_path):
    store = ClipStore(str(tmp_path))
    os.makedirs(store.clips_dir)
    with open(os.path.join(store.clips_dir, "broken.json"), "w") as f:
      f.write("{not json")
    assert store.list() == []


def test_current_segment(tmp_path):
  for name in (f"{ROUTE}--0", f"{ROUTE}--1", f"{ROUTE}--10", "00000001--aaaaaaaaaa--99", "boot", "crash"):
    os.makedirs(tmp_path / name)
  assert current_segment(str(tmp_path), ROUTE) == 10
  assert current_segment(str(tmp_path), "nope") is None


def test_clip_cameras(tmp_path):
  clip = make_clip(ROUTE, 3, 0.0, now=0.0)
  (tmp_path / f"{ROUTE}--1").mkdir()
  (tmp_path / f"{ROUTE}--1" / "ecamera.hevc").write_bytes(b"x")
  (tmp_path / f"{ROUTE}--2").mkdir()
  (tmp_path / f"{ROUTE}--2" / "fcamera.hevc").write_bytes(b"x")
  assert clip_cameras(clip, [str(tmp_path / "missing"), str(tmp_path)]) == ["fcamera.hevc", "ecamera.hevc"]


class TestButtonHold:
  def test_fires_once_after_one_second(self):
    hold = ButtonHold()
    assert not hold.update(True, 10.0)
    assert not hold.update(None, 10.5)
    assert hold.update(None, 11.0)
    assert not hold.update(None, 12.0)   # still held: no repeat
    assert not hold.update(False, 12.5)

  def test_short_press_does_nothing(self):
    hold = ButtonHold()
    assert not hold.update(True, 0.0)
    assert not hold.update(False, 0.4)
    assert not hold.update(None, 5.0)

  def test_rearms_after_release(self):
    hold = ButtonHold()
    hold.update(True, 0.0)
    assert hold.update(None, 1.0)
    hold.update(False, 1.2)
    hold.update(True, 2.0)
    assert hold.update(None, 3.0)

  def test_reset_cancels_hold(self):
    hold = ButtonHold()
    hold.update(True, 0.0)
    hold.reset()
    assert not hold.update(None, 2.0)


class TestButtonTap:
  def test_short_press_fires_on_release(self):
    tap = ButtonTap()
    assert not tap.update(True, 10.0)
    assert not tap.update(None, 10.3)
    assert tap.update(False, 10.4)

  def test_hold_does_not_fire(self):
    tap = ButtonTap()
    tap.update(True, 0.0)
    assert not tap.update(None, 1.5)
    assert not tap.update(False, 1.6)

  def test_press_and_release_in_one_frame(self):
    tap = ButtonTap()
    assert not tap.update(True, 5.0)
    assert tap.update(False, 5.0)

  def test_release_without_press(self):
    assert not ButtonTap().update(False, 1.0)

  def test_tap_and_hold_agree_at_the_threshold(self):
    # the same press is either a tap or a hold, never both and never neither
    for held in (0.2, 0.99, 1.0, 1.5):
      tap, hold = ButtonTap(), ButtonHold()
      fired = [hold.update(True, 0.0), tap.update(True, 0.0)]
      fired += [hold.update(None, held), hold.update(False, held), tap.update(False, held)]
      assert sum(fired) == 1, held


class TestWheelButton:
  def test_default_and_saved(self, tmp_path):
    root = str(tmp_path)
    assert wheel_button(root) == DEFAULT_WHEEL_BUTTON == "lkas"
    set_wheel_button(root, "distance")
    assert wheel_button(root) == "distance"

  def test_rejects_unknown(self, tmp_path):
    with pytest.raises(ValueError):
      set_wheel_button(str(tmp_path), "horn")

  def test_bad_file_falls_back(self, tmp_path):
    (tmp_path / "settings.json").write_text("[1, 2]")
    assert wheel_button(str(tmp_path)) == "lkas"
    (tmp_path / "settings.json").write_text('{"wheel_button": "horn"}')
    assert wheel_button(str(tmp_path)) == "lkas"
