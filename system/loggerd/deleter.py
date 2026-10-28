#!/usr/bin/env python3
import os
import time
import shutil
import threading
from pathlib import Path
from openpilot.system.hardware.hw import Paths
from openpilot.common.swaglog import cloudlog
from openpilot.system.loggerd.config import get_available_bytes, get_available_percent
from openpilot.system.loggerd.uploader import listdir_by_creation
from openpilot.system.loggerd.xattr_cache import getxattr
from openpilot.sunnypilot.dashcam.clips import ClipStore, dashcam_root

MIN_BYTES = 5 * 1024 * 1024 * 1024
MIN_PERCENT = 10

DELETE_LAST = ['boot', 'crash']

PRESERVE_ATTR_NAME = 'user.preserve'
PRESERVE_ATTR_VALUE = b'1'
PRESERVE_COUNT = 5


def has_preserve_xattr(d: str) -> bool:
  return getxattr(os.path.join(Paths.log_root(), d), PRESERVE_ATTR_NAME) == PRESERVE_ATTR_VALUE


def get_preserved_segments(dirs_by_creation: list[str]) -> set[str]:
  # skip deleting most recent N preserved segments (and their prior segment)
  preserved = set()
  for n, d in enumerate(filter(has_preserve_xattr, reversed(dirs_by_creation))):
    if n == PRESERVE_COUNT:
      break
    date_str, _, seg_str = d.rpartition("--")

    # ignore non-segment directories
    if not date_str:
      continue
    try:
      seg_num = int(seg_str)
    except ValueError:
      continue

    # preserve segment and two prior
    for _seg_num in range(max(0, seg_num - 2), seg_num + 1):
      preserved.add(f"{date_str}--{_seg_num}")

  return preserved


def deleter_thread(exit_event: threading.Event):
  dashcam = ClipStore(dashcam_root())
  while not exit_event.is_set():
    out_of_bytes = get_available_bytes(default=MIN_BYTES + 1) < MIN_BYTES
    out_of_percent = get_available_percent(default=MIN_PERCENT + 1) < MIN_PERCENT

    if out_of_percent or out_of_bytes:
      dirs = listdir_by_creation(Paths.log_root())
      preserved_dirs = get_preserved_segments(dirs)
      saved_dirs = dashcam.saved_segments()

      # remove the earliest directory we can. saved dash cam clips go last of all
      for delete_dir in sorted(dirs, key=lambda d: (d in saved_dirs, d in DELETE_LAST, d in preserved_dirs)):
        delete_path = os.path.join(Paths.log_root(), delete_dir)

        if any(name.endswith(".lock") for name in os.listdir(delete_path)):
          continue

        if delete_dir in saved_dirs:
          # nothing else left to delete: forget the oldest saved clip so its segments can roll off
          clip = dashcam.oldest_clip_using(delete_dir)
          if clip is not None:
            cloudlog.warning(f"out of space, dropping oldest dash cam clip {clip.id}")
            dashcam.delete(clip.id)
          break

        if Path(Paths.log_root_external()).is_mount():
          out_of_bytes_external = get_available_bytes(default=MIN_BYTES + 1, path_type="external") < MIN_BYTES
          out_of_percent_external = get_available_percent(default=MIN_PERCENT + 1, path_type="external") < MIN_PERCENT

          if out_of_percent_external or out_of_bytes_external:
            dirs_external = listdir_by_creation(Paths.log_root_external())

            # remove the earliest external directory we can
            for delete_dir_external in sorted(dirs_external, key=lambda d: d in saved_dirs):
              delete_path_external = os.path.join(Paths.log_root_external(), delete_dir_external)
              try:
                cloudlog.warning(f"deleting {delete_path_external}")
                shutil.rmtree(delete_path_external)
                break
              except OSError:
                cloudlog.exception(f"issue deleting {delete_path_external}")

          # move directory from internal to external
          path_external = os.path.join(Paths.log_root_external(), delete_dir)
          try:
            cloudlog.warning(f"moving {delete_path} to {path_external}")
            start = time.monotonic()
            shutil.move(delete_path, path_external)
            cloudlog.warning(f"moved {delete_path} to {path_external} in {time.monotonic() - start:.2f}s")
            break
          except Exception:
            cloudlog.error(f"issue moving {delete_path} to {path_external}")
            try:
              cloudlog.warning(f"deleting {delete_path}")
              shutil.rmtree(delete_path)
              break
            except OSError:
              cloudlog.exception(f"issue deleting {delete_path}")
          continue

        try:
          cloudlog.info(f"deleting {delete_path}")
          shutil.rmtree(delete_path)
          break
        except OSError:
          cloudlog.exception(f"issue deleting {delete_path}")
      exit_event.wait(.1)
    else:
      exit_event.wait(30)


def main():
  deleter_thread(threading.Event())


if __name__ == "__main__":
  main()
