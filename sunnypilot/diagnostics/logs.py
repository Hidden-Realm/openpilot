"""
Where diagnostics logs live, and packing recent ones up for download from the Wi-Fi viewers.

One JSON-lines file per boot, newest kept, oldest removed past a count and size limit. The bundle is a zip of
the most recent files, ready to hand to whoever is looking into a problem.
"""
import io
import os
import time
import zipfile

MAX_FILES = 20
MAX_BYTES = 50 * 1024 ** 2
BUNDLE_FILES = 5


def diag_dir() -> str:
  from openpilot.system.hardware.hw import Paths
  return os.path.join(os.path.dirname(os.path.normpath(Paths.log_root())), "diagnostics")


def list_logs(root: str) -> list[tuple[str, int, float]]:
  """(name, bytes, mtime) of every log, newest first."""
  try:
    names = [n for n in os.listdir(root) if n.endswith(".jsonl")]
  except OSError:
    return []
  out = []
  for n in names:
    try:
      st = os.stat(os.path.join(root, n))
      out.append((n, st.st_size, st.st_mtime))
    except OSError:
      pass
  return sorted(out, key=lambda r: r[2], reverse=True)


def trim(root: str, keep: str | None = None) -> None:
  """Remove the oldest logs beyond MAX_FILES or MAX_BYTES (never `keep`, the one being written)."""
  total = 0
  for i, (name, size, _) in enumerate(list_logs(root)):
    total += size
    if name != keep and (i >= MAX_FILES or total > MAX_BYTES):
      try:
        os.remove(os.path.join(root, name))
      except OSError:
        pass


def bundle(root: str, max_files: int = BUNDLE_FILES) -> bytes:
  buf = io.BytesIO()
  # strict_timestamps off: before the clock syncs the device thinks it's 1970, which ZIP can't store
  with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, strict_timestamps=False) as z:
    logs = list_logs(root)[:max_files]
    for name, _, _ in logs:
      try:
        z.write(os.path.join(root, name), name)
      except OSError:
        pass
    if not logs:
      z.writestr("README.txt", "No diagnostics logs yet. They are recorded while the device is on.\n")
  return buf.getvalue()


def bundle_name() -> str:
  return f"sunnypilot-diagnostics-{time.strftime('%Y%m%d-%H%M')}.zip"
