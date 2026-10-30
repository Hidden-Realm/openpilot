#!/usr/bin/env python3
"""
Build step: compile the detector ONNX for the device GPU with tinygrad (the same compile3 path the driver
monitoring model uses).

A failure here must never break the device's build, so it is caught: the detector process then finds no
compiled model and simply stays off, and everything else runs as normal.
"""
import os
import subprocess
import sys

TIMEOUT_S = 40 * 60


def main() -> int:
  compile3, onnx, out = sys.argv[1], sys.argv[2], sys.argv[3]
  try:
    subprocess.run([sys.executable, compile3, onnx, out], check=True, timeout=TIMEOUT_S)
    print(f"worldmap detector compiled: {out}")
  except (subprocess.SubprocessError, OSError) as e:
    print(f"WARNING: worldmap detector failed to compile ({e}); detection will be off", file=sys.stderr)
    try:
      os.remove(out)
    except OSError:
      pass
  return 0


if __name__ == "__main__":
  sys.exit(main())
