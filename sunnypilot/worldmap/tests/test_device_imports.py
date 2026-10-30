"""The device's Python only has the runtime dependencies in pyproject.toml. A world map module importing a
development-only package (OpenCV above all) would crash the manager at boot, so it is caught here."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV_ONLY = {"cv2", "scipy", "torch", "onnx", "onnxruntime", "ultralytics", "matplotlib", "pandas", "sklearn"}


def imported_packages(path: Path) -> set[str]:
  tree = ast.parse(path.read_text(encoding="utf-8"))
  names = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      names.update(a.name.split(".")[0] for a in node.names)
    elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
      names.add(node.module.split(".")[0])
  return names


def test_no_development_only_imports():
  bad = {}
  for path in ROOT.rglob("*.py"):
    if "tests" in path.parts:
      continue
    found = imported_packages(path) & DEV_ONLY
    if found:
      bad[str(path.relative_to(ROOT))] = sorted(found)
  assert not bad, f"not installed on the device: {bad}"
