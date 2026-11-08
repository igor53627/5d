"""
Path setup for running scripts without pip install.

This module finds the repo root by looking for pyproject.toml, then adds
src/ and experiments/part* to sys.path. Works when:
- Running from repo root: python experiments/part06/glider_cont.py
- Running from script dir: cd experiments/part06 && python glider_cont.py
- Running from anywhere with absolute path

Usage: import this module at the top of experiment/archive scripts.
    import _path  # noqa: F401
"""
import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    """Walk up from start until we find pyproject.toml."""
    for parent in [start] + list(start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"Cannot find repo root (pyproject.toml) from {start}")


# Find repo root from this file's location
_root = _find_repo_root(Path(__file__).resolve().parent)

# Add paths
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))
for _p in (_root / "experiments").glob("part*"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
