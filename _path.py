"""
Path setup for running scripts without pip install -e.

Usage: import this module at the top of experiment/archive scripts.
    import _path  # noqa: F401
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))
for _p in (_root / "experiments").glob("part*"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
