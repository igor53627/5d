"""Put src/ and experiments/part*/ on sys.path.

No-op when this file is not inside a checkout (Modal copies it to /root
with the worker scripts, and those imports resolve as siblings).
"""
import sys
from pathlib import Path


def install(start=None):
    start = Path(start) if start else Path(__file__).resolve().parent
    root = next((p for p in [start, *start.parents] if (p / "pyproject.toml").exists()), None)
    if root is None:
        return None
    entries = [root / "src"]
    experiments = root / "experiments"
    if experiments.is_dir():
        entries.extend(sorted(p for p in experiments.glob("part*") if p.is_dir()))
    for entry in entries:
        path = str(entry)
        if path not in sys.path:
            sys.path.insert(0, path)
    return root


_root = install()
