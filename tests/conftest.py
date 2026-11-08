import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "experiments" / "part10", ROOT / "experiments" / "part18", ROOT):
    path = str(entry)
    if path not in sys.path:
        sys.path.insert(0, path)
