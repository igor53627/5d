"""Local bootstrap so `import _path` works from a script directory."""
import importlib.util
from pathlib import Path

_here = Path(__file__).resolve().parent
_repo = next((p for p in [_here, *_here.parents] if (p / "pyproject.toml").exists()), None)
_target = None if _repo is None else _repo / "_path.py"
if _target is not None and _target.resolve() != Path(__file__).resolve():
    _spec = importlib.util.spec_from_file_location("_feigenbaum_lenia_path", _target)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _root = _mod._root
else:
    _root = None
