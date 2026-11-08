import _path  # noqa: F401
_root = _path._root
_archive = str(_root / "archive") if _root is not None else ""
"""λc (условный показатель) и λ (собственный) при фиксированном c под входом MG — для выбора знака gsign."""
import json, sys, numpy as np
from reservoir_gpu import run_batch
L = int(sys.argv[1]) if len(sys.argv) > 1 else 6
el = []
for a in (0.2, -0.05):
    for c in (2.0, 2.5, 2.8, 3.5, 4.2, 5.0, 6.0, 7.0):
        el.append(dict(method="fixed", a=a, gamma=2.0, eps=0.1, c0=c, seed=0, rep=0))
        el.append(dict(method="fixed", a=a, gamma=2.0, eps=0.1, c0=c, seed=0, rep=1))
res = run_batch("mg", 4, L, el)
for r in res["runs"]:
    print(f"a={r['a']:+.2f} c={r['c0']:.1f}  λc={r['lamc']:+.4f} λ={r['lam']:+.4f} chaos={r['chaos']:.2f} twin={r.get('twin_corr', float('nan')):.3f} valMG={np.round(r['val'],3)}")
json.dump(res, open(f"{_archive}/results/sweep_lamc_L{L}.json", "w"))
