import sys; from pathlib import Path; _root = next(p for p in [Path(__file__).resolve().parent] + list(Path(__file__).resolve().parents) if (p / "pyproject.toml").exists()); sys.path.insert(0, str(_root / "src"))
_archive = str(_root / "archive")
import json, numpy as np, copy, itertools
rng = np.random.default_rng(3)
S1 = {c["seed"]: c for call in json.load(open(f"{_archive}/specs/glider_s1.json")) for c in call[0]["cands"]}
S2 = {c["seed"]: c for call in json.load(open(f"{_archive}/specs/glider_s2.json")) for c in call[0]["cands"]}
calls = []; i = 20000
def mk(base, **o):
    global i
    c = copy.deepcopy(base); c.update(o); c["seed"] = i; i += 1; return c
# A: твёрдые шары (#23) + медленный геном-ингибитор
A = [mk(S1[23], gam=g, tau_c=t, beta=0.0, chi=0.0, asym=0.0, mu=float(S1[23]["mu"] * m))
     for g, t, m in itertools.product((0.2, 0.4, 0.6, 0.8), (5.0, 20.0, 50.0, 100.0), (1.0, 0.92))]
# B: однокольцевые пятна (#37 без асимметрии) + ингибитор
Bc = [mk(S1[37], gam=g, tau_c=t, beta=0.0, chi=0.0, asym=0.0, mu=float(S1[37]["mu"] * m))
      for g, t, m in itertools.product((0.1, 0.3, 0.5, 0.8), (5.0, 20.0, 50.0, 100.0), (1.0,))]
# C: Flow + χ + ингибитор вокруг лучших спонтанных движков
Cc = [mk(S2[b], gam=g, tau_c=t, chi=ch, asym=0.0) for b in (10110, 10073, 10079, 10085)
      for g, t, ch in [(0.0, 20.0, 1.0), (0.3, 20.0, 1.0), (0.3, 100.0, 1.0), (0.6, 100.0, 1.0), (0.0, 20.0, 2.0), (0.3, 100.0, 2.0), (0.6, 50.0, 2.0), (0.0, 100.0, 0.5)]]
for grp in (A[:16], A[16:], Bc, Cc[:16], Cc[16:]):
    calls.append([dict(cands=grp, N=4, L=32, steps=4000, every=25)])
json.dump(calls, open(f"{_archive}/specs/glider_s3.json", "w")); print(len(calls), i - 20000)
