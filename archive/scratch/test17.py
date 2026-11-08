import sys; from pathlib import Path; _root = next(p for p in [Path(__file__).resolve().parent] + list(Path(__file__).resolve().parents) if (p / "pyproject.toml").exists()); sys.path.insert(0, str(_root / "src"))
_archive = str(_root / "archive")
import json, time, numpy as np, glider_eco as GE
sp = json.load(open(_root / 'archive/specs/b15m.json'))[0]['runs'][0]
c = sp['c']; e = dict(sp['ecos'][0])
for nm, ex in [("evo", {}), ("blind", dict(blind=1.0)), ("nobrain", dict(kb=0.0, kr=0.0)), ("nopoison", dict(P0=0.0))]:
    t = time.time()
    r = GE.run_eco(c, [dict(e, **ex)], N=4, L=24, seed=7, steps=300, every=100, K=29, p0=[0.0]*29, init="multi", n_init=3, p_spread=0.5, min_sep=1, batch=1,
                   model="poison", opts=dict(coarse=True, fnoise=True, m=5), proj_every=1000)
    rr = r["runs"][0]; o = rr["log"][-1][1]
    print(nm, "n", len(o), "mass", [round(x[0], 2) for x in o], "tot", [round(a, 3) for a, _ in rr["tot"]], "D(eG,eP)", [np.round(x[2][29:31], 6).tolist() for x in o], f"{time.time()-t:.0f}s", flush=True)
