import _path  # noqa: F401
import json, time, numpy as np, glider_eco as GE, fl_season as Z
sp = json.load(open(_path._root / 'archive/specs/b15m.json'))[0]['runs'][0]
c = sp['c']; e = dict(sp['ecos'][0])
for nm, ex in [("mem", dict(Ts=200.0)), ("nomem", dict(mem=0.0, Ts=200.0))]:
    t = time.time()
    r = GE.run_eco(c, [dict(e, **ex)], N=4, L=24, seed=7, steps=300, every=100, K=Z.KB, p0=[0.0]*Z.KB, init="multi", n_init=3, p_spread=0.5, min_sep=1, batch=1,
                   model="season", opts=dict(coarse=True, fnoise=True, m=5), proj_every=1000)
    rr = r["runs"][0]
    for tt, o in rr["log"]:
        print(nm, tt, "n", len(o), "mass", [round(x[0], 2) for x in o], "h", [round(x[2][Z.iM], 3) for x in o], "D", [np.round(x[2][Z.KB:], 4).tolist() for x in o][:2])
    print(f"{time.time()-t:.0f}s", flush=True)
