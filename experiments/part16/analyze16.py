import _path  # noqa: F401
"""Part 16: fusion catastrophe test. python analyze16.py"""
import pickle, glob, re, numpy as np, analyze15b as Z, analyze_eco as AE, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = {}
for f in sorted(glob.glob("results/eco16/b16_*.pkl")) + ["results/eco15/b15d5_0.pkl", "results/eco15/b15d5_1.pkl"]:
    for t, r in pickle.load(open(f, "rb"))["out"][0].items():
        if t == "summary" or not isinstance(r, dict) or not r.get("runs"): continue
        rr = r["runs"][0]; R[t] = dict(log=rr["log"], eco=rr["eco"], partial=bool(rr.get("partial")))
def gs(objs, clamp):
    W = np.array([o[2][Z.K0:Z.iD] for o in objs]); o = Z.brain_batch(W, Z.SSTAR)[:, 0]
    if clamp: o[:, 1] = np.maximum(o[:, 1], 0)
    return o
out = {}
for t, r in R.items():
    log = [x for x in r["log"] if x[0] <= 12000]; cl = r["eco"].get("rclamp", 0) > 0
    ts = []
    for tt, objs in log:
        if not objs: continue
        m = np.array([o[0] for o in objs]); o = gs(objs, cl)
        ts.append((tt, len(objs), m.max(), m.sum(), np.average(o[:, 0], weights=m), np.average(o[:, 1], weights=m)))
    ts = np.array(ts); t_, n, mx, tot = ts[:, 0], ts[:, 1], ts[:, 2], ts[:, 3]
    fused = mx / tot > 0.5; tf = float(t_[np.argmax(fused)]) if fused.any() else np.nan
    pr = {}
    for k, nm in ((0, "gF"), (1, "gR")):
        lg = [(tt, [(o[0], o[1], [float(v)]) for o, v in zip(objs, gs(objs, cl)[:, k])]) for tt, objs in log if tt <= 5000]
        S = AE.selection_diff(lg, 40, lag=5, k=0)[0]
        z = np.array([np.average([o[2][0] for o in objs], weights=[o[0] for o in objs]) if objs else np.nan for _, objs in lg]); dz = float(np.nanmean(z[5:] - z[:-5]))
        pr[nm] = dict(S=float(S), dz=dz, T=dz - float(S))
    out[t] = dict(ts=ts, t_fuse=tf, max_frac=float((mx / tot).max()), max_mass=float(mx.max()), n_min=int(n.min()), n_end=int(n[-1]), gF_end=float(ts[-1, 4]), gR_end=float(ts[-1, 5]),
                  gR_4k=float(ts[np.searchsorted(t_, 4000), 5]), price=pr, partial=r["partial"], T=int(t_[-1]))
pickle.dump(out, open("results/eco16/main16.pkl", "wb"))
L = ["| run | T | fused (max object > 50% of mass) | t_fusion | max object (M1) | max fraction | n min / end | g_F end | g_R 4k / end | Price g_R 0–5k: S / Δ / transmission | Price g_F: S / Δ |", "|" + "---|" * 11]
for t in sorted(out):
    v = out[t]; p = v["price"]
    L.append(f"| {t} | {v['T']} | {'yes' if np.isfinite(v['t_fuse']) else 'no'} | {v['t_fuse']:.0f} | {v['max_mass']:.0f} | {v['max_frac']:.2f} | {v['n_min']} / {v['n_end']} | {v['gF_end']:+.2f} | {v['gR_4k']:+.2f} / {v['gR_end']:+.2f} | {p['gR']['S']:+.3f} / {p['gR']['dz']:+.3f} / {p['gR']['T']:+.3f} | {p['gF']['S']:+.3f} / {p['gF']['dz']:+.3f} |")
open("results/part16_table.md", "w").write("\n".join(L) + "\n\n(t ≤ 12000 for all runs; s401 from Part 15 truncated to 12k; clamp: effective g_R = max(g_R,0); Price over 0–5k, lag 1000 steps)\n"); print("\n".join(L))
fig, ax = plt.subplots(1, 4, figsize=(19, 4.5)); col = {"evo": "C3", "clamp": "C2", "nobrain": "C0"}
for t, v in sorted(out.items()):
    c = col[t.split("_")[0]]; ts = v["ts"]
    for a, j, y in ((ax[0], 2, None), (ax[1], 1, None), (ax[2], 4, None), (ax[3], 5, None)): a.plot(ts[:, 0], ts[:, j], color=c, lw=1.2, label=t)
for a, tl in zip(ax, ["largest object mass (M1)", "number of objects", "population g_F at s*", "population g_R at s* (effective)"]): a.set_title(tl); a.set_xlabel("step")
ax[0].set_yscale("log"); ax[0].legend(fontsize=7); fig.suptitle("Part 16: 5D 40⁵ micro-brain — fusion catastrophe test (red evo, green g_R clamped ≥ 0, blue no brain)"); fig.tight_layout(); fig.savefig("glider_part16_fusion.png", dpi=85)
