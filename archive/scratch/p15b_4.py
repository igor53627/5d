import sys; from pathlib import Path; _root = next(p for p in [Path(__file__).resolve().parent] + list(Path(__file__).resolve().parents) if (p / "pyproject.toml").exists()); sys.path.insert(0, str(_root / "src"))
_archive = str(_root / "archive")
import pickle, numpy as np, analyze15b as Z, analyze_eco as AE, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = Z.load()
def series(t):
    log = R[t]["log"]; out = []
    for tt, objs in log:
        if not objs: continue
        W = np.array([o[2][Z.K0:Z.iD] for o in objs]); m = np.array([o[0] for o in objs]); o = Z.brain_batch(W, Z.SSTAR)[:, 0]
        c = np.corrcoef(o[:, 0], o[:, 1])[0, 1] if len(o) > 3 else np.nan
        out.append((tt, len(objs), m.max(), np.average(o[:, 0], weights=m), np.average(o[:, 1], weights=m), o[:, 0].std(), o[:, 1].std(), c, np.average(W, 0, weights=m)))
    return out
def logaug(t, k):
    log = R[t]["log"]; return [(tt, [(o[0], o[1], [float(Z.brain_batch(np.array([o[2][Z.K0:Z.iD]]), Z.SSTAR)[0, 0, k])]) for o in objs]) for tt, objs in log]
res = {}
for t in ["evo_5D40_s401", "nobrain_5D40_s401"] + [f"evo_4D56_s{s}" for s in range(311, 317)]:
    S = series(t); res[t] = S
    tt = np.array([x[0] for x in S]); gF = np.array([x[3] for x in S]); gR = np.array([x[4] for x in S])
    early = tt <= 4000
    # Price for gF and gR in early window (0..4000): S vs Δz per 1000 steps
    pr = {}
    for k, nm in ((0, "gF"), (1, "gR")):
        lg = [x for x in logaug(t, k) if x[0] <= 5000]; Sx = AE.selection_diff(lg, R[t]["L"], lag=5, k=0)[0]
        z = np.array([np.average([o[2][0] for o in objs], weights=[o[0] for o in objs]) if objs else np.nan for _, objs in lg]); dz = np.nanmean(z[5:21] - z[0:16])
        pr[nm] = (Sx, dz)
    W0 = R[t]["log"][0][1]; Wf = np.array([o[2][Z.K0:Z.iD] for o in W0]); mf = np.array([o[0] for o in W0])
    gavg = Z.brain_batch(np.average(Wf, 0, weights=mf)[None], Z.SSTAR)[0, 0]
    print(f"{t:18s} early(≤4k) corr(gF,gR) among individuals {np.nanmean([x[7] for x in S if x[0] <= 4000]):+.2f}; gR at 2k/4k/end {gR[np.searchsorted(tt,2000)]:+.2f}/{gR[np.searchsorted(tt,4000)]:+.2f}/{gR[-1]:+.2f}; "
          f"Price(0-5k) gF S={pr['gF'][0]:+.3f} dz={pr['gF'][1]:+.3f}, gR S={pr['gR'][0]:+.3f} dz={pr['gR'][1]:+.3f}; brain(mean founder w) gF {gavg[0]:+.2f} gR {gavg[1]:+.2f}; end gF {gF[-1]:+.2f} gR {gR[-1]:+.2f}")
pickle.dump({t: [x[:8] for x in S] for t, S in res.items()}, open(f"{_archive}/results/eco15/fused15b.pkl", "wb"))
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for t, S in res.items():
    tt = [x[0] for x in S]; c = "k" if "5D" in t and "evo" in t else ("c" if "5D" in t else "C3"); lw = 2 if "5D" in t else .8; lb = t if ("5D" in t or t.endswith("311")) else None
    ax[0].plot(tt, [x[3] for x in S], color=c, lw=lw, label=lb); ax[1].plot(tt, [x[4] for x in S], color=c, lw=lw); ax[2].plot(tt, [x[2] for x in S], color=c, lw=lw)
for a, tl in zip(ax, ["population g_F at s*", "population g_R at s* (− = weaker repulsion)", "largest object mass (M1)"]): a.set_title(tl); a.set_xlabel("step"); a.axhline(0, color="k", lw=.4)
ax[2].set_yscale("log"); ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig("glider_part15b_fused.png", dpi=85)
