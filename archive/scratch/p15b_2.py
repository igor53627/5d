import sys; from pathlib import Path; _root = next(p for p in [Path(__file__).resolve().parent] + list(Path(__file__).resolve().parents) if (p / "pyproject.toml").exists()); sys.path.insert(0, str(_root / "src"))
_archive = str(_root / "archive")
import pickle, numpy as np, analyze15b as Z, fl_brain as B, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
R = Z.load(); NW = B.NW
names = [f"W1[h{h},s{i}]" for h in range(3) for i in range(3)] + [f"b1[h{h}]" for h in range(3)] + [f"W2[{o},h{h}]" for o in "FRE" for h in range(3)] + [f"b2[{o}]" for o in "FRE"]
grid = np.array([[a, b, c] for a in np.linspace(-1, 0, 5) for b in np.linspace(-.6, .4, 5) for c in np.linspace(0, .25, 5)])
res = {}
for t, r in R.items():
    W, M, _ = Z.plateau_W(r); g = Z.cond(t)
    o_s = Z.brain_batch(W, Z.SSTAR)[:, 0]                       # n × 3
    og = Z.brain_batch(W, grid)                                 # n × 125 × 3
    inp_dep = np.average(og.std(1), axis=0, weights=M)          # input dependence of each output
    sens = np.zeros((NW, 3)); abl = np.zeros((NW, 3)); eps = 1e-3
    for j in range(NW):
        Wp = W.copy(); Wp[:, j] += eps; sens[j] = np.average(np.abs(Z.brain_batch(Wp, Z.SSTAR)[:, 0] - o_s) / eps, axis=0, weights=M)
        Wa = W.copy(); Wa[:, j] = 0; abl[j] = np.average(Z.brain_batch(Wa, Z.SSTAR)[:, 0] - o_s, axis=0, weights=M)
    W0 = np.array([o[2][Z.K0:Z.iD] for o in r["log"][0][1]]); M0 = np.array([o[0] for o in r["log"][0][1]])
    res.setdefault(g, []).append(dict(wbar=np.average(W, 0, weights=M), w0=np.average(W0, 0, weights=M0), sd=W.std(0), sd0=W0.std(0), sens=sens, abl=abl, inp_dep=inp_dep, inp_dep0=np.average(Z.brain_batch(W0, grid).std(1), axis=0, weights=M0)))
pickle.dump(dict(res=res, names=names), open(f"{_archive}/results/eco15/weights15b.pkl", "wb"))
L = []
for g, rs in res.items():
    ip = np.array([x["inp_dep"] for x in rs]); ip0 = np.array([x["inp_dep0"] for x in rs])
    L.append(f"{g:14s} input-dependence (sd over realistic grid) gF {ip[:,0].mean():.3f} gR {ip[:,1].mean():.3f}  (founders {ip0[:,0].mean():.3f} {ip0[:,1].mean():.3f})")
    S = np.mean([x["sens"] for x in rs], 0); A = np.mean([x["abl"] for x in rs], 0)
    top = np.argsort(-S[:, 0])[:4]; topR = np.argsort(-S[:, 1])[:4]
    L.append("   top sensitivity gF: " + ", ".join(f"{names[j]} {S[j,0]:.2f} (ablate {A[j,0]:+.2f})" for j in top))
    L.append("   top sensitivity gR: " + ", ".join(f"{names[j]} {S[j,1]:.2f} (ablate {A[j,1]:+.2f})" for j in topR))
    if len(rs) > 2:
        d = np.array([x["wbar"] - x["w0"] for x in rs]); tt = stats.ttest_1samp(d, 0).statistic; p = stats.ttest_1samp(d, 0).pvalue
        sig = [f"{names[j]} Δ{d[:,j].mean():+.2f} (p={p[j]:.3f})" for j in np.argsort(p)[:5]]
        L.append("   most consistent shifts across seeds: " + ", ".join(sig))
        L.append(f"   within-pop sd: start {np.mean([x['sd0'].mean() for x in rs]):.3f} → plateau {np.mean([x['sd'].mean() for x in rs]):.3f}; b2[F] sd {np.mean([x['sd'][21] for x in rs]):.3f}, W1 mean sd {np.mean([x['sd'][:9].mean() for x in rs]):.3f}, W2 mean sd {np.mean([x['sd'][12:21].mean() for x in rs]):.3f}")
open(f"{_archive}/results/eco15/weights15b.txt", "w").write("\n".join(L)); print("\n".join(L))
# figure
fig, ax = plt.subplots(4, 1, figsize=(15, 13), sharex=True); x = np.arange(NW)
cols = {"evo_4D56": "C3", "frozen_4D56": "C1", "evo_rep_4D56": "C4", "nobrain_4D56": "C0", "evo_5D40": "k"}
for i, (g, c) in enumerate(cols.items()):
    rs = res[g]; o = (i - 2) * .16
    ax[0].bar(x + o, np.mean([r["sens"][:, 0] for r in rs], 0), .16, color=c, label=g)
    ax[1].bar(x + o, np.mean([r["sens"][:, 1] for r in rs], 0), .16, color=c)
    d = np.array([r["wbar"] - r["w0"] for r in rs]); ax[2].bar(x + o, d.mean(0), .16, yerr=d.std(0) / np.sqrt(len(d)) if len(d) > 1 else None, color=c)
    ax[3].bar(x + o, np.mean([r["sd"] for r in rs], 0), .16, color=c)
for a, tl in zip(ax, ["|∂g_F/∂w| at s*", "|∂g_R/∂w| at s*", "Δ mean weight from founders (± SE over seeds)", "within-population sd of weight (plateau)"]): a.set_title(tl, fontsize=10); a.axhline(0, color="k", lw=.5)
ax[3].set_xticks(x); ax[3].set_xticklabels(names, rotation=60, fontsize=8); ax[0].legend(fontsize=8, ncol=5)
fig.tight_layout(); fig.savefig("glider_part15b_weights.png", dpi=80)
