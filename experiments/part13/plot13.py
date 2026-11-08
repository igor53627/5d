import sys; from pathlib import Path; _root = next(p for p in [Path(__file__).resolve().parent] + list(Path(__file__).resolve().parents) if (p / "pyproject.toml").exists()); sys.path.insert(0, str(_root / "src"))
import pickle, re, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = pickle.load(open("results/eco13/main13.pkl", "rb"))
G = {}
for t, v in R.items(): G.setdefault(re.sub(r"_s\d+$", "", t), []).append(v)
dims = [("2D512", "2D (12 founders)"), ("2D512n48", "2D (48 founders)"), ("3D96", "3D 96³"), ("4D56", "4D 56⁴"), ("5D40", "5D 40⁵")]
cols = dict(zip([d for d, _ in dims], ["#999", "#555", "C0", "C2", "C3"]))
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
def ts(g, k):
    rs = G.get(g, []); 
    if not rs: return None
    T = min(len(r["ts"]["t"]) for r in rs); Y = np.array([r["ts"][k][:T] for r in rs], float)
    return rs[0]["ts"]["t"][:T], np.nanmean(Y, 0), np.nanstd(Y, 0) / np.sqrt(len(rs))
for j, k in enumerate(["n", "p0", "sd"]):
    for d, lab in dims:
        for cn, ls in (("sel", "-"), ("neu", "--")):
            z = ts(f"{cn}_{d}", k)
            if z is None: continue
            t, m, se = z; ax[j].plot(t, m, ls, color=cols[d], label=f"{lab} {cn}" if j == 0 else None); ax[j].fill_between(t, m - se, m + se, color=cols[d], alpha=.15)
    ax[j].set_title({"n": "number of individuals", "p0": "⟨p0⟩", "sd": "sd(p0)"}[k]); ax[j].set_xlabel("step")
ax[0].legend(fontsize=7, ncol=2)
bars = [("dp0_5k", "Δ⟨p0⟩ in first 5k steps"), ("dp0", "Δ⟨p0⟩ total (30k)"), ("S_growth", "Price S, growth phase"), ("sd_pl", "sd(p0) on plateau"),
        ("h", "heritability h (parent→child)"), ("div_rate_k", "divisions / individual / 1000 steps"), ("merge_rate_k", "merges / individual / 1000 steps (upper bound)"), ("t_plateau", "t_plateau (steps)"), ("lin_eff", "effective founder lineages (end)")]
fig2, ax2 = plt.subplots(3, 3, figsize=(15, 11)); axs = ax2.ravel()
for a, (k, lab) in zip(axs, bars):
    x = np.arange(len(dims))
    for o, cn, c in ((-.2, "sel", "C3"), (.2, "neu", "C0")):
        m = [np.nanmean([r[k] for r in G.get(f"{cn}_{d}", [])]) if G.get(f"{cn}_{d}") else np.nan for d, _ in dims]
        se = [np.nanstd([r[k] for r in G.get(f"{cn}_{d}", [])], ddof=1) / np.sqrt(len(G[f"{cn}_{d}"])) if len(G.get(f"{cn}_{d}", [])) > 1 else 0 for d, _ in dims]
        a.bar(x + o, m, .4, yerr=se, color=c, alpha=.7, label=cn)
        for i, (d, _) in enumerate(dims):
            v = [r[k] for r in G.get(f"{cn}_{d}", [])]; a.plot([i + o] * len(v), v, "k.", ms=3)
    a.set_xticks(x); a.set_xticklabels([l for _, l in dims], fontsize=7, rotation=15); a.set_title(lab, fontsize=9); a.axhline(0, color="k", lw=.5)
axs[0].legend()
fig.tight_layout(); fig.savefig("glider_part13_timeseries.png", dpi=90); fig2.tight_layout(); fig2.savefig("glider_part13_metrics.png", dpi=90)
