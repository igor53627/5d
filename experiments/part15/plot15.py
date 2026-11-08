import sys; from pathlib import Path; _root = next(p for p in [Path(__file__).resolve().parent] + list(Path(__file__).resolve().parents) if (p / "pyproject.toml").exists()); sys.path.insert(0, str(_root / "src"))
import pickle, re, sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = pickle.load(open("results/eco15/main.pkl", "rb")); R5 = pickle.load(open("results/eco15/d5.pkl", "rb")) if len(sys.argv) > 1 else {}
G = {}
for t, v in R.items(): G.setdefault(re.sub(r"_s\d+$", "", t), []).append(v)
conds = [("evo_4D56", "evo", "C3"), ("frozen_4D56", "frozen (no mut.)", "C1"), ("evo_rep_4D56", "evo + krep 20", "C4"), ("nobrain_4D56", "no brain (weights drift)", "C0")]
fig, ax = plt.subplots(2, 3, figsize=(16, 8.5)); ax = ax.ravel()
for a, (k, lab) in zip(ax, [("n", "number of individuals"), ("gF", "chemotaxis gain g_F(s=0)"), ("CI", "chemotaxis index v·ĝ/|v|"), ("intake", "intake per mass per step"), ("wsd", "weight diversity (mean sd)"), ("wdist", "|Δ mean weights| from start")]):
    for g, lb, c in conds:
        rs = G.get(g, [])
        if not rs: continue
        T = min(len(r["t"]) for r in rs); Y = np.array([(r["n"] if k == "n" else r["ts"][k])[:T] for r in rs], float); t = rs[0]["t"][:T]
        m = np.nanmean(Y, 0); se = np.nanstd(Y, 0) / np.sqrt(len(rs)); a.plot(t, m, color=c, label=f"{lb} (n={len(rs)})"); a.fill_between(t, m - se, m + se, color=c, alpha=.2)
    for t5, v in R5.items():
        y = v["n"] if k == "n" else v["ts"][k]; a.plot(v["t"], y, "--", color="C3" if t5.startswith("evo") else "C0", lw=1, label=f"5D {t5.split('_')[0]}")
    a.set_title(lab); a.set_xlabel("step")
ax[0].legend(fontsize=7); fig.suptitle("Part 15: micro-brain, 4D 56⁴ (mean ± SE over seeds), dashed: 5D 40⁵ (1 seed)"); fig.tight_layout(); fig.savefig("glider_part15_timeseries.png", dpi=90)
