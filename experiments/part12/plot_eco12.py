import sys; from pathlib import Path; _root = next(p for p in [Path(__file__).resolve().parent] + list(Path(__file__).resolve().parents) if (p / "pyproject.toml").exists()); sys.path.insert(0, str(_root / "src"))
"""Часть 12: отбор и разнообразие при ёмкости (fl_eco3 all5 везде)."""
import glob, pickle, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
import analyze12 as A
R = {}
for f in sorted(glob.glob("results/eco12/eco12m_*.pkl")):
    o = pickle.load(open(f, "rb"))["out"][0]
    for t, r in o.items():
        if t == "summary" or not r.get("runs"): continue
        cond, geo, sd = t.rsplit("_", 2); L = {"96": 96, "128": 128, "5D40": 40}[geo]
        rr = r["runs"][0]; m = A.metrics(rr, L, eco=rr["eco"]); m.update(seed=int(sd[1:]), partial=r.get("partial"), geo=geo)
        R.setdefault((cond, geo), []).append(m)
pickle.dump(R, open("results/eco12/main_metrics.pkl", "wb"))
COL = {"sel": "tab:red", "neu": "tab:gray", "res2k": "tab:green", "seasonK1": "tab:purple"}
LAB = {"sel": "направленный отбор (e^{p0})", "neu": "нейтрально", "res2k": "2 ресурса, компромисс (γ=2)", "seasonK1": "сезонный оптимум"}
C4 = [c for c in ("sel", "neu", "res2k", "seasonK1") if (c, "96") in R]
fig, ax = plt.subplots(2, 3, figsize=(18, 9.5)); ax = ax.ravel()
for c in C4:
    for i, m in enumerate(R[(c, "96")]):
        kw = dict(color=COL[c], lw=1, alpha=.8, label=LAB[c] if i == 0 else None)
        ax[0].plot(m["t"], m["n"], **kw); ax[1].plot(m["t"], m["sd"], **kw); ax[2].plot(m["t"], m["p0"], **kw)
        ax[4].plot(m["t"], m["lin_eff"], **kw)
    Sw = np.array([[x[1] for x in m["Sw"]] for m in R[(c, "96")]]); tw = [x[0] + 1000 for x in R[(c, "96")][0]["Sw"]]
    ax[3].errorbar(tw, Sw.mean(0), yerr=Sw.std(0, ddof=1) / np.sqrt(len(Sw)) if len(Sw) > 1 else None, color=COL[c], capsize=2, label=LAB[c])
for a, yl in zip(ax[:5], ["число особей", "sd p0 между особями", "⟨p0⟩", "S Прайса (окна 2000 шагов), среднее ± SE по зёрнам", "эфф. число линий основателей (exp H)"]):
    a.set_xlabel("шаг"); a.set_ylabel(yl); a.grid(alpha=.3)
ax[3].axhline(0, color="k", lw=.8); ax[0].legend(fontsize=8); ax[3].legend(fontsize=7)
a = ax[5]; bins = np.linspace(-1, 1, 41)
for c in C4:
    x = np.concatenate([m["p0_last"] for m in R[(c, "96")]]); a.hist(x, bins=bins, histtype="step", density=True, color=COL[c], lw=2, label=LAB[c])
a.set_xlabel("p0 особей в конце (все зёрна)"); a.set_ylabel("плотность"); a.legend(fontsize=7); a.grid(alpha=.3)
fig.suptitle("Часть 12: 4D 96⁴, 25 000 шагов, fl_eco3 all5 во всех условиях"); fig.tight_layout(); fig.savefig("glider_part12_population.png", dpi=85)
rows = ["| условие | решётка | зерно | особей на плато | sd p0 на плато | ⟨p0⟩ в конце | S на плато (corr) | наследуемость | делений | мод p0 (GMM) | линий осн. (эфф.) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for (c, g), ms in R.items():
    for m in sorted(ms, key=lambda m: m["seed"]):
        rows.append(f"| {LAB.get(c, c)} | {g} | {m['seed']} | {m['n_pl']:.0f} | {m['sd_pl']:.2f} | {m['p0'][-1]:+.2f} | {m['S_plateau']:+.3f} ({m['corr_plateau']:+.2f}) | {m['h']:.2f} | {m['div']} | {m['k_mode']} | {m['lin_n'][-1]} ({m['lin_eff'][-1]:.1f}) |")
st = {}
g = lambda c, k, geo="96": [m[k] for m in R.get((c, geo), [])]
def welch(a, b): return float(stats.ttest_ind(a, b, equal_var=False).pvalue) if len(a) > 1 and len(b) > 1 else None
for c in C4:
    st[c] = dict(n=len(g(c, "S_plateau")), S_plateau_mean=float(np.mean(g(c, "S_plateau"))), S_plateau_p_vs0=float(stats.ttest_1samp(g(c, "S_plateau"), 0).pvalue) if len(g(c, "S_plateau")) > 1 else None,
                 sd_pl_mean=float(np.mean(g(c, "sd_pl"))), h_mean=float(np.mean(g(c, "h"))), n_pl=float(np.mean(g(c, "n_pl"))))
    if c != "neu":
        st[c]["S_plateau_vs_neu_p"] = welch(g(c, "S_plateau"), g("neu", "S_plateau")); st[c]["sd_pl_vs_neu_p"] = welch(g(c, "sd_pl"), g("neu", "sd_pl"))
        st[c]["sd_pl_vs_sel_p"] = welch(g(c, "sd_pl"), g("sel", "sd_pl")) if c != "sel" else None
# рост sd на плато (sd в конце − sd в середине)
for c in C4:
    d = [m["sd"][-1] - m["sd"][len(m["sd"]) // 2] for m in R[(c, "96")]]; st[c]["dsd_plateau_mean"] = float(np.mean(d)); st[c]["dsd_plateau_p_vs0"] = float(stats.ttest_1samp(d, 0).pvalue) if len(d) > 1 else None
    Sw = np.array([[x[1] for x in m["Sw"]] for m in R[(c, "96")]]); late = Sw[:, len(Sw[0]) // 2:].mean(1); st[c]["S_late_windows_mean"] = float(late.mean())
    st[c]["S_late_windows_p_vs0"] = float(stats.ttest_1samp(late, 0).pvalue) if len(late) > 1 else None
open("results/part12_table.md", "w").write("\n".join(rows) + "\n\n```\n" + json.dumps(st, ensure_ascii=False, indent=1) + "\n```\n")
print("\n".join(rows)); print(json.dumps(st, ensure_ascii=False, indent=1))
