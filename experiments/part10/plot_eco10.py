import sys; from pathlib import Path; _root = next(p for p in [Path(__file__).resolve().parent] + list(Path(__file__).resolve().parents) if (p / "pyproject.toml").exists()); sys.path.insert(0, str(_root / "src"))
"""Часть 10: популяция при ёмкости, наследуемость, межособевой отбор (Прайс) — отбор / нейтр. / без переноса, 3 зерна."""
import glob, pickle, json, sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
import analyze_eco as AE
L = 96; runs = {}
for f in sorted(glob.glob("results/eco10/eco10m_*.pkl")):
    o = pickle.load(open(f, "rb"))["out"][0]; tag = o["tag"]; cond, sd = tag.split("_s"); r = o["runs"][0] if o["runs"] else None
    if r is None: continue
    s = AE.summarize(r, L); S = AE.selection_diff(r["log"], L, lag=10, t_min=2000); h = AE.herit(s["pairs"], 0)
    ps = AE.pstats(r["log"], 0)
    win = lambda a, b: AE.selection_diff([x for x in r["log"] if a <= x[0] <= b], L, lag=5)
    Se, Sl = win(0, 7000), win(10000, 25000)           # фаза роста (окна до 5000 + лаг) и фаза ёмкости
    runs.setdefault(cond, []).append(dict(seed=int(sd), t=[x[0] for x in r["log"]], n=s["n"], M=s["M"], p0=[x[1] for x in ps], p0sd=[x[2] for x in ps],
        div=s["n_div"], h=h, S=S, Se=Se, Sl=Sl, dp0=float(np.nanmean([x[1] for x in ps[-5:]]) - ps[0][1]), partial=o.get("partial"), T=r["log"][-1][0], p_init=o.get("p_init")))
COL = {"sel": "tab:red", "neu": "tab:gray", "notr": "tab:blue"}; LAB = {"sel": "отбор (keat·e^{p0})", "neu": "нейтрально (p ни на что не влияет)", "notr": "отбор, но БЕЗ переноса p"}
fig, ax = plt.subplots(2, 3, figsize=(17, 9)); ax = ax.ravel()
for cond, R in runs.items():
    for i, r in enumerate(R):
        kw = dict(color=COL[cond], lw=1, alpha=.85, label=LAB[cond] if i == 0 else None)
        ax[0].plot(r["t"], r["n"], **kw); ax[1].plot(r["t"], r["M"], **kw); ax[2].plot(r["t"], r["p0"], **kw)
        ax[3].plot(r["t"], r["p0sd"], **kw)
for a, yl in zip(ax[:4], ["число особей (≥0.3 M1)", "общая масса / M1", "⟨p0⟩ (массо-взвеш. по особям)", "sd p0 между особями"]):
    a.set_xlabel("шаг"); a.set_ylabel(yl); a.grid(alpha=.3)
ax[0].legend(fontsize=8)
conds = [c for c in ("sel", "neu", "notr") if c in runs]
a = ax[4]
for i, cnd in enumerate(conds):
    vals = [r["dp0"] for r in runs[cnd]]
    a.scatter([i] * len(vals), vals, color=COL[cnd], s=60, zorder=3); a.bar(i, np.nanmean(vals), color=COL[cnd], alpha=.3)
a.set_xticks(range(len(conds))); a.set_xticklabels([{"sel": "отбор", "neu": "нейтрально", "notr": "без переноса p\n(p→0 сразу)"}[c] for c in conds]); a.axhline(0, color="k", lw=.8); a.set_ylabel("Δ⟨p0⟩ за прогон (конец − старт)"); a.grid(alpha=.3)
a = ax[5]
for i, cnd in enumerate(conds):
    for j, (key, lab) in enumerate([("Se", "рост 0–5k"), ("Sl", "ёмкость 10–25k")]):
        vals = [r[key][0] for r in runs[cnd]]; x = i * 3 + j
        a.scatter([x] * len(vals), vals, color=COL[cnd], s=50, zorder=3, marker="o" if j == 0 else "s"); a.bar(x, np.nanmean(vals), color=COL[cnd], alpha=.3 if j == 0 else .15)
a.set_xticks([i * 3 + j for i in range(len(conds)) for j in range(2)]); a.set_xticklabels([f"{c}\n{w}" for c in conds for w in ("рост", "ёмкость")], fontsize=8)
a.axhline(0, color="k", lw=.8); a.set_ylabel("S = cov(p0, w)/⟨w⟩ (Прайс, лаг 1000 шагов)"); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig("glider_part10_population.png", dpi=90); print("fig")
rows = ["| условие | зерно | шагов | особей в конце (макс) | масса в конце | делений | corr(p0 родитель, потомок) (пар) | S рост 0–5k (corr) | S ёмкость 10–25k (corr) | Δ⟨p0⟩ |", "|---|---|---|---|---|---|---|---|---|---|"]
for cnd in conds:
    for r in runs[cnd]:
        rows.append(f"| {LAB[cnd]} | {r['seed']} | {r['T']} | {r['n'][-1]} ({max(r['n'])}) | {r['M'][-1]:.0f} | {r['div']} | {r['h'][0]:.2f} ({r['h'][1]}) | {r['Se'][0]:+.3f} ({r['Se'][1]:+.2f}) | {r['Sl'][0]:+.3f} ({r['Sl'][1]:+.2f}) | {r['dp0']:+.2f} |")
st = {}
if "sel" in runs and "neu" in runs:
    a = [r["dp0"] for r in runs["sel"]]; b = [r["dp0"] for r in runs["neu"]]; st["dp0_sel_vs_neu_welch_p"] = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    for key in ("Se", "Sl"):
        a = [r[key][0] for r in runs["sel"]]; b = [r[key][0] for r in runs["neu"]]
        st[f"{key}_sel_vs_neu_welch_p"] = float(stats.ttest_ind(a, b, equal_var=False).pvalue); st[f"{key}_sel_mean"] = float(np.mean(a)); st[f"{key}_neu_mean"] = float(np.mean(b))
        st[f"{key}_sel_one_sample_p"] = float(stats.ttest_1samp(a, 0).pvalue)
    a = [r["h"][0] for r in runs["sel"] + runs["neu"]]; b = [r["h"][0] for r in runs.get("notr", [])]
    if b: st["h_tr_vs_notr_welch_p"] = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    st["n_end"] = {c: [r["n"][-1] for r in runs[c]] for c in conds}
    a = [r["n"][-1] for r in runs["sel"]]; b = [r["n"][-1] for r in runs["neu"]]; st["n_sel_vs_neu_welch_p"] = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
if "notr" in runs: st["h_notr"] = [r["h"][0] for r in runs["notr"]]
open("results/part10_table.md", "w").write("\n".join(rows) + "\n\n" + json.dumps(st, ensure_ascii=False) + "\n"); print("\n".join(rows)); print(st)
