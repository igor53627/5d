"""Часть 9: отбор — ⟨p0⟩(t) при отборе (pk=1) и нейтрально (pk=0), парные Δ, наследуемость родитель→потомок."""
import pickle, glob, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import analyze_eco as AE
files = ["results/eco/eco5_0.pkl", "results/eco/eco5_3.pkl"] + sorted(glob.glob("results/eco/eco6_[0-5].pkl"))
fig, ax = plt.subplots(1, 3, figsize=(17, 5)); rows = []; paired = {}
for f in files:
    o = pickle.load(open(f, "rb"))["out"][0]; tag = o.get("tag", ""); kind = "single" if "single" in tag else "multi"
    for r in o["runs"]:
        ps = AE.pstats(r["log"], 0); t = [x[0] for x in ps]; m = np.array([x[1] for x in ps])
        sel = r["eco"]["pk"] > 0; s = AE.summarize(r, 64); h, n = AE.herit(s["pairs"], 0)
        ax[0 if kind == "single" else 1].plot(t, m, color="tab:red" if sel else "tab:gray", lw=1.2 if sel else 0.9, alpha=.85)
        d = float(np.nanmean(m[-10:]) - m[0]); paired.setdefault((kind, tag), {})["sel" if sel else "neu"] = d
        rows.append(dict(file=f.split("/")[-1], tag=tag, kind=kind, pk=r["eco"]["pk"], steps=len(r["log"]) * 100, n_div=s["n_div"], herit_p0=h, n_pairs=n, dp0=d, n_end=s["n"][-1], M_end=s["M"][-1]))
for k, a in enumerate(ax[:2]):
    a.plot([], [], color="tab:red", label="отбор: keat·e^{p0}"); a.plot([], [], color="tab:gray", label="нейтрально (p ни на что не влияет)")
    a.set_xlabel("шаг"); a.set_ylabel("⟨p0⟩ по объектам (массо-взвеш.)"); a.grid(alpha=.3); a.legend(fontsize=8)
ax[0].set_title("старт с ОДНОГО солитона (p=0, мутации 0.1)"); ax[1].set_title("старт с 6 солитонов, p ~ N(0, 0.5)")
D = [(k, v["sel"] - v["neu"]) for k, v in paired.items() if "sel" in v and "neu" in v]
ax[2].bar(range(len(D)), [d for _, d in D], color="tab:red"); ax[2].set_xticks(range(len(D))); ax[2].set_xticklabels([k[1] for k, _ in D], rotation=45, fontsize=7)
ax[2].axhline(0, color="k", lw=.8); ax[2].set_ylabel("Δ⟨p0⟩(отбор) − Δ⟨p0⟩(нейтр.)"); ax[2].set_title(f"парная разница: >0 в {sum(d > 0 for _, d in D)}/{len(D)}")
fig.tight_layout(); fig.savefig("glider_part9_selection.png", dpi=90)
# контроль без переноса и без пищи
for f in ["results/eco/eco5_1.pkl", "results/eco/eco5_2.pkl"]:
    o = pickle.load(open(f, "rb"))["out"][0]; r = o["runs"][0]; s = AE.summarize(r, 64); h, n = AE.herit(s["pairs"], 0)
    ps = AE.pstats(r["log"], 0); rows.append(dict(file=f.split("/")[-1], tag=o["tag"], kind="multi", pk=r["eco"]["pk"], steps=len(r["log"]) * 100, n_div=s["n_div"], herit_p0=h, n_pairs=n, dp0=float(np.nanmean([x[1] for x in ps[-10:]]) - ps[0][1]), n_end=s["n"][-1], M_end=s["M"][-1]))
json.dump(rows, open("results/part9_runs.json", "w"), indent=0)
L = ["| прогон | старт | p влияет | шагов | делений | corr(p0 родителя, p0 потомка) (пар) | Δ⟨p0⟩ | объектов в конце | масса в конце /M1 |", "|---|---|---|---|---|---|---|---|---|"]
for r in rows: L.append(f"| {r['tag']} | {r['kind']} | {'да' if r['pk'] else 'нет'} | {r['steps']} | {r['n_div']} | {r['herit_p0']:.2f} ({r['n_pairs']}) | {r['dp0']:+.2f} | {r['n_end']} | {r['M_end']:.1f} |")
open("results/part9_table.md", "w").write("\n".join(L) + "\n"); print("\n".join(L)); print(D)
