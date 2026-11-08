import sys; _R = __import__("pathlib").Path(__file__).resolve().parents[1]; [sys.path.insert(0, str(p)) for p in [_R, _R / "src"] + list((_R / "experiments").glob("part*"))]
"""Сводка поиска (стадии s1–s5): таблица по механизмам + рисунок."""
import json, glob, collections, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows = []
for st in ("s1", "s2", "s3", "s4", "s5"):
    for d in json.load(open(f"results/glider/glider_{st}_summary.json")):
        d["stage"] = st; rows.append(d)
spec = {}
for st in ("s1", "s2", "s3", "s4", "s5_5d"):
    for call in json.load(open(f"specs/glider_{st}.json")):
        for c in call[0]["cands"]: spec[c["seed"]] = c
def mech(d):
    c = spec[d["i"]]; N = "5D" if d["stage"] == "s5" else "4D"
    if c.get("asym", 0) > 0: return f"{N} навязанный дрейф (asym>0)"
    if c["mode"] == "flow": return f"{N} Flow-Lenia" + (" + χ∇φ" if c.get("chi", 0) else "")
    if c.get("lag", 0) or c.get("gam", 0): return f"{N} Lenia + медленный геном (lag/γ)"
    return f"{N} Lenia" + (" + χ∇φ" if c.get("chi", 0) else "")
agg = collections.defaultdict(lambda: dict(n=0, loc=0, static=0, glider_runs=0, gliders=0, best_speed=0.0, best_net=0.0))
for d in rows:
    m = mech(d); a = agg[m]; a["n"] += 1
    a["loc"] += int(1e-3 < d["mfrac"] < 0.3 and d["nloc"] > 0); a["static"] += int(d["ns"] > 0)
    a["glider_runs"] += int(d["ng"] > 0); a["gliders"] += d["ng"]
    if d["ng"] > 0: a["best_speed"] = max(a["best_speed"], d["best_speed"])
    a["best_net"] = max(a["best_net"], d["best_net"])
lines = ["| механизм | кандидатов | с локализованными структурами | со статичными солитонами | с глайдерами (детектор) | всего глайдеров | max скорость глайдера, кл/шаг | max смещение долгого трека, кл |", "|---|---|---|---|---|---|---|---|"]
for m in sorted(agg):
    a = agg[m]; lines.append(f"| {m} | {a['n']} | {a['loc']} | {a['static']} | {a['glider_runs']} | {a['gliders']} | {a['best_speed']:.3f} | {a['best_net']:.1f} |")
open("glider_search.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
cols = {m: plt.cm.tab10(i) for i, m in enumerate(sorted(agg))}
for d in rows:
    m = mech(d); ax[0].scatter(d["best_net"], d["best_str"], s=10 + 15 * d["ng"], color=cols[m], alpha=.7)
    ax[1].scatter(d["best_speed"], d["best_corr"], s=10 + 15 * d["ng"], color=cols[m], alpha=.7)
for m, c in cols.items(): ax[0].scatter([], [], color=c, label=m)
ax[0].axvline(9, color="k", ls=":", lw=.8); ax[0].axhline(0.6, color="k", ls=":", lw=.8)
ax[0].set_xlabel("смещение лучшего долгого локализованного трека, клеток"); ax[0].set_ylabel("прямолинейность"); ax[0].legend(fontsize=7)
ax[1].set_xlabel("скорость, клеток/шаг"); ax[1].set_ylabel("сохранение формы (корр. сопутствующего окна)"); ax[1].axhline(0.6, color="k", ls=":", lw=.8)
fig.suptitle(f"Поиск движущихся структур: {len(rows)} кандидатов (4D 32⁴, 5D 20⁵); размер точки ~ число глайдеров")
fig.tight_layout(); fig.savefig("glider_search_overview.png", dpi=90)
