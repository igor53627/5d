import sys; _R = __import__("pathlib").Path(__file__).resolve().parents[2]; [sys.path.insert(0, str(p)) for p in [_R, _R / "src"] + list((_R / "experiments").glob("part*"))]
"""Часть 8: таблица частоты репликации, фигура (сетка исходов, дочерние v(m), каскад), mp4 события и каскада."""
import sys, os, glob, pickle, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.animation as an
V = {"R2g5": "results/rep/repver2_0.pkl", "R10g3": "results/rep/repver1_1.pkl", "R2g4": "results/rep/repver1_0.pkl", "R11g5": "results/rep/repver1_2.pkl"}
def load(f):
    o = pickle.load(open(f, "rb"))["out"][0]; runs = []; meta = None
    for k, v in o["runs"].items():
        meta = v; runs += v.get("runs", [])
    return runs, meta
def is_rep(r, thr=0.45): return r["n_end"] >= 3 and sorted(r["masses"])[-3] >= thr
lines = ["| особь | v одиночного (4D 64⁴) | столкновений | ≥3 объекта | репликация (3 объекта ≥0.45 M1) | строго ≥0.6 M1 | слияние в 1 |", "|---|---|---|---|---|---|---|"]
allD = []; tabs = {}
for name, f in V.items():
    if not os.path.exists(f): continue
    runs, meta = load(f); tabs[name] = runs
    lines.append(f"| {name} | {meta['v_single']:.3f} | {len(runs)} | {sum(r['n_end']>=3 for r in runs)} | **{sum(is_rep(r) for r in runs)}** | {sum(is_rep(r,0.6) for r in runs)} | {sum(r['n_end']==1 for r in runs)} |")
    for r in runs:
        for d in r.get("daughters", []): allD.append((name, d))
det = ["| θ | τ | b | n объектов | массы /M1 |", "|---|---|---|---|---|"]
for r in tabs.get("R2g5", []):
    c = r["cfg"]; det.append(f"| {c['theta']} | {c.get('tau',0)} | {c.get('b',0)} | {r['n_end']} | {', '.join(f'{m:.2f}' for m in r['masses'])} |")
dl = ["| особь | масса /M1 | v в одиночку | объектов через 1500 шагов | масса через 1500 | ⟨c⟩ внутри |", "|---|---|---|---|---|---|"]
for n, d in allD: dl.append(f"| {n} | {d['mass']:.2f} | {d['v_alone']:.3f} | {d['n_after']} | {d['mass_after']:.2f} | {d['c_mean_in']:.2f} |")
open("results/part8_tables.md", "w").write("## Частота репликации (4D 64⁴, 19 конфигураций)\n\n" + "\n".join(lines) + "\n\n## R2g5 по конфигурациям\n\n" + "\n".join(det) + "\n\n## Дочерние объекты (каждый запущен в одиночку на 1500 шагов)\n\n" + "\n".join(dl) + "\n")
print("\n".join(lines)); print(len(allD), "daughters; survive:", sum(d['n_after'] == 1 and abs(d['mass_after'] - d['mass']) < 0.05 for _, d in allD))
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
runs = tabs["R2g5"]; th = sorted(set(r["cfg"]["theta"] for r in runs)); ta = sorted(set(r["cfg"].get("tau", 0) for r in runs))
for r in runs:
    c = r["cfg"]; x = c["theta"] + (6 if c.get("b", 0) else 0); y = c.get("tau", 0)
    col = "tab:green" if is_rep(r) else ("tab:red" if r["n_end"] == 1 else ("tab:orange" if r["n_end"] >= 3 else "tab:blue"))
    ax[0].scatter(x, y, s=260, color=col, edgecolor="k"); ax[0].text(x, y, str(r["n_end"]), ha="center", va="center", fontsize=9, color="w")
for lab, col in [("репликация: 3 объекта ≥0.45 M1", "tab:green"), ("≥3, но мелкий осколок", "tab:orange"), ("2 объекта", "tab:blue"), ("слияние", "tab:red")]: ax[0].scatter([], [], color=col, label=lab)
ax[0].set_xlabel("угол θ, ° (+6: прицельный параметр b=6)"); ax[0].set_ylabel("сдвиг фазы τ"); ax[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2); ax[0].set_title("R2g5, 4D 64⁴: исходы (цифра — число объектов)")
for n, mk in zip(V, "osD^"):
    D = [d for nn, d in allD if nn == n]
    if D: ax[1].scatter([d["mass"] for d in D], [d["v_alone"] for d in D], marker=mk, label=f"дочерние {n}", alpha=.8)
for n, f in V.items():
    if os.path.exists(f): ax[1].scatter([1.0], [load(f)[1]["v_single"]], marker="*", s=150, color="k")
ax[1].scatter([], [], marker="*", color="k", label="исходный одиночный (M1)"); ax[1].set_xlabel("масса дочернего / M1"); ax[1].set_ylabel("скорость в одиночку, кл/шаг"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
ax[1].set_title("дочерние — полноценные солитоны: живут и едут (меньше = быстрее)")
for f in sorted(glob.glob("results/rep/cascade[12]_*.pkl")):
    o = pickle.load(open(f, "rb"))["out"][0]
    for k, v in o["runs"].items():
        t = [x[0] for x in v["log"]]; ax[2].plot(t, [len(x[1]) for x in v["log"]], label=f"{k}: всего {len(v['log'][0][1])}→{len(v['log'][-1][1])}")
        ax[2].plot(t, [sum(m >= 0.45 for m in x[1]) for x in v["log"]], ":", color=ax[2].lines[-1].get_color(), lw=1)
ax[2].set_xlabel("шаг"); ax[2].set_ylabel("число объектов (пунктир: ≥0.45 M1)"); ax[2].set_title("популяционный тест: 6 солитонов в 96⁴"); ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)
fig.tight_layout(); fig.savefig("glider_part8_replication.png", dpi=95); print("fig")
def movie(S, out, title):
    S = [np.asarray(s, np.float32) for s in S]; vmax = float(np.percentile([s.max() for s in S], 95))
    fig, a = plt.subplots(figsize=(5, 5)); im = a.imshow(S[0].T, origin="lower", cmap="magma", vmin=0, vmax=vmax); a.set_xticks([]); a.set_yticks([]); tx = a.set_title("")
    def upd(i): im.set_data(S[i].T); tx.set_text(title(i)); return im,
    A = an.FuncAnimation(fig, upd, frames=len(S), interval=80); A.save(out, writer=an.FFMpegWriter(fps=12, bitrate=500)); plt.close(fig); print("mp4", out, os.path.getsize(out) // 1024, "KB")
r = max([r for r in runs if is_rep(r)], key=lambda r: sorted(r["masses"])[-3])
movie(r["proj"], "glider_part8_replication.mp4", lambda i: f"R2g5 4D: θ={r['cfg']['theta']}°, τ={r['cfg'].get('tau',0)} → 3 солитона\n{', '.join(f'{m:.2f}' for m in r['masses'])} M1; шаг {i*100}")
cf = sorted(glob.glob("results/rep/cascade2_*.pkl")) or sorted(glob.glob("results/rep/cascade1_*.pkl"))
o = pickle.load(open(cf[0], "rb"))["out"][0]; k, v = next(iter(o["runs"].items()))
movie(v["proj"][::2], "glider_part8_cascade.mp4", lambda i: f"{k}: 6 солитонов, 4D 96⁴ (проекция)\nшаг {i*400}, объектов {len(v['log'][min(i*4, len(v['log'])-1)][1])}")
