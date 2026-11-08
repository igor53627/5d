import sys; from pathlib import Path; _root = next(p for p in [Path(__file__).resolve().parent] + list(Path(__file__).resolve().parents) if (p / "pyproject.toml").exists()); sys.path.insert(0, str(_root / "src"))
"""Таблица и фигура исходов столкновений; mp4 одного столкновения. python plot_coll.py TAG out.png file1.pkl ... [--mp4 idx]"""
import sys, os, pickle, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
tag, png = sys.argv[1], sys.argv[2]; args = sys.argv[3:]; mp4 = None
if "--mp4" in args: i = args.index("--mp4"); mp4 = int(args[i + 1]); args = args[:i] + args[i + 2:]
runs = []; meta = None
for f in args:
    R = pickle.load(open(f, "rb"))["out"][0]; meta = R; runs += R["runs"]
import glider_coll as GC
for f in args:
    pass
for r in runs:
    if "bound_steps" not in r:   # старые pkl: переклассифицировать по всей траектории
        Lr = int(round(max(max(abs(x) for _, cc in comps for x in cc) for _, comps in r["log"] if comps))) + 1
        Lr = 64 if Lr <= 64 else 80
        lg = [(t, [(m, np.array(c)) for m, c in comps]) for t, comps in r["log"]]
        r.update(GC.classify2(lg, meta["M1"], [np.array(meta["u"]) * meta["speed"]], Lr, 50, r["t_hit"])); r["n_final"] = r["n_end"]
runs.sort(key=lambda r: (r["cfg"].get("tau", 0), -r["cfg"]["theta"], r["cfg"].get("b", 0)))
COL0 = {"слияние": "tab:red", "захват→переизлучение + обмен массой": "tab:olive", "захват→переизлучение": "tab:green", "прохождение": "tab:cyan", "отклонение": "tab:blue", "отскок + обмен массой": "tab:pink", "отскок": "tab:purple", "рождение": "tab:orange", "аннигиляция": "black"}
class _C(dict):
    def get(self, k, d=None):
        for kk, v in COL0.items():
            if k == kk: return v
        for kk, v in COL0.items():
            if k.startswith(kk): return v
        return d
COL = _C(COL0)
rows = ["| θ (цель → факт) | b | τ | исход | шагов в связанном состоянии | массы продуктов /M1 | скорости продуктов | угол между продуктами |", "|---|---|---|---|---|---|---|---|"]
for r in runs:
    c = r["cfg"]; ms = r.get("masses") or ([r["mass_ratio"]] if "mass_ratio" in r else [])
    ao = f"{r['angle_out']:.0f}°" if "angle_out" in r else ""
    rows.append(f"| {c['theta']}→{c['theta_real']:.0f}° | {c.get('b',0)} | {c.get('tau',0)} | {r['outcome']} | {r['bound_steps']} | {', '.join(f'{m:.2f}' for m in ms)} | {', '.join(f'{x:.3f}' for x in r.get('speeds_out', []))} | {ao} |")
json.dump([dict(cfg=r["cfg"], outcome=r["outcome"], bound_steps=r["bound_steps"], masses=r.get("masses"), speeds_out=r.get("speeds_out"), angle_out=r.get("angle_out")) for r in runs], open(f"results/{tag}_outcomes.json", "w"), ensure_ascii=False, indent=0)
open(f"results/{tag}_table.md", "w").write(f"скорость одиночного v={meta['speed']:.4f}, M1={meta['M1']:.0f}\n\n" + "\n".join(rows) + "\n")
from collections import Counter; cnt = Counter(r["outcome"] for r in runs); print(dict(cnt))
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for r in runs:
    c = r["cfg"]; y = c.get("b", 0) if c.get("tau", 0) == 0 else -4 - c["tau"] / 50
    ax[0].scatter(c["theta_real"], y, s=160, color=COL.get(r["outcome"], "gray"), edgecolor="k")
for k in cnt: ax[0].scatter([], [], color=COL.get(k, "gray"), label=f"{k} ({cnt[k]})", s=80)
ax[0].set_xlabel("угол между скоростями θ, °"); ax[0].set_ylabel("прицельный параметр b (клетки);  ниже 0: сдвиг фазы τ/50"); ax[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2); ax[0].grid(alpha=.3)
ax[0].set_title(f"{tag}: исходы столкновений")
for r in runs:
    t = np.arange(len(r["ncomp"])) * 50 - r["t_hit"]
    ax[1].plot(t, np.array(r["ncomp"]) + np.random.uniform(-.08, .08), color=COL.get(r["outcome"], "gray"), lw=0.8)
ax[1].axvline(0, color="k", ls=":"); ax[1].set_xlabel("шаг − t_столкн."); ax[1].set_ylabel("число объектов"); ax[1].grid(alpha=.3)
for r in runs:
    tot = [sum(m for m, _ in comps) / meta["M1"] for _, comps in r["log"]]
    ax[2].plot(np.arange(len(tot)) * 50 - r["t_hit"], tot, color=COL.get(r["outcome"], "gray"), lw=0.8)
ax[2].set_xlabel("шаг − t_столкн."); ax[2].set_ylabel("масса в объектах / M1"); ax[2].grid(alpha=.3)
fig.tight_layout(); fig.savefig(png, dpi=95); print("fig", png)
if mp4 is not None:
    import matplotlib.animation as an
    r = runs[mp4]; S = [np.asarray(s, np.float32) for s in r["proj"]]; vmax = float(np.percentile(np.max(S, axis=(1, 2)), 95))
    fig, a = plt.subplots(figsize=(5, 5)); im = a.imshow(S[0].T, origin="lower", cmap="magma", vmin=0, vmax=vmax); a.set_xticks([]); a.set_yticks([])
    tx = a.set_title("")
    def upd(i):
        im.set_data(S[i].T); tx.set_text(f"{tag}: θ={r['cfg']['theta_real']:.0f}°, b={r['cfg'].get('b',0)} — {r['outcome']}\nшаг {i*100} (столкн. ≈{r['t_hit']:.0f})"); return im,
    A = an.FuncAnimation(fig, upd, frames=len(S), interval=80); out = f"glider_{tag}_collision_{mp4}.mp4"
    A.save(out, writer=an.FFMpegWriter(fps=12, bitrate=500)); print("mp4", out, os.path.getsize(out) // 1024, "KB")
