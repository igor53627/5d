import sys; _R = __import__("pathlib").Path(__file__).resolve().parents[2]; [sys.path.insert(0, str(p)) for p in [_R, _R / "src"] + list((_R / "experiments").glob("part*"))]
"""Часть 9: фигуры экологии. python plot_eco.py OUT.png L file.pkl[:idx,idx] ... [--mp4 file.pkl:idx out.mp4]"""
import sys, os, pickle, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import analyze_eco as AE
args = sys.argv[1:]; mp4 = None
if "--mp4" in args: i = args.index("--mp4"); mp4 = (args[i + 1], args[i + 2]); args = args[:i] + args[i + 3:]
out, L = args[0], int(args[1]); sel = []
def lab(r):
    e = r["eco"]; s = f"keat={e.get('keat',0):g}, rF={e.get('rF',0):g}"
    if not r["transport"]: s += ", БЕЗ переноса p"
    if "pk" in e: s += ", отбор (keat·e^p0)" if e["pk"] > 0 else ", p нейтрален"
    if e.get("keat", 0) == 0: s = "без пищи (контроль)"
    return s
for a in args[2:]:
    f, _, ix = a.partition(":"); o = pickle.load(open(f, "rb"))["out"][0]
    for j, r in enumerate(o["runs"]):
        if not ix or str(j) in ix.split(","): sel.append((r, o))
fig, ax = plt.subplots(2, 3, figsize=(17, 9)); ax = ax.ravel()
for r, o in sel:
    s = AE.summarize(r, L); t = np.array([x[0] for x in r["log"]])
    l, = ax[0].plot(t, s["n"], label=f"{lab(r)} (делений {s['n_div']})"); col = l.get_color()
    ax[1].plot(t, s["M"], color=col); ax[2].plot(t, s["F"], color=col)
    ax[3].plot(t, s["p0_mean"], color=col)
    if s["pairs"]:
        P = np.array([(a[0], b[0]) for a, b in s["pairs"]]); ax[4].scatter(P[:, 0], P[:, 1], s=8, color=col, alpha=.6)
    k = max(1, len(r["log"]) // 6)
    first = [o_[2][0] for o_ in r["log"][0][1]]; last = [o_[2][0] for o_ in r["log"][-1][1]]
    ax[5].plot(t, [x[1] for x in AE.pstats(r["log"], 1)], color=col)
ax[0].set_ylabel("число объектов ≥0.3 M1"); ax[1].set_ylabel("общая масса / M1"); ax[2].set_ylabel("средняя пища F"); ax[3].set_ylabel("⟨p0⟩ (массо-взвеш., Δμ = 0.02·p0)")
ax[4].set_xlabel("p0 родителя"); ax[4].set_ylabel("p0 потомка"); ax[4].set_title("наследование при делениях"); ax[4].plot([-1, 1], [-1, 1], "k:", lw=.8)
ax[5].set_xlabel("шаг"); ax[5].set_ylabel("⟨p1⟩ (Δμ = 0.02·p1)"); ax[5].grid(alpha=.3)
for a in ax[:4]: a.set_xlabel("шаг"); a.grid(alpha=.3)
ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out, dpi=90); print("fig", out)
if mp4:
    import matplotlib.animation as an
    f, _, ix = mp4[0].partition(":"); r = pickle.load(open(f, "rb"))["out"][0]["runs"][int(ix or 0)]
    S = [np.asarray(s, np.float32) for s in r["proj"]]; vmax = float(np.percentile([s.max() for s in S], 90))
    pe = (r["log"][1][0] - r["log"][0][0]) if len(r["log"]) > 1 else 100
    step_of = lambda i: i * int(os.environ.get("PROJ_EVERY", 500))
    fg, a = plt.subplots(figsize=(5, 5)); im = a.imshow(S[0].T, origin="lower", cmap="magma", vmin=0, vmax=vmax); a.set_xticks([]); a.set_yticks([]); tx = a.set_title("")
    def upd(i):
        t = step_of(i); k = min(len(r["log"]) - 1, t // pe); n = len([o for o in r["log"][k][1] if o[0] >= 0.3])
        im.set_data(S[i].T); tx.set_text(f"{lab(r)}\nшаг {t}, объектов {n}, масса {r['tot'][k][0]:.1f} M1"); return im,
    A = an.FuncAnimation(fg, upd, frames=len(S), interval=100); A.save(mp4[1], writer=an.FFMpegWriter(fps=10, bitrate=500)); print("mp4", mp4[1], os.path.getsize(mp4[1]) // 1024, "KB")
