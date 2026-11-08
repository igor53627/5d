import sys; _R = __import__("pathlib").Path(__file__).resolve().parents[2]; [sys.path.insert(0, str(p)) for p in [_R, _R / "src"] + list((_R / "experiments").glob("part*"))]
"""Часть 13: видео особей (проекция центров на оси x1,x2), цвет = p0, размер ∝ масса. python plot_mp4_12.py out.mp4 L file:tag:title ..."""
import sys, pickle, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.animation as an
out = sys.argv[1]; items = [a.split(":", 2) for a in sys.argv[2:]]; import re; LS = [int(re.match(r"\w+?_\dD(\d+)", t).group(1)) for _, t, _ in items]
logs = [pickle.load(open(f, "rb"))["out"][0][t]["runs"][0]["log"] for f, t, _ in items]
nf = min(len(l) for l in logs); fr = list(range(0, nf, 2))
fg, ax = plt.subplots(1, len(logs), figsize=(4.8 * len(logs), 5.2)); ax = np.atleast_1d(ax)
sc = [a.scatter([], [], c=[], cmap="coolwarm", vmin=-1, vmax=1, s=[]) for a in ax]
for a, (_, _, ti) in zip(ax, items): a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
fg.colorbar(sc[-1], ax=ax[-1], fraction=.046, label="p0"); tt = fg.suptitle("")
def upd(k):
    for i, l in enumerate(logs):
        O = l[fr[k]][1]
        if O:
            xy = np.array([o[1][:2] for o in O]) / LS[i]; p = np.array([o[2][0] for o in O]); m = np.array([o[0] for o in O])
            sc[i].set_offsets(xy); sc[i].set_array(p); sc[i].set_sizes(12 * m)
        ax[i].set_title(f"{items[i][2]}\nособей {len(O)}, sd p0 {np.std([o[2][0] for o in O]) if O else 0:.2f}", fontsize=9)
    tt.set_text(f"шаг {logs[0][fr[k]][0]}"); return sc
A = an.FuncAnimation(fg, upd, frames=len(fr), interval=120); A.save(out, writer=an.FFMpegWriter(fps=8, bitrate=700)); print(out, os.path.getsize(out) // 1024, "KB", len(fr))
