import _path  # noqa: F401
"""Часть 10: видео проекций (среднее по осям x3..xN) для нескольких прогонов рядом. python plot_mp4_10.py out.mp4 proj_every f1.pkl:title f2.pkl:title ..."""
import sys, os, pickle, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.animation as an
out, pe = sys.argv[1], int(sys.argv[2]); items = [a.split(":", 1) for a in sys.argv[3:]]
R = [pickle.load(open(f, "rb"))["out"][0]["runs"][0] for f, _ in items]
S = [[np.asarray(s, np.float32) for s in r["proj"]] for r in R]; nf = min(len(s) for s in S)
vm = [float(np.percentile([x.max() for x in s], 90)) for s in S]
fg, ax = plt.subplots(1, len(R), figsize=(4.6 * len(R), 4.8)); ax = np.atleast_1d(ax)
ims = [a.imshow(S[i][0], cmap="magma", vmin=0, vmax=vm[i]) for i, a in enumerate(ax)]
for a, (_, t) in zip(ax, items): a.set_title(t, fontsize=9); a.axis("off")
tt = fg.suptitle("")
def nobj(r, step):
    j = min(range(len(r["log"])), key=lambda k: abs(r["log"][k][0] - step)); return len(r["log"][j][1]), np.average([o[2][0] for o in r["log"][j][1]], weights=[o[0] for o in r["log"][j][1]]) if r["log"][j][1] else np.nan
def upd(k):
    for i in range(len(R)):
        ims[i].set_data(S[i][k]); n, p = nobj(R[i], k * pe); ax[i].set_xlabel(f"особей {n}, ⟨p0⟩={p:+.2f}")
        ax[i].set_title(items[i][1] + f"\nособей {n}, ⟨p0⟩={p:+.2f}", fontsize=9)
    tt.set_text(f"шаг {k * pe}"); return ims
A = an.FuncAnimation(fg, upd, frames=nf, interval=250); A.save(out, writer=an.FFMpegWriter(fps=4, bitrate=900)); print(out, os.path.getsize(out) // 1024, "KB", nf, "frames")
