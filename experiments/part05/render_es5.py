import _path  # noqa: F401
"""Фигура проверки лучшей особи 5D-CMA-ES и анимация (сопутствующий срез) по ver5-pkl."""
import sys, os, pickle, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R0 = pickle.load(open(sys.argv[1], "rb"))["out"][0]; tag = sys.argv[2]
def unwrap(T, L):
    X = np.array(T); d = (np.diff(X, axis=0) + L / 2) % L - L / 2; return np.vstack([X[:1], X[0] + np.cumsum(d, 0)])
fig, ax = plt.subplots(2, 3, figsize=(15, 8.5)); ax = ax.ravel()
for key, L, N in [(f"{tag}_5D32", 32, 5), (f"{tag}_5D32_flip", 32, 5), (f"{tag}_4D64", 64, 4)]:
    for x in R0["runs"].get(key, []):
        lab = f"{key.split('_',1)[1]} {'χ=0' if x['cand'] == 1 else ''} s{x['seed']}"
        P = unwrap(x["traj"], L); D = P - P[0]; t = np.arange(len(P)) * 50
        ls = "--" if x["cand"] == 1 else "-"
        l, = ax[0].plot(D[:, 0], D[:, 1], ls, label=lab, lw=1); ax[1].plot(D[:, 2], D[:, 3], ls, color=l.get_color(), lw=1)
        ax[2].plot(t, np.linalg.norm(D, axis=1) / 6, ls, color=l.get_color())
        ax[3].plot(t, x["c50"], ls, color=l.get_color(), lw=0.8); ax[4].plot(t, x["c250"], ls, color=l.get_color(), lw=0.8)
        ax[5].plot(t, np.array(x["mass"]) / x["mass"][0], ls, color=l.get_color())
for a, (xl, yl) in zip(ax, [("Δx1", "Δx2"), ("Δx3", "Δx4"), ("шаг", "|смещение| / R"), ("шаг", "корр. формы, лаг 50"), ("шаг", "корр. формы, лаг 250"), ("шаг", "масса / нач.")]):
    a.set_xlabel(xl); a.set_ylabel(yl); a.grid(alpha=0.3)
ax[3].set_ylim(0.5, 1); ax[4].set_ylim(0.5, 1); ax[0].legend(fontsize=6)
fig.suptitle(f"Лучшая особь 5D-CMA-ES ({tag}), asym=0: 5D 32⁵ (3 зерна, χ=0 пунктир, зеркало) и 4D 64⁴"); fig.tight_layout()
fig.savefig(f"glider_es5_{tag}_verify.png", dpi=100); print("fig")
if len(sys.argv) > 3:
    import matplotlib.animation as an
    x = R0["runs"][f"{tag}_5D32"][0]; S = x["slices"]; P = unwrap(x["traj"], 32)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.3))
    im = ax[0].imshow(np.asarray(S[0], np.float32).T, origin="lower", cmap="magma", vmin=0, vmax=float(max(np.max(s) for s in S)))
    ax[0].set_title("срез массы в сопутствующей системе (5D 32⁵)")
    ax[1].set_xlim(P[:, 0].min() - 3, P[:, 0].max() + 3); ax[1].set_ylim(P[:, 1].min() - 3, P[:, 1].max() + 3); ln, = ax[1].plot([], [], "c-"); pt, = ax[1].plot([], [], "ro")
    ax[1].set_title("траектория центра (x1, x2), развёрнутая"); tx = fig.suptitle("")
    def upd(i):
        j = min(2 * i, len(P) - 1); im.set_data(np.asarray(S[i], np.float32).T); ln.set_data(P[:j + 1, 0], P[:j + 1, 1]); pt.set_data([P[j, 0]], [P[j, 1]])
        tx.set_text(f"самодвижущийся солитон {tag}, 5D, asym=0, шаг {j*50}"); return im, ln, pt
    a = an.FuncAnimation(fig, upd, frames=len(S), interval=80); out = f"glider_es5_{tag}_5D.mp4"
    a.save(out, writer=an.FFMpegWriter(fps=12, bitrate=600)); print("mp4", os.path.getsize(out) // 1024, "KB")
