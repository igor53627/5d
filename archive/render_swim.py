import sys; _R = __import__("pathlib").Path(__file__).resolve().parents[1]; [sys.path.insert(0, str(p)) for p in [_R, _R / "src"] + list((_R / "experiments").glob("part*"))]
"""Фигура и анимация для изолированных «пловцов» (iso-прогоны run_verify / evaluate)."""
import sys, os, pickle, subprocess, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "."); import fl_glider as FG

def unwrap(tr, L):
    X = np.array([x for _, x in tr]); d = np.array([FG.pdelta(X[j], X[j + 1], L) for j in range(len(X) - 1)])
    return np.array([t for t, _ in tr]), np.vstack([X[:1], X[0] + np.cumsum(d, 0)])

def figure(items, out, title):
    """items: list of (label, out_dict, L, R)."""
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5)); ax = ax.ravel()
    for lab, o, L, R in items:
        if len(o.get("traj", [])) < 3: continue
        t, P = unwrap(o["traj"], L); D = P - P[0]
        l, = ax[0].plot(D[:, 0], D[:, 1], "-", label=lab); ax[0].plot(D[-1, 0], D[-1, 1], "o", color=l.get_color())
        ax[1].plot(D[:, 2], D[:, 3] if D.shape[1] > 3 else D[:, 2] * 0, "-", color=l.get_color())
        ax[2].plot(t, np.linalg.norm(D, axis=1) / R, color=l.get_color())
        ax[3].plot(t[1:len(o["corrs"]) + 1], o["corrs"], color=l.get_color(), lw=0.8)
        ax[4].plot(t[:len(o["masses"])], np.array(o["masses"]) / o["masses"][0], color=l.get_color())
        if o.get("cenC") and len(o["cenC"]) == len(o["traj"]):
            C = np.array(o["cenC"]); X = np.array([x for _, x in o["traj"]])
            off = np.array([FG.pdelta(X[j], C[j], L) for j in range(len(X))])
            v = D[-1] - D[0]; u = v / (np.linalg.norm(v) + 1e-9)
            ax[5].plot(t, off @ u, color=l.get_color())
    for a, (xl, yl) in zip(ax, [("Δx1", "Δx2"), ("Δx3", "Δx4"), ("шаг", "|смещение| / R"), ("шаг", "корр. формы (50 шагов)"), ("шаг", "масса / нач."), ("шаг", "смещение генома вдоль v, кл")]):
        a.set_xlabel(xl); a.set_ylabel(yl); a.grid(alpha=0.3)
    ax[0].legend(fontsize=7); fig.suptitle(title); fig.tight_layout(); fig.savefig(out, dpi=100); print("fig", out)

def movie(o, out, title, fps=12):
    import matplotlib.animation as an
    S = o["slices"]
    if not S: return
    fig, ax = plt.subplots(1, 2, figsize=(8, 4.2))
    vmax = max(float(np.max(s[0])) for s in S)
    im0 = ax[0].imshow(np.asarray(S[0][0], np.float32).T, origin="lower", cmap="magma", vmin=0, vmax=vmax); ax[0].set_title("масса A (срез через центр)")
    im1 = ax[1].imshow(np.asarray(S[0][1], np.float32).T, origin="lower", cmap="viridis", vmin=2.5, vmax=5.0); ax[1].set_title("геном c")
    tx = fig.suptitle(title)
    def upd(i):
        im0.set_data(np.asarray(S[i][0], np.float32).T); im1.set_data(np.asarray(S[i][1], np.float32).T); tx.set_text(f"{title}  кадр {i}")
        return im0, im1
    a = an.FuncAnimation(fig, upd, frames=len(S), interval=1000 / fps)
    a.save(out, writer=an.FFMpegWriter(fps=fps, bitrate=600)); print("mp4", out, os.path.getsize(out) // 1024, "KB")
