import _path  # noqa: F401
"""Визуализация результатов Feigenbaum-Lenia: анимация, диагностика, финальный кадр."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from feigenbaum_lenia import participation_dim, correlation_dimension, rossler_bifurcation, single_lyapunov

BG = "#0b0b12"

def _style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values(): s.set_color("#555")
    ax.tick_params(colors="#bbb", labelsize=8)
    ax.xaxis.label.set_color("#ddd"); ax.yaxis.label.set_color("#ddd"); ax.title.set_color("#eee")

def render_animation(d, p, out):
    ph, c, lam = d["phase"], d["c"], d["lam"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.6), facecolor=BG)
    cm_ph = plt.get_cmap("twilight").copy(); cm_ph.set_bad(BG)
    cm_c = plt.get_cmap("magma").copy(); cm_c.set_bad(BG)
    cm_l = plt.get_cmap("coolwarm").copy(); cm_l.set_bad(BG)
    lmax = float(np.nanpercentile(np.abs(lam[len(lam)//4:]), 98)) or 0.1
    ims = [axs[0].imshow(ph[0], cmap=cm_ph, vmin=-np.pi, vmax=np.pi, interpolation="nearest"),
           axs[1].imshow(c[0], cmap=cm_c, vmin=p["cmin"], vmax=p["cmax"], interpolation="nearest"),
           axs[2].imshow(lam[0], cmap=cm_l, vmin=-lmax, vmax=lmax, interpolation="nearest")]
    titles = ["фаза осциллятора φ = atan2(y,x)", "геном c (бифуркационный параметр)", "локальный показатель Ляпунова λ"]
    for ax, t, im in zip(axs, titles, ims):
        _style(ax); ax.set_title(t, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.ax.tick_params(colors="#bbb", labelsize=7)
    sub = ("сетка 4×4 срезов (x₁,x₂): столбцы — x₃, строки — x₄" if p["N"] >= 4 else
           "срезы по x₃" if p["N"] == 3 else "2D поле")
    sup = fig.suptitle("", color="#eee", fontsize=12)
    fig.text(0.5, 0.02, f"Feigenbaum-Lenia, N={p['N']}, решётка {p['L']}^{p['N']} — {sub}", ha="center", color="#999", fontsize=9)
    w = imageio.get_writer(os.path.join(out, "animation.mp4"), fps=20, quality=8, macro_block_size=1)
    for i in range(len(ph)):
        ims[0].set_data(ph[i]); ims[1].set_data(c[i]); ims[2].set_data(lam[i])
        sup.set_text(f"t = {d['t'][i]:.0f}   R_глоб={d['Rglob'][i]:.2f}   доля хаоса={d['frac_chaos'][i]:.2f}   ⟨c⟩={d['c_mean'][i]:.2f}")
        fig.canvas.draw()
        w.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    w.close(); plt.close(fig)
    # лёгкий GIF (каждый 3-й кадр)
    os.system(f"ffmpeg -loglevel error -y -i {out}/animation.mp4 -vf 'fps=7,scale=900:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse' {out}/animation.gif")

def render_diagnostics(d, p, out):
    t = d["t"]; S = d["snaps"].astype(np.float64)
    burn = len(S) // 4
    fig, axs = plt.subplots(3, 2, figsize=(14, 12), facecolor=BG)
    axs = axs.ravel(); [_style(a) for a in axs]
    a = axs[0]; a.plot(t, d["Rglob"], label="R глобальный |⟨e^{iφ}⟩|", c="#4fc3f7"); a.plot(t, d["Rloc"], label="R локальный ⟨|K∗e^{iφ}|⟩", c="#ffb74d")
    a.set_title("Синхронизация (параметр порядка Курамото)"); a.legend(fontsize=8, facecolor=BG, labelcolor="#ddd"); a.set_xlabel("t")
    a = axs[1]; a.plot(t, d["lam_mean"], c="#ef5350", label="⟨λ_loc⟩"); a.axhline(p["lam_target"], ls="--", c="#888", label="λ*")
    a2 = a.twinx(); a2.plot(t, d["frac_chaos"], c="#81c784", label="доля клеток λ>0.01"); a2.tick_params(colors="#bbb", labelsize=8)
    a.set_title("Локальный хаос"); a.legend(loc="upper left", fontsize=8, facecolor=BG, labelcolor="#ddd"); a2.legend(loc="upper right", fontsize=8, facecolor=BG, labelcolor="#ddd"); a.set_xlabel("t")
    a = axs[2]
    a.imshow(np.log10(d["chist"].T + 1e-5), aspect="auto", origin="lower", cmap="inferno",
             extent=[t[0], t[-1], d["cbins"][0], d["cbins"][-1]], vmin=-4)
    a.axhline(4.3, c="#4fc3f7", ls=":", lw=1); a.text(t[0], 4.35, " порог хаоса Рёсслера ≈4.3", color="#4fc3f7", fontsize=8)
    a.set_title("Распределение генома c во времени (log)"); a.set_xlabel("t"); a.set_ylabel("c")
    # PCA размерность в скользящем окне
    a = axs[3]; win = max(20, len(S) // 8); dims, tc = [], []
    for i in range(0, len(S) - win, max(1, win // 4)):
        dims.append(participation_dim(S[i:i+win])[0]); tc.append(t[i + win // 2])
    a.plot(tc, dims, c="#ce93d8", marker="o", ms=3)
    a.set_title(f"PCA-размерность глобального поля (participation ratio, окно {win} кадров)"); a.set_xlabel("t")
    a = axs[4]; D, ev = participation_dim(S[burn:]); ev = ev / ev.sum()
    a.semilogy(np.arange(1, min(60, len(ev)) + 1), ev[:60], "o-", c="#ffd54f", ms=3)
    a.set_title(f"Спектр PCA (после переходного процесса): D_PR = {D:.1f}"); a.set_xlabel("номер компоненты"); a.set_ylabel("доля дисперсии")
    a = axs[5]; nu, rs, C = correlation_dimension(S[burn:])
    a.loglog(rs, C, "o-", c="#80cbc4", ms=3)
    a.set_title(f"Корреляционная размерность (Грассбергер–Прокачча, 12 ГК): ν ≈ {nu:.2f}"); a.set_xlabel("r"); a.set_ylabel("C(r)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "diagnostics.png"), dpi=110, facecolor=BG); plt.close(fig)
    return {"D_PR": D, "corr_dim": nu}

def render_still(d, p, out):
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.25, 1])
    ph = d["phase"][-1]; c = d["c"][-1]; lam = d["lam"][-1]
    import matplotlib.colors as mc
    # цвет — геном c (где клетка на бифуркационной диаграмме), яркость — фазовый сдвиг относительно среднего поля
    mean_ph = np.angle(np.nanmean(np.exp(1j * ph)))
    rel = np.cos(np.nan_to_num(ph) - mean_ph)
    rel = (rel - np.nanmin(rel)) / (np.nanmax(rel) - np.nanmin(rel) + 1e-9)
    cn = (np.nan_to_num(c, nan=p["cmin"]) - p["cmin"]) / (p["cmax"] - p["cmin"])
    rgb = plt.get_cmap("turbo")(cn)[..., :3] * (0.35 + 0.65 * rel[..., None])
    rgb[np.isnan(ph)] = mc.to_rgb(BG)
    ax = fig.add_subplot(gs[:, 0:2]); _style(ax); ax.imshow(rgb, interpolation="bicubic"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Feigenbaum-Lenia {p['N']}D, t={d['t'][-1]:.0f}: цвет — геном c (turbo: синий=порядок, красный=хаос), "
                 f"яркость — фаза относительно среднего поля", fontsize=11)
    # бифуркационная диаграмма + распределение c
    ax = fig.add_subplot(gs[0, 2]); _style(ax)
    cs = np.linspace(p["cmin"], p["cmax"], 300)
    bc, bx = rossler_bifurcation(p["a"], p["b"], cs, T=500, burn=300)
    ax.scatter(bc, bx, s=0.2, c="#9e9e9e", alpha=0.5, lw=0)
    ax2 = ax.twinx(); h = d["chist"][len(d["chist"]) // 2:].mean(0); cb = d["cbins"]
    ax2.fill_between(0.5 * (cb[1:] + cb[:-1]), h, color="#ff7043", alpha=0.6, step="mid"); ax2.set_yticks([])
    ax.set_title("Бифуркационная диаграмма Рёсслера (серое)\nи где «живут» клетки (оранжевое)", fontsize=10); ax.set_xlabel("c"); ax.set_ylabel("max x")
    ax = fig.add_subplot(gs[1, 2]); _style(ax)
    lyap = single_lyapunov(p["a"], p["b"], cs[::3])
    ax.plot(cs[::3], lyap, c="#4fc3f7"); ax.axhline(0, c="#666", lw=0.8)
    ax.set_title("λ_max одиночного Рёсслера (c)", fontsize=10); ax.set_xlabel("c")
    fig.tight_layout(); fig.savefig(os.path.join(out, "still.png"), dpi=120, facecolor=BG); plt.close(fig)

def render_all(out):
    p = json.load(open(os.path.join(out, "params.json")))
    d = dict(np.load(os.path.join(out, "run.npz")))
    m = render_diagnostics(d, p, out); print("диагностика:", m)
    json.dump(m, open(os.path.join(out, "metrics.json"), "w"), indent=2)
    render_still(d, p, out)
    render_animation(d, p, out)
    print("готово:", out)

if __name__ == "__main__":
    render_all(sys.argv[1] if len(sys.argv) > 1 else "out")
