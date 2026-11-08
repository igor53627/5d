"""Бифуркационная диаграмма континуации по asym: скорость / доля движущихся / корреляция формы vs asym."""
import sys, json, pickle, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
prefix = sys.argv[1]; n = int(sys.argv[2])
runs = [pickle.load(open(f"results/cont/{prefix}_{i}.pkl", "rb"))["out"][0] for i in range(n)]
fig, ax = plt.subplots(3, n, figsize=(4.2 * n, 10), squeeze=False)
sty = {"base": ("k", "o-"), "flip": ("tab:blue", "s--"), "reverse": ("tab:red", "^-"), "mu+3%": ("tab:green", ".-"), "mu-3%": ("tab:olive", ".-"),
       "beta2x": ("tab:purple", ".-"), "sigma-10%": ("tab:cyan", ".-"), "chi0": ("tab:brown", "x-"), "chi2x": ("tab:pink", "x-"), "iso": ("tab:orange", "d:"), "rot": ("gray", ",")}
for j, r in enumerate(runs):
    c = r["cand"]
    for b, br in enumerate(r["branches"]):
        nm = br["name"]
        if nm not in sty or nm == "rot": continue
        S = r["stats"][b]
        a = np.array([s["asym"] for s in S]); v = np.array([s.get("v_axis_med", np.nan) for s in S])
        fm = np.array([s.get("speed_med", np.nan) for s in S]); cr = np.array([s.get("corr_med", np.nan) for s in S])
        col, ls = sty[nm]
        if nm == "reverse": ls = "^"
        ax[0, j].plot(a, v, ls, color=col, label=nm, ms=4); ax[1, j].plot(a, fm, ls, color=col, ms=4); ax[2, j].plot(a, cr, ls, color=col, ms=4)
    q = r["branches"].index(next(br for br in r["branches"] if br["name"] == "quench"))
    vq = [s.get("v_axis_med", np.nan) for s in r["stats"][q][1:]]
    ax[0, j].plot([0] * len(vq), vq, "r*", ms=9, label="quench → 0")
    ax[0, j].set_title(f"#{c['seed']} {c['mode']} χ={c.get('chi', 0)}  R={c['R']:.0f}\nμ={c['mu']:.3f} σ={c['sigma']:.3f} β={c['beta']}", fontsize=9)
    for k, yl in enumerate(["медиана v вдоль оси, кл/шаг", "медиана |v| (любое направление)", "корреляция формы"]):
        ax[k, j].axvline(0, color="gray", lw=0.5); ax[k, j].axhline(0, color="gray", lw=0.5); ax[k, j].set_xlabel("asym"); ax[k, j].set_ylabel(yl)
    ax[0, j].legend(fontsize=6)
fig.suptitle("Континуация по asym (4D 40⁴, популяция из супа, стадии по 800 шагов): живёт ли подвижная ветка при asym=0?")
fig.tight_layout(); fig.savefig(f"glider_cont_bifurcation.png", dpi=110)
print("ok")
