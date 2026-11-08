import sys; _R = __import__("pathlib").Path(__file__).resolve().parents[2]; [sys.path.insert(0, str(p)) for p in [_R, _R / "src"] + list((_R / "experiments").glob("part*"))]
"""Part 17 figures/table/mp4 from results/eco17/main17.pkl and b17m logs."""
import pickle, re, glob, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import fl_poison as Z, analyze17 as A
M = pickle.load(open("results/eco17/main17.pkl", "rb")); CONDS = ["evo", "blind", "nobrain"]; COL = dict(evo="tab:red", blind="tab:orange", nobrain="tab:blue")
cond = lambda t: re.sub(r"_s\d+$", "", t)
keys = [("n", "n (особей)", 1), ("mass", "масса (M1)", 1), ("p0", "⟨p0⟩", 2), ("GP", "G/P съедено", 1), ("fP", "доля яда в рационе", 3), ("gF", "g_F", 2), ("gP", "g_P", 2), ("gR", "g_R", 2)]
L = ["| условие | " + " | ".join(k[1] for k in keys) + " | 1−R² g_P | sd g_P по сетке | Ashman D / доля меньшего кластера | Price g_P: S / Δ / передача |", "|" + "---|" * (len(keys) + 5)]
NL = lambda r, k: r.get('nonlin', {}).get('gP', {}).get(k, np.nan); TY = lambda r, k: r.get('types', {}).get(k, np.nan)
def ms(v, d): v = [x for x in v if np.isfinite(x)]; return f"{np.mean(v):.{d}f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.{d}f}"
for c in CONDS:
    R = [v for t, v in sorted(M.items()) if cond(t) == c]
    if not R: continue
    L.append(f"| **{c}** (n={len(R)}) | " + " | ".join(ms([r[k] for r in R], d) for k, _, d in keys) + f" | {np.nanmean([NL(r, 'nonlin') for r in R]):.2f} | {np.nanmean([NL(r, 'range_sd') for r in R]):.3f} | "
             f"{np.nanmean([TY(r, 'D') for r in R]):.1f} / {np.nanmean([TY(r, 'frac') for r in R]):.2f} | {np.mean([r['price']['gP']['S'] for r in R]):+.3f} / {np.mean([r['price']['gP']['dz'] for r in R]):+.3f} / {np.mean([r['price']['gP']['T'] for r in R]):+.3f} |")
for t, r in sorted(M.items()):
    L.append(f"| {t}{' (partial)' if r['partial'] else ''} | " + " | ".join(f"{r[k]:.{d}f}" for k, _, d in keys) + f" | {NL(r, 'nonlin'):.2f} | {NL(r, 'range_sd'):.3f} | {TY(r, 'D'):.1f} / {TY(r, 'frac'):.2f} | "
             f"{r['price']['gP']['S']:+.3f} / {r['price']['gP']['dz']:+.3f} / {r['price']['gP']['T']:+.3f} |")
L.append(""); L.append("Welch t-тест (по сидам):")
for a, b in (("evo", "nobrain"), ("evo", "blind"), ("blind", "nobrain")):
    s = []
    for k, nm, _ in keys:
        x = [v[k] for t, v in M.items() if cond(t) == a and np.isfinite(v[k])]; y = [v[k] for t, v in M.items() if cond(t) == b and np.isfinite(v[k])]
        if len(x) > 1 and len(y) > 1: s.append(f"{nm} p={ttest_ind(x, y, equal_var=False).pvalue:.3g}")
    L.append(f"- {a} vs {b}: " + "; ".join(s))
L.append(""); L.append("Price для g_F и p0 (среднее по сидам, S / Δ / передача):")
for c in CONDS:
    R = [v for t, v in M.items() if cond(t) == c]
    L.append(f"- {c}: " + "; ".join(f"{k}: {np.mean([r['price'][k]['S'] for r in R]):+.3f} / {np.mean([r['price'][k]['dz'] for r in R]):+.3f} / {np.mean([r['price'][k]['T'] for r in R]):+.3f}" for k in ("gF", "p0")))
L.append(""); L.append("Центроиды 2-means (g_F, g_P, g_R) по прогонам:")
for t, r in sorted(M.items()):
    if 'types' in r: L.append(f"- {t}: " + " vs ".join("(" + ", ".join(f"{x:+.2f}" for x in cc) + ")" for cc in r["types"]["c"]) + f", доля {r['types']['frac']:.2f}")
open("results/part17_table.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
# --- figure 1: dynamics ---
fig, ax = plt.subplots(2, 3, figsize=(16, 8.5)); ax = ax.ravel()
for t, r in sorted(M.items()):
    c = cond(t); ts = r["ts"]; kw = dict(color=COL[c], lw=1, alpha=.8, label=c if t.endswith("511") else None)
    ax[0].plot(ts[:, 0], ts[:, 1], **kw); ax[1].plot(ts[:, 0], ts[:, 2], **kw)
    sm = lambda y: np.convolve(np.nan_to_num(y), np.ones(5) / 5, "same")
    ax[2].semilogy(ts[:, 0], sm(ts[:, 4]), **kw); ax[3].plot(ts[:, 0], ts[:, 5], **kw); ax[4].plot(ts[:, 0], ts[:, 6], **kw); ax[5].plot(ts[:, 0], ts[:, 3], **kw)
for a, tt in zip(ax, ["число особей n", "общая масса (M1)", "съедено хорошего / яда (сглаж.)", "g_F (своими входами)", "g_P (своими входами)", "⟨p0⟩"]): a.set_title(tt); a.set_xlabel("шаг"); a.grid(alpha=.3)
ax[0].legend(); fig.suptitle("Часть 17: ядовитая еда, 4D 56⁴, cp=2 (красный evo, оранжевый blind — яд неотличим, синий nobrain)"); fig.tight_layout(); fig.savefig("glider_part17_dynamics.png", dpi=80)
# --- figure 2: response curves + types ---
fig, ax = plt.subplots(1, 4, figsize=(18, 4.3))
for t, r in sorted(M.items()):
    if "curves" not in r: continue
    c = cond(t); cv = r["curves"]; kw = dict(color=COL[c], lw=1.2, alpha=.8, label=c if t.endswith("511") else None)
    ax[0].plot(cv[1][0], cv[1][1][:, 1], **kw); ax[1].plot(cv[1][0], cv[1][1][:, 0], **kw); ax[2].plot(cv[0][0], cv[0][1][:, 0], **kw)
    G = r["types"]["G"]; i = np.random.default_rng(0).choice(len(G), min(300, len(G)), replace=False); ax[3].scatter(G[i, 0], G[i, 1], s=4, color=COL[c], alpha=.35)
for a, tt, xl in zip(ax, ["g_P(P̃): реакция на яд", "g_F(P̃)", "g_F(F̃) (у blind — по C̃)", "особи плато: g_F vs g_P (свои входы)"], ["P̃", "P̃", "F̃−1 (blind: C̃−1)", "g_F"]):
    a.set_title(tt); a.set_xlabel(xl); a.grid(alpha=.3)
ax[3].set_ylabel("g_P"); ax[0].legend(); fig.tight_layout(); fig.savefig("glider_part17_brains.png", dpi=80)
# --- mp4: evo vs nobrain seed 511, projection onto (x0,x1) with poison pattern ---
import subprocess, os
R = A.load(); os.makedirs("/tmp/f17", exist_ok=True)
pr = {c: R[f"{c}_s511"] for c in ("evo", "blind", "nobrain") if f"{c}_s511" in R}
P = dict(Z.BDEF); n = min(len(v["log"]) for v in pr.values()); 
import pickle as _p
raw = {}
for f in sorted(glob.glob("results/eco17/b17m_*.pkl")):
    for t, r in _p.load(open(f, "rb"))["out"][0].items():
        if t in [f"{c}_s511" for c in pr]: raw[cond(t)] = r["runs"][0]["proj"]
nf = min(len(v) for v in raw.values()); vmx = max(float(np.percentile(np.asarray(v, float), 99.7)) for v in raw.values())
for k in range(nf):
    fig, ax = plt.subplots(1, len(raw), figsize=(4 * len(raw), 4.2))
    pt = Z.ptarget((56, 56), P, k * 200.0, np)
    for a, (c, pj) in zip(ax, raw.items()):
        a.imshow(pt.T, origin="lower", cmap="Greens", alpha=.6, vmin=0, vmax=P["P0"]); a.imshow(np.ma.masked_less(pj[k].astype(float).T, .03 * vmx), origin="lower", cmap="Reds", vmin=0, vmax=vmx, alpha=.9)
        a.set_title(f"{c}  t={k*200}"); a.set_xticks([]); a.set_yticks([])
    fig.suptitle("зелёный — яд (профиль), тела — проекция на (x0,x1)"); fig.tight_layout(); fig.savefig(f"/tmp/f17/{k:04d}.png", dpi=60); plt.close(fig)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "12", "-i", "/tmp/f17/%04d.png", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "30", "glider_part17_s511.mp4"], check=True)
print("frames", nf)
