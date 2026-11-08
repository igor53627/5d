import sys; _R = __import__("pathlib").Path(__file__).resolve().parents[1]; [sys.path.insert(0, str(p)) for p in [_R, _R / "src"] + list((_R / "experiments").glob("part*"))]
"""Анизотропия направлений движения: гистограммы max|u_k| и угла до ближайшего направления решётки ({-1,0,1}^N) vs изотропный эталон."""
import os, sys, pickle, itertools, numpy as np, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
def latdirs(N):
    D = np.array([d for d in itertools.product([-1, 0, 1], repeat=N) if any(d)], float); return D / np.linalg.norm(D, axis=1, keepdims=True)
def feats(U, N):
    D = latdirs(N); cos = np.clip(np.abs(U @ D.T).max(1), -1, 1)
    nz = (np.abs(U) > 0.25).sum(1)
    return np.abs(U).max(1), np.degrees(np.arccos(cos)), nz
def ref(N, n=20000, seed=0):
    g = np.random.default_rng(seed).standard_normal((n, N)); return g / np.linalg.norm(g, axis=1, keepdims=True)
files = sys.argv[1:]; rows = {}; data = {}
for f in files:
    R = pickle.load(open(f, "rb"))["out"][0]
    for k, v in R["runs"].items():
        V = np.array([x["vel"] for x in v if x["ok"]]); N = V.shape[1]; sp = np.linalg.norm(V, axis=1); U = V / sp[:, None]
        a, ang, nz = feats(U, N); ar, angr, nzr = feats(ref(N), N)
        key = f"{N}D {k}"; data[key] = (a, ang, sp, N)
        rows[key] = dict(n=len(V), speed=float(sp.mean()), speed_cv=float(sp.std() / sp.mean()), r50=float(np.mean([x["r50"] for x in v if x["ok"]])),
                         r250=float(np.mean([x["r250"] for x in v if x["ok"]])), maxu_mean=float(a.mean()), maxu_iso=float(ar.mean()),
                         ang_mean=float(ang.mean()), ang_iso=float(angr.mean()), ks_maxu=float(ks_2samp(a, ar).statistic), ks_p=float(ks_2samp(a, ar).pvalue),
                         ks_ang=float(ks_2samp(ang, angr).statistic), nz_mean=float(nz.mean()), nz_iso=float(nzr.mean()),
                         corr_speed_maxu=float(np.corrcoef(sp, a)[0, 1]) if len(sp) > 2 else np.nan, ok_frac=float(np.mean([x["ok"] for x in v])))
for k, r in rows.items():
    print(f"{k:28s} n={r['n']:2d} v={r['speed']:.4f}±{r['speed_cv']*100:.1f}% r50={r['r50']:.3f} r250={r['r250']:.3f} | max|u| {r['maxu_mean']:.3f} (изотр {r['maxu_iso']:.3f}) KS={r['ks_maxu']:.2f} p={r['ks_p']:.1e} | угол до реш. {r['ang_mean']:.1f}° (изотр {r['ang_iso']:.1f}°) | осей>0.25: {r['nz_mean']:.2f} ({r['nz_iso']:.2f}) | corr(v,max|u|)={r['corr_speed_maxu']:.2f} ok={r['ok_frac']:.2f}")
json.dump(rows, open("results/iso_table.json", "w"), indent=1)
if len(sys.argv) > 1:
    keys = [k for k in os.environ.get("ISO_KEYS", "").split(";") if k in data] or list(data); fig, ax = plt.subplots(2, len(keys), figsize=(3.2 * len(keys), 6.5), squeeze=False)
    for j, k in enumerate(keys):
        a, ang, sp, N = data[k]; ar, angr, _ = feats(ref(N), N)
        ax[0, j].hist(ar, bins=20, range=(1 / np.sqrt(N), 1), density=True, alpha=0.4, color="gray", label="изотропно")
        ax[0, j].hist(a, bins=20, range=(1 / np.sqrt(N), 1), density=True, alpha=0.7, label="солитон"); ax[0, j].set_title(k + f"\nKS={rows[k]['ks_maxu']:.2f}", fontsize=8)
        ax[0, j].set_xlabel("max|u_k| (1 = ось, 1/√N = диагональ)", fontsize=7)
        ax[1, j].hist(angr, bins=20, range=(0, 30), density=True, alpha=0.4, color="gray"); ax[1, j].hist(ang, bins=20, range=(0, 30), density=True, alpha=0.7)
        ax[1, j].set_xlabel("угол до ближайшего направления решётки, °", fontsize=7)
    ax[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(os.environ.get("ISO_OUT", "glider_iso_hist_all.png"), dpi=95); print("fig")
