import sys; _R = __import__("pathlib").Path(__file__).resolve().parents[1]; [sys.path.insert(0, str(p)) for p in [_R, _R / "src"] + list((_R / "experiments").glob("part*"))]
"""Рисунок для режима без входа (ALife) с гомеостазом по условному λc."""
import pickle, sys, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
res = pickle.load(open(sys.argv[1], "rb"))
runs = res["out"]
fig = plt.figure(figsize=(18, 5.2 * len(runs)))
for i, r in enumerate(runs):
    p = r["params"]; f = r["frames"]; ts = r["ts"]
    lab = f"λ*={p['lam_target']}, κ={p['kappa']}"
    for j, (k, cm, vr, ttl) in enumerate([("phase", "twilight", (-np.pi, np.pi), "фаза"), ("c", "viridis", (2, 7), "геном c"),
                                          ("lamc", "RdBu_r", (-0.08, 0.08), "условный λc")]):
        ax = fig.add_subplot(len(runs), 4, 4 * i + j + 1)
        im = ax.imshow(f[k][-1].astype(np.float32), cmap=cm, vmin=vr[0], vmax=vr[1]); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{ttl} ({lab}), 4×4 срезов {p['L']}⁴, t={ts['t'][-1]:.0f}", fontsize=9); plt.colorbar(im, ax=ax, fraction=.046)
    ax = fig.add_subplot(len(runs), 4, 4 * i + 4)
    ax.imshow(r["chist"].T, aspect="auto", origin="lower", extent=[ts["t"][0], ts["t"][-1], 2, 7], cmap="magma")
    ax2 = ax.twinx(); ax2.plot(ts["t"], ts["lamc"], "c", lw=1, label="⟨λc⟩"); ax2.axhline(p["lam_target"], color="c", ls=":")
    ax2.plot(ts["t"], ts["chaos"] * 0.1 - 0.05, "w", lw=.8, label="доля λc>0 (×0.1−0.05)")
    ax2.set_ylim(-0.08, 0.08); ax2.legend(fontsize=7, loc="upper right")
    ax.set_xlabel("t"); ax.set_ylabel("гистограмма c"); ax.set_title("эволюция генома и ⟨λc⟩", fontsize=9)
fig.suptitle("Feigenbaum‑Lenia без входа: гомеостаз по условному λc (двойник решётки, локальная перенормировка)")
fig.tight_layout(); out = os.path.join(HERE, f"alife_condlyap_L{runs[0]['params']['L']}.png"); fig.savefig(out, dpi=90); print(out)
