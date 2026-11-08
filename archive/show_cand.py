import sys; _R = __import__("pathlib").Path(__file__).resolve().parents[1]; [sys.path.insert(0, str(p)) for p in [_R, _R / "src"] + list((_R / "experiments").glob("part*"))]
import pickle, glob, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
prefix, ids, out = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], sys.argv[3]
rows = {}
for f in sorted(glob.glob(f"results/glider/{prefix}_*.pkl")):
    for job in pickle.load(open(f, "rb"))["out"]:
        for o in job["out"]: rows[o["cand"]["seed"]] = o
fig, axs = plt.subplots(len(ids), 6, figsize=(14, 2.4 * len(ids)), squeeze=False)
for r, i in enumerate(ids):
    o = rows[i]; P = o["proj"]; T = len(P)
    for j, t in enumerate(np.linspace(0, T - 1, 6).astype(int)):
        axs[r, j].imshow(P[t].astype(np.float32), vmin=0, vmax=1 if o["cand"]["mode"] == "lenia" else None, cmap="inferno")
        axs[r, j].set_xticks([]); axs[r, j].set_yticks([]); axs[r, j].set_title(f"#{i} t={t*25}", fontsize=7)
fig.tight_layout(); fig.savefig(out, dpi=80)
