"""Часть 8: кривые CMA-ES и таблица верификации репликации."""
import sys, glob, pickle, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for f in sorted(glob.glob("results/rep/rep[12]_*.pkl")):
    o = pickle.load(open(f, "rb"))["out"][0]; H = o["hist"]; g = sorted(set(h["gen"] for h in H))
    lab = {"rep1_0": "S1a (fv1)", "rep1_1": "S1b (fv1)", "rep1_2": "S2 (fv1)", "rep2_0": "S1c (fv2, гладкая)"}[f.split("/")[-1][:-4]]
    ax[0].plot(g, [max(h["fit"] for h in H if h["gen"] == x) for x in g], "-o", ms=3, label=lab)
    ax[1].plot(g, [np.mean([sum(r["n_end"] >= 3 for r in h["info"].get("runs", [])) for h in H if h["gen"] == x]) for x in g], "-o", ms=3, label=lab)
ax[0].set_xlabel("поколение"); ax[0].set_ylabel("лучший фитнес"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[1].set_xlabel("поколение"); ax[1].set_ylabel("ср. число столкновений с ≥3 объектами (из 6)"); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig("glider_part8_cma.png", dpi=95); print("fig")
