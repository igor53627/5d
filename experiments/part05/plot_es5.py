import sys; _R = __import__("pathlib").Path(__file__).resolve().parents[2]; [sys.path.insert(0, str(p)) for p in [_R, _R / "src"] + list((_R / "experiments").glob("part*"))]
import sys, pickle, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
r = pickle.load(open(sys.argv[1], "rb"))["out"][0]; H = r["hist"]
G = sorted(set(h["gen"] for h in H))
best = [max(h["fit"] for h in H if h["gen"] == g) for g in G]; mean = [np.mean([h["fit"] for h in H if h["gen"] == g]) for g in G]
med = [np.median([h["fit"] for h in H if h["gen"] == g]) for g in G]
def bestrun(g, k):
    h = max((h for h in H if h["gen"] == g), key=lambda h: h["fit"]); return np.mean([x[k] for x in h["runs"]])
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].plot(G, best, "k-", label="лучший"); ax[0].plot(G, mean, "b-", label="среднее"); ax[0].plot(G, med, "c--", label="медиана")
ax[0].plot(G, np.maximum.accumulate(best), "r:", label="лучший за всё время"); ax[0].set_xlabel("поколение"); ax[0].set_ylabel("фитнес"); ax[0].legend()
ax[1].plot(G, [bestrun(g, "net") / 6 for g in G], "g-", label="смещение/R (окно 600–2000)"); ax[1].plot(G, [bestrun(g, "straight") for g in G], "m-", label="прямолин.")
ax[1].set_xlabel("поколение"); ax[1].legend()
ax[2].plot(G, [bestrun(g, "r50") for g in G], "-", label="корр. формы, лаг 50"); ax[2].plot(G, [bestrun(g, "r250") for g in G], "-", label="лаг 250")
ax[2].plot(G, [bestrun(g, "r50_p10") for g in G], ":", label="лаг 50, 10-й перцентиль"); ax[2].axhline(0.8, color="gray", lw=0.6); ax[2].set_xlabel("поколение"); ax[2].legend()
fig.suptitle("CMA-ES в 5D 20⁵ (λ=32, 2 зерна на особь, 2000 шагов, asym=0): лучший в поколении")
fig.tight_layout(); fig.savefig("glider_es5_fitness.png", dpi=100); print("ok", len(G))
