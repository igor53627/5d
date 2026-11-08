import _path  # noqa: F401
import pickle, numpy as np, analyze15b as Z, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T = pickle.load(open("results/eco15/types15b.pkl", "rb")); R = Z.load()
Vt, mu0, grid = T["Vt"], T["mu0"], T["grid"]
fig, ax = plt.subplots(2, 3, figsize=(15, 9)); ax = ax.ravel()
order = ["nobrain_4D56", "frozen_4D56", "evo_4D56", "evo_rep_4D56", "nobrain_5D40", "evo_5D40"]
allW = {g: [] for g in order}; corr = []
for t, r in R.items():
    W, M, _ = Z.plateau_W(r, every=10); allW[Z.cond(t)].append(W)
for a, g in zip(ax, order):
    W = np.concatenate(allW[g]); o = Z.brain_batch(W, Z.SSTAR)[:, 0]
    a.hist2d(o[:, 0], o[:, 1], bins=40, range=[[-1, 1], [-1, 1]], cmap="magma"); a.set_title(f"{g}: n={len(W)}"); a.set_xlabel("g_F at s*"); a.set_ylabel("g_R at s*")
    W0 = np.concatenate([np.array([ob[2][Z.K0:Z.iD] for ob in R[t]["log"][0][1]]) for t in R if Z.cond(t) == g]); o0 = Z.brain_batch(W0, Z.SSTAR)[:, 0]
    a.scatter(o0[:, 0], o0[:, 1], s=8, c="cyan", marker="+", label="founders"); a.legend(fontsize=7, loc="lower left")
    F = Z.brain_batch(W, grid).reshape(len(W), -1); P = (F - mu0) @ Vt[:3].T
    corr.append((g, [[round(float(np.corrcoef(P[:, i], o[:, j])[0, 1]), 2) for j in range(3)] for i in range(3)]))
fig.suptitle("Part 15b: behaviour at typical inputs s*: g_F (chemotaxis) vs g_R (extra repulsion) of plateau individuals; + = founders"); fig.tight_layout(); fig.savefig("glider_part15b_behaviour.png", dpi=85)
W = np.concatenate([np.concatenate(v) for v in allW.values()]); o = Z.brain_batch(W, Z.SSTAR)[:, 0]; P = ((Z.brain_batch(W, grid).reshape(len(W), -1)) - mu0) @ Vt[:3].T
print("corr(PC_i, [gF,gR,gE] at s*) pooled:", [[round(float(np.corrcoef(P[:, i], o[:, j])[0, 1]), 2) for j in range(3)] for i in range(3)])
