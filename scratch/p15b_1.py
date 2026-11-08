import pickle, numpy as np, analyze15b as Z, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = Z.load(); C = {}
for t, r in R.items():
    W, M, _ = Z.plateau_W(r); C.setdefault(Z.cond(t), []).append(Z.curves(W, M))
    W0 = np.array([o[2][Z.K0:Z.iD] for o in r["log"][0][1]]); M0 = np.array([o[0] for o in r["log"][0][1]]); C.setdefault("start_" + Z.cond(t)[-4:], []).append(Z.curves(W0, M0))
pickle.dump(C, open("results/eco15/curves15b.pkl", "wb"))
sty = {"evo_4D56": ("C3", "-", "4D evo"), "frozen_4D56": ("C1", "-", "4D frozen"), "evo_rep_4D56": ("C4", "-", "4D evo krep20"), "nobrain_4D56": ("C0", "-", "4D nobrain (drift)"),
       "start_4D56": ("0.6", ":", "4D founders (t=0)"), "evo_5D40": ("C3", "--", "5D evo (1 seed)"), "nobrain_5D40": ("C0", "--", "5D nobrain (1 seed)")}
outs = ["g_F (chemotaxis)", "g_R (extra repulsion)", "g_E (eat gain, unused)"]
fig, ax = plt.subplots(3, 3, figsize=(15, 12))
lines = []
for j, (nm, xs) in enumerate(Z.RANGES):
    for g, (c, ls, lb) in sty.items():
        if g not in C: continue
        Y = np.array([cv[nm][1] for cv in C[g]])     # seeds × n_s × NOUT
        m = Y.mean(0); se = Y.std(0, ddof=1) / np.sqrt(len(Y)) if len(Y) > 1 else 0 * m
        for k in range(3):
            ax[k, j].plot(xs, m[:, k], ls, color=c, label=lb if (j == 0 and k == 0) else None); ax[k, j].fill_between(xs, m[:, k] - se[:, k], m[:, k] + se[:, k], color=c, alpha=.15)
        lines.append(f"{g:14s} {nm:22s} " + " ".join(f"{outs[k][:3]}: {m[0,k]:+.2f}→{m[-1,k]:+.2f}" for k in range(3)))
    for k in range(3): ax[k, j].axvline(Z.SSTAR[j], color="k", lw=.5, ls=":"); ax[k, j].axhline(0, color="k", lw=.5); ax[k, j].set_xlabel(nm); ax[k, j].set_ylabel(outs[k])
ax[0, 0].legend(fontsize=7); fig.suptitle("Part 15b: response curves of evolved brains (others inputs at typical s* = (−.6, −.4, .06)); mean over seeds ± SE")
fig.tight_layout(); fig.savefig("glider_part15b_curves.png", dpi=85)
open("results/eco15/curves15b.txt", "w").write("\n".join(lines)); print("\n".join(lines))
# at s*
for g in sty:
    if g in C:
        v = np.array([cv[Z.RANGES[0][0]][1][np.argmin(abs(Z.RANGES[0][1] - Z.SSTAR[0]))] for cv in C[g]]); print(g, "at s*: gF %.2f±%.2f gR %.2f±%.2f gE %.2f" % (v[:, 0].mean(), v[:, 0].std(), v[:, 1].mean(), v[:, 1].std(), v[:, 2].mean()))
