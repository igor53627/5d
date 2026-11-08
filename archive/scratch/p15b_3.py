import _path  # noqa: F401
import pickle, numpy as np, analyze15b as Z, analyze12 as A12, analyze_eco as AE, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
grid = np.array([[a, b, c] for a in (-1, -.6, 0) for b in (-.6, -.4, .3) for c in (0, .06, .2)])
feat = lambda W: Z.brain_batch(W, grid).reshape(len(W), -1)
def gmm(X, K, it=150, seed=0):
    r = np.random.default_rng(seed); n, d = X.shape; mu = X[r.choice(n, K, replace=False)]; var = np.tile(X.var(0) + 1e-4, (K, 1)); pi = np.full(K, 1 / K)
    for _ in range(it):
        lp = -0.5 * (((X[:, None] - mu[None]) ** 2) / var[None] + np.log(2 * np.pi * var[None])).sum(2) + np.log(pi)[None]
        m = lp.max(1, keepdims=True); ll = (m[:, 0] + np.log(np.exp(lp - m).sum(1))).sum(); g = np.exp(lp - m); g /= g.sum(1, keepdims=True)
        Nk = g.sum(0) + 1e-9; pi = Nk / n; mu = (g.T @ X) / Nk[:, None]; var = (g.T @ X ** 2) / Nk[:, None] - mu ** 2 + 1e-4
    return dict(mu=mu, var=var, pi=pi, ll=ll, bic=-2 * ll + (K * 2 * d + K - 1) * np.log(n), lab=g.argmax(1))
R = Z.load(); rows = []
for t, r in R.items():
    W, M, S = Z.plateau_W(r, every=10); rows += [(t, w, m) for w, m in zip(W, M)]
idx = rng.choice(len(rows), min(6000, len(rows)), replace=False); Wall = np.array([rows[i][1] for i in idx]); tags = np.array([rows[i][0] for i in idx])
F = feat(Wall); mu0 = F.mean(0); U, s, Vt = np.linalg.svd(F - mu0, full_matrices=False); ev = s ** 2 / (s ** 2).sum()
P = (F - mu0) @ Vt[:5].T; print("PCA explained", np.round(ev[:5], 3))
fits = {K: min((gmm(P, K, seed=sd) for sd in range(3)), key=lambda f: f["bic"]) for K in range(1, 9)}
Kb = min(fits, key=lambda K: fits[K]["bic"]); fb = fits[Kb]; print("BIC", {K: round(f["bic"]) for K, f in fits.items()}, "best K", Kb)
lab = fb["lab"]; conds = sorted({Z.cond(t) for t in tags})
comp = {g: np.bincount(lab[np.array([Z.cond(t) == g for t in tags])], minlength=Kb) for g in conds}
for g in conds: print(f"{g:14s}", comp[g], "types per run:", np.mean([len(set(lab[tags == t])) for t in set(tags) if Z.cond(t) == g]), "eff types/run %.2f" % np.mean([np.exp(-(lambda q: (q * np.log(q + 1e-12)).sum())(np.bincount(lab[tags == t], minlength=Kb) / (tags == t).sum())) for t in set(tags) if Z.cond(t) == g]))
# type centroids in behaviour terms
cent = fb["mu"] @ Vt[:5] + mu0; cent = cent.reshape(Kb, len(grid), 3)
js = int(np.argmin(((grid - Z.SSTAR) ** 2).sum(1)))
for k in range(Kb): print(f"type {k}: pi {fb['pi'][k]:.2f}  at s*: gF {cent[k, js, 0]:+.2f} gR {cent[k, js, 1]:+.2f} gE {cent[k, js, 2]:+.2f}; gF range over grid {cent[k,:,0].min():+.2f}..{cent[k,:,0].max():+.2f}")
# classify all individuals in full logs; lineage purity & heritability of type
def classify(W): return np.array([np.argmax([-0.5 * (((x - fb['mu'][k]) ** 2) / fb['var'][k] + np.log(fb['var'][k])).sum() + np.log(fb['pi'][k]) for k in range(Kb)]) for x in (feat(W) - mu0) @ Vt[:5].T]) if len(W) else np.array([], int)
LH = {}
for t, r in R.items():
    log = r["log"]; anc = A12.founders(log, r["L"]); T = log[-1][0]; ks = [k for k, (tt, _) in enumerate(log) if tt >= T / 2][::10]
    pur = []
    for k in ks:
        objs = log[k][1]; W = np.array([o[2][Z.K0:Z.iD] for o in objs]); ty = classify(W); fo = np.array(anc[k])
        for f in set(fo[fo >= 0]):
            m = fo == f
            if m.sum() > 1: pur.append(np.bincount(ty[m]).max() / m.sum())
    pairs = AE.summarize(dict(log=log, tot=r["tot"]), r["L"])["pairs"]
    if len(pairs) > 5:
        Pp = np.array([a[Z.K0:Z.iD] for a, b in pairs]); Pc = np.array([b[Z.K0:Z.iD] for a, b in pairs])
        tp, tc = classify(Pp), classify(Pc); fp = (feat(Pp) - mu0) @ Vt[:2].T; fc = (feat(Pc) - mu0) @ Vt[:2].T
        h1 = np.corrcoef(fp[:, 0], fc[:, 0])[0, 1]; agree = (tp == tc).mean(); base = sum((np.bincount(tp, minlength=Kb) / len(tp)) * (np.bincount(tc, minlength=Kb) / len(tc)))
    else: h1 = agree = base = np.nan
    LH[t] = dict(purity=float(np.mean(pur)) if pur else np.nan, h_pc1=float(h1), type_agree=float(agree), agree_chance=float(base), n_pairs=len(pairs))
for g in conds:
    v = [LH[t] for t in LH if Z.cond(t) == g]
    print(f"{g:14s} lineage purity {np.nanmean([x['purity'] for x in v]):.2f}  h(PC1) {np.nanmean([x['h_pc1'] for x in v]):.2f}  parent-child same type {np.nanmean([x['type_agree'] for x in v]):.2f} (chance {np.nanmean([x['agree_chance'] for x in v]):.2f})")
pickle.dump(dict(ev=ev, Vt=Vt[:5], mu0=mu0, fits={K: dict(bic=f["bic"], pi=f["pi"]) for K, f in fits.items()}, Kb=Kb, gmm=fb, comp=comp, cent=cent, LH=LH, grid=grid), open("results/eco15/types15b.pkl", "wb"))
# figure: PCA scatter by condition + type centroids + type frequency over time for one 4D evo run and 5D evo
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
cc = {"evo_4D56": "C3", "frozen_4D56": "C1", "evo_rep_4D56": "C4", "nobrain_4D56": "C0", "evo_5D40": "k", "nobrain_5D40": "c"}
for g, c in cc.items():
    m = np.array([Z.cond(t) == g for t in tags]); ax[0].scatter(P[m, 0], P[m, 1], s=3, color=c, alpha=.35, label=g)
ax[0].scatter(fb["mu"][:, 0], fb["mu"][:, 1], marker="x", s=80, color="k"); [ax[0].annotate(str(k), fb["mu"][k, :2]) for k in range(Kb)]
ax[0].set_xlabel(f"PC1 ({ev[0]:.0%})"); ax[0].set_ylabel(f"PC2 ({ev[1]:.0%})"); ax[0].legend(fontsize=7, markerscale=3); ax[0].set_title("brain types: PCA of response curves (plateau)")
for j, (t, a) in enumerate([("evo_4D56_s311", ax[1]), ("evo_5D40_s401", ax[2])]):
    log = R[t]["log"]; ts = [x[0] for x in log][::5]; fr = []
    for k in range(0, len(log), 5):
        objs = log[k][1]; ty = classify(np.array([o[2][Z.K0:Z.iD] for o in objs])); m = np.array([o[0] for o in objs])
        fr.append(np.array([m[ty == q].sum() for q in range(Kb)]) / max(m.sum(), 1e-9))
    a.stackplot(ts, np.array(fr).T, labels=[f"type {q}" for q in range(Kb)]); a.set_title(f"type mass fractions over time: {t}"); a.set_xlabel("step")
ax[2].legend(fontsize=7, loc="upper right"); fig.tight_layout(); fig.savefig("glider_part15b_types.png", dpi=85)
