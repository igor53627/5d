import _path  # noqa: F401
"""Part 17: poison food in 4D. Loads results/eco17/b17*.pkl (fl_poison logs) → table, main17.pkl, figures."""
import pickle, glob, re, sys, numpy as np, fl_poison as Z, analyze_eco as AE
K0, NW, iD, KEAT, F0 = Z.K0, Z.NW, Z.K0 + Z.NW, 0.01, 0.08
S23 = (-0.4, 0.06)                         # typical own-density / neighbour inputs (Part 15b)
PAT = sys.argv[1] if len(sys.argv) > 1 else "results/eco17/b17m_*.pkl"
def load(pat=PAT):
    R = {}
    for f in sorted(glob.glob(pat)):
        for t, r in pickle.load(open(f, "rb"))["out"][0].items():
            if t == "summary" or not isinstance(r, dict) or not r.get("runs"): continue
            rr = r["runs"][0]; R[t] = dict(log=rr["log"], eco=rr["eco"], tot=rr["tot"], partial=r.get("partial", False))
    return R
def cond(t): return re.sub(r"_s\d+$", "", t)
def bb(W, s):
    """W (n,NW), s (n_s,NIN) → (n, n_s, NOUT)."""
    s = np.atleast_2d(s); W1, b1, W2, b2 = Z.unpack(W.T, 2.0)
    h = np.tanh(np.einsum("hin,si->snh", W1, s) + b1.T[None])
    return np.transpose(np.tanh(np.einsum("ohn,snh->sno", W2, h) + b2.T[None]), (1, 0, 2))
def inputs(q, blind):
    """per-object sensory inputs reconstructed from intake: F_loc ≈ D0/keat, P_loc ≈ D1/keat (unblurred approximation)."""
    Fl = q[iD] / KEAT / F0; Pl = q[iD + 1] / KEAT / F0
    return np.array([Fl + Pl - 1, 0.0, *S23]) if blind else np.array([Fl - 1, Pl, *S23])
def own_gains(objs, blind):
    """realised gains of each object at its own inputs."""
    if not objs: return np.zeros((0, 3))
    W = np.array([o[2][K0:iD] for o in objs]); S = np.array([inputs(np.array(o[2]), blind) for o in objs])
    W1, b1, W2, b2 = Z.unpack(W.T, 2.0); h = np.tanh(np.einsum("hin,in->hn", W1, S.T) + b1); return np.tanh(np.einsum("ohn,hn->on", W2, h) + b2).T
def one(r):
    bl = r["eco"].get("blind", 0) > 0; log = r["log"]; ts = []
    for t, objs in log:
        if not objs: ts.append((t, 0, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0)); continue
        m = np.array([o[0] for o in objs]); q = np.array([o[2] for o in objs]); g = own_gains(objs, bl)
        eG = (m * q[:, iD]).sum(); eP = (m * q[:, iD + 1]).sum()
        ts.append((t, len(objs), m.sum(), np.average(q[:, 0], weights=m), eG / max(eP, 1e-12), np.average(g[:, 0], weights=m), np.average(g[:, 1], weights=m),
                   np.average(g[:, 2], weights=m), np.average(q[:, iD + 2], weights=m), np.average(q[:, iD + 3], weights=m), eP / max(eG + eP, 1e-12), eG, eP))
    ts = np.array(ts, float); T = ts[-1, 0]; pl = ts[:, 0] >= T / 2
    # plateau individuals
    W, M, S = [], [], []
    for k, (t, objs) in enumerate(log):
        if t < T / 2 or k % 5: continue
        for o in objs: W.append(o[2][K0:iD]); M.append(o[0]); S.append(inputs(np.array(o[2]), bl))
    W, M, S = np.array(W), np.array(M), np.array(S)
    out = dict(ts=ts, T=T, partial=r["partial"], n=np.nanmean(ts[pl, 1]), mass=np.nanmean(ts[pl, 2]), p0=np.nanmean(ts[pl, 3]), GP=ts[pl, 11].sum() / max(ts[pl, 12].sum(), 1e-12), fP=ts[pl, 12].sum() / max(ts[pl, 11].sum() + ts[pl, 12].sum(), 1e-12), eGm=ts[pl, 11].sum() / max(ts[pl, 2].sum(), 1e-12),
               gF=np.nanmean(ts[pl, 5]), gP=np.nanmean(ts[pl, 6]), gR=np.nanmean(ts[pl, 7]), vF=np.nanmean(ts[pl, 8]), vP=np.nanmean(ts[pl, 9]), blind=bl)
    if len(W):
        # response curves (population mean over plateau individuals, mass-weighted), other inputs at median observed values
        smed = np.median(S, 0); out["smed"] = smed; cv = {}
        for j, xs in ((0, np.linspace(-1, 0.5, 31)), (1, np.linspace(0, 1.0, 31))):
            SS = np.repeat(smed[None], len(xs), 0); SS[:, j] = xs; cv[j] = (xs, np.average(bb(W, SS), axis=0, weights=M))
        out["curves"] = cv
        # nonlinearity: per-individual outputs on a grid over observed (s0,s1) range; 1−R² of linear and of linear+interaction fit
        lo, hi = np.percentile(S[:, :2], 5, 0), np.percentile(S[:, :2], 95, 0); hi = np.maximum(hi, lo + 0.05)
        g0, g1 = np.meshgrid(np.linspace(lo[0], hi[0], 7), np.linspace(lo[1], hi[1], 7) if not bl else [0.0]); G = np.repeat(smed[None], g0.size, 0)
        G[:, 0] = g0.ravel(); G[:, 1] = g1.ravel(); idx = np.random.default_rng(0).choice(len(W), min(400, len(W)), replace=False)
        o = bb(W[idx], G)                                                           # (n, ng, 3)
        X1 = np.c_[np.ones(len(G)), G[:, 0], G[:, 1]]; X2 = np.c_[X1, G[:, 0] * G[:, 1], G[:, 0] ** 2, G[:, 1] ** 2]
        nl = {}
        for k, nm in enumerate(("gF", "gP")):
            Y = o[:, :, k]; var = Y.var(1)
            r1 = [(Y[i] - X1 @ np.linalg.lstsq(X1, Y[i], rcond=None)[0]).var() for i in range(len(Y))]
            nl[nm] = dict(range_sd=float(np.average(np.sqrt(var), weights=M[idx])), nonlin=float(np.average(r1, weights=M[idx]) / max(np.average(var, weights=M[idx]), 1e-12)))
        out["nonlin"] = nl
        # discrete types: realised gains of plateau individuals; 2-means on (gF,gP,gR) and Ashman's D along the split axis
        Gn = np.array([bb(W[i:i + 1], S[i])[0, 0] for i in range(len(W))])
        c = Gn[[np.argmin(Gn[:, 1]), np.argmax(Gn[:, 1])]]
        for _ in range(30):
            lab = np.argmin(((Gn[:, None] - c[None]) ** 2).sum(-1), 1)
            c = np.array([Gn[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(2)])
        ax = c[1] - c[0]; ax /= np.linalg.norm(ax) + 1e-12; pr = Gn @ ax; a, b = pr[lab == 0], pr[lab == 1]
        D = abs(a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-12) if len(a) > 2 and len(b) > 2 else np.nan
        out["types"] = dict(D=float(D), frac=float(min(lab.mean(), 1 - lab.mean())), c=c.tolist(), G=Gn, M=M)
    # Price (whole run) for realised g_F, g_P and p0
    pr = {}
    for k, nm in ((0, "gF"), (1, "gP"), (None, "p0")):
        lg = [(tt, [(o[0], o[1], [float(v)]) for o, v in zip(objs, (own_gains(objs, bl)[:, k] if k is not None else [x[2][0] for x in objs]))]) for tt, objs in log]
        Sx = AE.selection_diff(lg, 56, lag=5, k=0)[0]
        z = np.array([np.average([o[2][0] for o in objs], weights=[o[0] for o in objs]) if objs else np.nan for _, objs in lg]); dz = float(np.nanmean(z[5:] - z[:-5]))
        pr[nm] = dict(S=float(Sx), dz=dz, T=dz - float(Sx))
    out["price"] = pr
    return out
if __name__ == "__main__":
    R = load(); out = {t: one(r) for t, r in R.items()}
    pickle.dump(out, open("results/eco17/main17.pkl" if "b17m" in PAT else "results/eco17/pilot17.pkl", "wb"))
    for t, v in sorted(out.items()):
        nl = v.get("nonlin", {}); ty = v.get("types", {})
        print(f"{t:22s} T{v['T']:.0f}{' PART' if v['partial'] else ''} n {v['n']:.1f} mass {v['mass']:.1f} p0 {v['p0']:+.2f} G/P {v['GP']:.2f} fP {v['fP']:.3f} gF {v['gF']:+.2f} gP {v['gP']:+.2f} gR {v['gR']:+.2f} vF {v['vF']:+.4f} vP {v['vP']:+.4f}",
              " nl", {k: (round(a['range_sd'], 3), round(a['nonlin'], 2)) for k, a in nl.items()}, " types D %.2f frac %.2f" % (ty.get('D', np.nan), ty.get('frac', np.nan)),
              " Price", {k: tuple(round(x, 3) for x in a.values()) for k, a in v["price"].items()})
