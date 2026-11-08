import sys; from pathlib import Path; _root = next(p for p in [Path(__file__).resolve().parent] + list(Path(__file__).resolve().parents) if (p / "pyproject.toml").exists()); sys.path.insert(0, str(_root / "src"))
"""Part 18: seasons in 4D. analyze18.py [glob] → main18.pkl / pilot18.pkl and a printed summary."""
import pickle, glob, re, sys, numpy as np, fl_season as Z, analyze_eco as AE
K0, NW, iM, iD, KEAT, F0 = Z.K0, Z.NW, Z.iM, Z.KB, 0.01, 0.08
S56 = (-0.4, 0.06)
import os
PAT = sys.argv[1] if len(sys.argv) > 1 else "results/eco18/b18m_*.pkl"
LBOX = int(os.environ.get("LBOX", "56")); OUT = os.environ.get("OUT18", "")
def load(pat=PAT):
    R = {}
    for f in sorted(glob.glob(pat)):
        for t, r in pickle.load(open(f, "rb"))["out"][0].items():
            if t == "summary" or not isinstance(r, dict) or not r.get("runs"): continue
            rr = r["runs"][0]; R[t] = dict(log=rr["log"], eco=rr["eco"], tot=rr["tot"], partial=r.get("partial", False))
    return R
cond = lambda t: re.sub(r"_s\d+$", "", t)
def bb(W, S):
    """W (n,NW), S (n,NIN) per-individual → (n, NOUT)."""
    W1, b1, W2, b2 = Z.unpack(W.T, 2.0); h = np.tanh(np.einsum("hin,in->hn", W1, S.T) + b1); return np.tanh(np.einsum("ohn,hn->on", W2, h) + b2).T
def inputs(Q, sg, e):
    """Q (n, K+ND) object rows → reconstructed sensory inputs (n, NIN)."""
    n = len(Q); mem = e.get("mem", 1.0); cue = e.get("cue", 0.0)
    return np.c_[Q[:, iD] / KEAT / F0, Q[:, iD + 1] / KEAT / F0, Q[:, iD + 2], Q[:, iM] * mem, np.full(n, cue * sg), np.full(n, S56[0]), np.full(n, S56[1])]
def one(r):
    e = r["eco"]; Ts = e.get("Ts", 6000.0); log = r["log"]; ts = []
    for t, objs in log:
        sg = float(Z.season(t, Ts, np))
        if not objs: ts.append((t, sg, 0, 0, *[np.nan] * 7, 0, 0)); continue
        m = np.array([o[0] for o in objs]); q = np.array([o[2] for o in objs]); eA = (m * q[:, iD]).sum(); eB = (m * q[:, iD + 1]).sum()
        eG, eP = (eA, eB) if sg > 0 else (eB, eA)
        av = lambda x: np.average(x, weights=m)
        ts.append((t, sg, len(objs), m.sum(), av(q[:, 0]), av(q[:, iD + 3]), av(q[:, iD + 4]), av(q[:, iM]), av(q[:, iD + 2]), eP / max(eG + eP, 1e-12), sg * av(q[:, iD + 3] - q[:, iD + 4]), eG, eP))
    ts = np.array(ts, float); T = ts[-1, 0]; pl = ts[:, 0] >= T / 2
    out = dict(ts=ts, T=T, Ts=Ts, partial=r["partial"], n=np.nanmean(ts[pl, 2]), mass=np.nanmean(ts[pl, 3]), p0=np.nanmean(ts[pl, 4]),
               fP=ts[pl, 12].sum() / max(ts[pl, 11].sum() + ts[pl, 12].sum(), 1e-12), adapt=np.nanmean(ts[pl, 10]),
               corr_dg=float(np.corrcoef(np.nan_to_num(ts[pl, 5] - ts[pl, 6]), ts[pl, 1])[0, 1]) if pl.sum() > 3 else np.nan,
               corr_h=float(np.corrcoef(np.nan_to_num(ts[pl, 7]), ts[pl, 1])[0, 1]) if pl.sum() > 3 else np.nan)
    # per-season poison fraction and lag after each switch (first snapshot after the switch where σ·Δg > 0 and stays > 0 for 3 snapshots)
    sw = [k for k in range(1, len(ts)) if ts[k, 1] != ts[k - 1, 1]]; seas = []; lags = []
    bounds = [0] + sw + [len(ts)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        fp = lambda u, v: ts[u:v, 12].sum() / max(ts[u:v, 11].sum() + ts[u:v, 12].sum(), 1e-12); q = max(1, (b - a) // 4)
        seas.append((ts[a, 0], ts[a, 1], fp(a, b), np.nanmean(ts[a:b, 10]), np.nanmean(ts[a:b, 3]), np.nanmean(ts[a:a + q, 10]), np.nanmean(ts[b - q:b, 10]), fp(a, a + q), fp(b - q, b)))
    for k in sw:
        s = ts[k:, 10]; ok = [j for j in range(len(s) - 2) if np.all(s[j:j + 3] > 0)]
        prev = np.nanmean(ts[max(0, k - 3):k, 10]) > 0          # lag defined only if the population was adapted at the end of the previous season
        lags.append((ts[k + ok[0], 0] - ts[k, 0] if ok and ok[0] < Ts / 2 / 200 else Ts / 2) if prev else np.nan)   # never re-adapted → lag = whole season
    out.update(seasons=np.array(seas), lags=np.array(lags))
    # plateau individuals: memory usage and nonlinearity
    W, H, Sx, M, SG = [], [], [], [], []
    for k, (t, objs) in enumerate(log):
        if t < T / 2 or k % 3 or not objs: continue
        sg = float(Z.season(t, Ts, np)); q = np.array([o[2] for o in objs])
        W += list(q[:, K0:iM]); Sx += list(inputs(q, sg, e)); M += [o[0] for o in objs]; SG += [sg] * len(objs)
    if W:
        W, Sx, M, SG = np.array(W), np.array(Sx), np.array(M), np.array(SG)
        o = bb(W, Sx); dg = o[:, 0] - o[:, 1]
        S0 = Sx.copy(); S0[:, 3] = 0.0; o0 = bb(W, S0); Sm = Sx.copy(); Sm[:, 3] = -Sx[:, 3]; om_ = bb(W, Sm)
        out["mem_effect"] = float(np.average(np.abs(dg - (o0[:, 0] - o0[:, 1])), weights=M))          # |Δ(gA−gB)| when h → 0
        out["adapt_lesion"] = (float(np.average(SG * dg, weights=M)), float(np.average(SG * (o0[:, 0] - o0[:, 1]), weights=M)), float(np.average(SG * (om_[:, 0] - om_[:, 1]), weights=M)))
        out["h_season"] = (float(np.average(Sx[SG > 0, 3], weights=M[SG > 0])) if (SG > 0).any() else np.nan, float(np.average(Sx[SG < 0, 3], weights=M[SG < 0])) if (SG < 0).any() else np.nan)
        # nonlinearity of Δg over the observed input range (Ã, B̃, R, h): 1 − R² of a linear fit, per individual on a random design
        rng = np.random.default_rng(0); idx = rng.choice(len(W), min(300, len(W)), replace=False)
        lo, hi = np.percentile(Sx[:, :4], 5, 0), np.percentile(Sx[:, :4], 95, 0); hi = np.maximum(hi, lo + 0.05)
        G = np.repeat(np.median(Sx, 0)[None], 64, 0); G[:, :4] = lo + (hi - lo) * rng.random((64, 4)); X = np.c_[np.ones(64), G[:, :4]]
        Xi = np.c_[X, G[:, 0] * G[:, 2], G[:, 1] * G[:, 2], G[:, 0] * G[:, 3], G[:, 1] * G[:, 3]]
        nl, nli, sd = [], [], []
        for i in idx:
            Y = bb(np.repeat(W[i][None], 64, 0), G); y = Y[:, 0] - Y[:, 1]; v = y.var() + 1e-12
            nl.append((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]).var() / v); nli.append((y - Xi @ np.linalg.lstsq(Xi, y, rcond=None)[0]).var() / v); sd.append(np.sqrt(v))
        out["nonlin"] = dict(lin=float(np.average(nl, weights=M[idx])), inter=float(np.average(nli, weights=M[idx])), sd=float(np.average(sd, weights=M[idx])))
        # response of Δg to reward R in an A patch vs a B patch, at h = ±0.5 (population mean)
        cv = {}
        for patch, (a, b) in (("A", (0.8, 0.0)), ("B", (0.0, 0.8))):
            for hv in (-0.5, 0.5):
                xs = np.linspace(-2, 1, 25); SS = np.repeat(np.median(Sx, 0)[None], len(xs), 0); SS[:, 0], SS[:, 1], SS[:, 2], SS[:, 3] = a, b, xs, hv * e.get("mem", 1.0)
                cv[(patch, hv)] = (xs, np.array([np.average(bb(W[idx], np.repeat(s[None], len(idx), 0)) @ np.array([1, -1, 0, 0]), weights=M[idx]) for s in SS]))
        out["curves"] = cv
    # Price for adaptive score a = σ·(gA−gB) (realised, per object) and for p0
    pr = {}
    for nm in ("adapt", "p0"):
        lg = []
        for tt, objs in log:
            sg = float(Z.season(tt, Ts, np))
            lg.append((tt, [(o[0], o[1], [sg * (o[2][iD + 3] - o[2][iD + 4]) if nm == "adapt" else o[2][0]]) for o in objs]))
        Sv = AE.selection_diff(lg, LBOX, lag=5, k=0)[0]
        z = np.array([np.average([o[2][0] for o in objs], weights=[o[0] for o in objs]) if objs else np.nan for _, objs in lg])
        dz = float(np.nanmean(z[5:] - z[:-5])) if nm == "p0" else float("nan")    # σ flips → Δz of the adaptive score is not a heritable change; report S only
        pr[nm] = dict(S=float(Sv), dz=dz, T=dz - float(Sv))
    out["price"] = pr
    return out
if __name__ == "__main__":
    R = load(); out = {t: one(r) for t, r in R.items()}
    pickle.dump(out, open(OUT or ("results/eco18/main18.pkl" if "b18m" in PAT else "results/eco18/pilot18.pkl"), "wb"))
    for t, v in sorted(out.items()):
        print(f"{t:20s} T{v['T']:.0f}{' PART' if v['partial'] else ''} n {v['n']:.1f} mass {v['mass']:.1f} p0 {v['p0']:+.2f} fP {v['fP']:.3f} adapt {v['adapt']:+.3f} corr(Δg,σ) {v['corr_dg']:+.2f} corr(h,σ) {v['corr_h']:+.2f}",
              "lags", np.round(v["lags"]).tolist(), "memeff %.3f" % v.get("mem_effect", np.nan), "lesion", np.round(v.get("adapt_lesion", ()), 3).tolist(), "h±", np.round(v.get("h_season", ()), 2).tolist(),
              "nl", {k: round(x, 2) for k, x in v.get("nonlin", {}).items()}, "Price", {k: round(a["S"], 3) for k, a in v["price"].items()})
        print("   seasons (t, σ, fP, adapt, mass):", [tuple(np.round(s, 2)) for s in v["seasons"]])
