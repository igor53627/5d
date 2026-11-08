import sys; _R = __import__("pathlib").Path(__file__).resolve().parents[2]; [sys.path.insert(0, str(p)) for p in [_R, _R / "src"] + list((_R / "experiments").glob("part*"))]
"""Part 15: micro-brain metrics. python analyze15.py prefix files...  → results/eco15/<prefix>.pkl, prints table."""
import pickle, sys, re, numpy as np, analyze_eco as AE, analyze12 as A12, fl_brain as B
K0, NW = B.K0, B.NW; iD = K0 + NW          # 26: D0 intake/mass, 27 v·ĝ, 28 |v|, 29 vb·ĝ
iGF, iGR, iCI, iWD = iD + 4, iD + 5, iD + 6, iD + 7
def augment(log, ws=2.0):
    out = []
    for t, objs in log:
        o2 = []
        for o in objs:
            q = np.asarray(o[2], float); w = q[K0:iD]
            g = B.brain(w, np.zeros(B.NIN), ws); ci = q[iD + 1] / max(q[iD + 2], 1e-9)
            o2.append((o[0], o[1], list(q) + [float(g[0]), float(g[1]), float(ci), 0.0]))
        out.append((t, o2))
    return out
def mw(objs, k):
    if not objs: return np.nan
    m = np.array([o[0] for o in objs]); v = np.array([o[2][k] for o in objs]); return float((m * v).sum() / m.sum())
def one(r, L, ws=2.0):
    log = augment(r["log"], ws); r2 = dict(r, log=log)
    m = A12.metrics13(r2, L)            # n, div, merges, t_plateau, lineages (p0-based S ignored)
    t = np.array([x[0] for x in log])
    ts = {k: np.array([mw(o, i) for _, o in log]) for k, i in (("intake", iD), ("vgrad", iD + 1), ("speed", iD + 2), ("vbgrad", iD + 3), ("gF", iGF), ("gR", iGR), ("CI", iCI))}
    W = [np.array([o[2][K0:iD] for o in objs]) for _, objs in log]
    W0 = np.average(W[0], axis=0, weights=[o[0] for o in log[0][1]])
    ts["wdist"] = np.array([np.linalg.norm(np.average(w, axis=0, weights=[o[0] for o in objs]) - W0) if len(w) else np.nan for w, (_, objs) in zip(W, log)])
    ts["wsd"] = np.array([w.std(0).mean() if len(w) > 1 else np.nan for w in W])
    ts["gFsd"] = np.array([np.std([o[2][iGF] for o in objs]) if len(objs) > 1 else np.nan for _, objs in log])
    s = AE.summarize(r2, L); pairs = s["pairs"]
    hW = np.nanmean([AE.herit(pairs, k)[0] for k in range(K0, iD)])
    Sw = {}
    for nm, k in (("gF", iGF), ("CI", iCI), ("intake", iD)):
        Sw[nm] = [(a, AE.selection_diff([x for x in log if a <= x[0] <= a + 3000], L, lag=5, k=k)[0]) for a in range(0, int(t[-1]) - 1000, 2000)]
    gm = ts["gF"]; dz = []
    for a, _ in Sw["gF"]:
        ii = [i for i in range(len(t) - 5) if a <= t[i] <= a + 2000]; dz.append(float(np.nanmean([gm[i + 5] - gm[i] for i in ii])) if ii else np.nan)
    Sw["dz_gF"] = dz
    pl = t >= t[-1] // 2
    out = dict(T=int(t[-1]), t=t, n=m["n"], n_pl=m["n_pl"], div_rate_k=1000 * m["div_rate"], merge_rate_k=1000 * m["merge_rate"], t_plateau=m["t_plateau"],
               lin_eff=float(m["lin_eff"][-1]), h_w=float(hW), h_gF=AE.herit(pairs, iGF)[0], h_CI=AE.herit(pairs, iCI)[0], h_intake=AE.herit(pairs, iD)[0], Sw=Sw,
               ts=ts, **{f"{k}_pl": float(np.nanmean(v[pl])) for k, v in ts.items()}, **{f"{k}_0": float(np.nanmean(v[:3])) for k, v in ts.items()},
               S_gF_pl=float(np.nanmean([x[1] for x in Sw["gF"] if x[0] >= t[-1] // 2])), S_gF_gr=float(np.nanmean([x[1] for x in Sw["gF"] if x[0] < max(m["t_plateau"], 2000)])),
               S_CI_pl=float(np.nanmean([x[1] for x in Sw["CI"] if x[0] >= t[-1] // 2])), S_CI_all=float(np.nanmean([x[1] for x in Sw["CI"]])), dz_gF_gr=float(np.nanmean([d for (a, _), d in zip(Sw["gF"], dz) if a < max(m["t_plateau"], 2000)])))
    return out
if __name__ == "__main__":
    pre = sys.argv[1]; R = {}
    for f in sys.argv[2:]:
        for tag, r in pickle.load(open(f, "rb"))["out"][0].items():
            if tag == "summary" or not isinstance(r, dict) or not r.get("runs"): continue
            L = int(re.search(r"_\dD(\d+)", tag).group(1)); R[tag] = one(r["runs"][0], L); R[tag]["partial"] = bool(r["runs"][0].get("partial"))
    import os; os.makedirs("results/eco15", exist_ok=True); pickle.dump(R, open(f"results/eco15/{pre}.pkl", "wb"))
    keys = ["T", "n_pl", "intake_pl", "CI_0", "CI_pl", "vbgrad_pl", "speed_pl", "gF_0", "gF_pl", "gR_pl", "wdist_pl", "wsd_pl", "h_w", "h_gF", "h_CI", "S_gF_gr", "dz_gF_gr", "S_gF_pl", "S_CI_all", "div_rate_k", "merge_rate_k", "lin_eff"]
    print("tag " + " ".join(keys))
    for t in sorted(R): print(t, " ".join(f"{R[t][k]:.3g}" if isinstance(R[t][k], (float, int)) else str(R[t][k]) for k in keys))
