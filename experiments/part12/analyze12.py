import sys; _R = __import__("pathlib").Path(__file__).resolve().parents[2]; [sys.path.insert(0, str(p)) for p in [_R, _R / "src"] + list((_R / "experiments").glob("part*"))]
"""Часть 12: метрики отбора и разнообразия при ёмкости."""
import numpy as np, analyze_eco as AE

def gmm_bic(x, kmax=3, iters=200):
    """1D-смесь гауссиан (EM), возвращает лучшее k по BIC и BIC-ы."""
    x = np.asarray(x, float); n = len(x); best = (1, None); bics = []
    if n < 10: return 1, []
    for k in range(1, kmax + 1):
        mu = np.quantile(x, (np.arange(k) + 0.5) / k); sd = np.full(k, x.std() / k + 1e-3); w = np.full(k, 1 / k)
        for _ in range(iters):
            pdf = w * np.exp(-(x[:, None] - mu) ** 2 / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi)) + 1e-300
            r = pdf / pdf.sum(1, keepdims=True); nk = r.sum(0) + 1e-9
            w = nk / n; mu = (r * x[:, None]).sum(0) / nk; sd = np.sqrt((r * (x[:, None] - mu) ** 2).sum(0) / nk) + 0.02
        ll = np.log((w * np.exp(-(x[:, None] - mu) ** 2 / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi))).sum(1) + 1e-300).sum()
        bics.append(-2 * ll + (3 * k - 1) * np.log(n))
    return int(np.argmin(bics)) + 1, bics

def founders(log, L):
    """Для каждого снимка: индекс основателя (объект на первом снимке) для каждого объекта."""
    par = AE.parents(log, L); anc = [list(range(len(log[0][1])))]
    for k in range(1, len(log)):
        anc.append([anc[k - 1][a] if 0 <= a < len(anc[k - 1]) else -1 for a in par[k]])
    return anc

def metrics(r, L, plateau_from=None, eco=None):
    log = r["log"]; T = log[-1][0]; pf = plateau_from or T // 2
    s = AE.summarize(r, L); ps = AE.pstats(log, 0)
    t = np.array([x[0] for x in log]); n = np.array(s["n"])
    # S по окнам 2000 шагов (лаг 5 снимков = 1000 шагов)
    Sw = []
    for a in range(0, T - 1000, 2000):
        S = AE.selection_diff([x for x in log if a <= x[0] <= a + 3000], L, lag=5); Sw.append((a, S[0], S[1]))
    Sp = AE.selection_diff([x for x in log if x[0] >= pf], L, lag=5)
    anc = founders(log, L)
    def lin(k):
        A = log[k][1]; fs = {}
        for j, o in enumerate(A):
            f = anc[k][j]; fs[f] = fs.get(f, 0) + o[0]
        m = np.array([v for f, v in fs.items() if f >= 0]); 
        if not len(m): return 0, 0.0
        q = m / m.sum(); return len(m), float(np.exp(-(q * np.log(q)).sum()))
    lins = [lin(k) for k in range(len(log))]
    pl = [k for k in range(len(log)) if t[k] >= pf]
    p0s = [np.array([o[2][0] for o in log[k][1]]) for k in pl]
    ks = [gmm_bic(p)[0] for p in p0s[::5]]
    sd_pl = float(np.mean([p.std() for p in p0s]))
    out = dict(T=T, n=n, t=t, M=np.array(s["M"]), p0=np.array([x[1] for x in ps]), sd=np.array([x[2] for x in ps]), div=s["n_div"], h=AE.herit(s["pairs"], 0)[0],
               Sw=Sw, S_plateau=Sp[0], corr_plateau=Sp[1], n_pl=float(n[pl].mean()), sd_pl=sd_pl, k_gmm=ks, k_mode=int(np.round(np.median(ks))) if ks else 1,
               lin_n=[x[0] for x in lins], lin_eff=[x[1] for x in lins], p0_last=p0s[-1] if p0s else None)
    if eco is not None and eco.get("kn", 0) > 0 and eco.get("na", 0) > 0:      # локальная адаптация
        cs = []
        for k in pl:
            A = log[k][1]; x0 = np.array([o[1][0] for o in A]); p = np.array([o[2][0] for o in A])
            popt = eco["na"] * np.sin(2 * np.pi * x0 / eco["nl"]); cs.append(np.corrcoef(p, popt)[0, 1] if len(p) > 3 else np.nan)
        out["local_adapt"] = float(np.nanmean(cs))
    if eco is not None and eco.get("sa", 0) > 0:          # отслеживание сезона
        popt_t = eco["sa"] * np.sin(2 * np.pi * t / eco["nt"]); m = t >= 2000
        lag = [np.corrcoef(out["p0"][m][:len(m.nonzero()[0]) - d], popt_t[m][d:] if d == 0 else popt_t[m][:-d])[0, 1] if True else 0 for d in (0,)]
        best = max(((d, np.corrcoef(out["p0"][m][d:], popt_t[m][:len(popt_t[m]) - d])[0, 1]) for d in range(0, 10)), key=lambda z: z[1])
        out["season_track"] = dict(corr0=float(lag[0]), best_lag_steps=int(best[0] * 200), corr_best=float(best[1]))
    return out

def merges(log, L, mmin=0.3):
    """Слияния: объект на снимке k+1, к которому ближайшими оказались ≥2 объекта (≥mmin) снимка k."""
    ev = 0; ind_steps = 0
    for (t0, A), (t1, B) in zip(log[:-1], log[1:]):
        if not A or not B: continue
        ch = AE.nearest(A, B, L); cnt = {}
        for i, j in enumerate(ch):
            if A[i][0] >= mmin: cnt[int(j)] = cnt.get(int(j), 0) + 1
        ev += sum(1 for v in cnt.values() if v >= 2); ind_steps += len(A) * (t1 - t0)
    return ev, ev / max(ind_steps, 1)

def t_plateau(t, n, frac=0.9):
    n = np.asarray(n); ref = n[int(len(n) * 0.8):].mean()
    k = np.argmax(n >= frac * ref) if (n >= frac * ref).any() else len(n) - 1
    return int(t[k]), float(ref)

def metrics13(r, L):
    m = metrics(r, L, eco=r["eco"]); log = r["log"]
    m["merges"], m["merge_rate"] = merges(log, L)
    ind = sum(len(x[1]) * (b[0] - x[0]) for x, b in zip(log[:-1], log[1:]))
    m["div_rate"] = m["div"] / max(ind, 1)
    m["t_plateau"], m["n_ref"] = t_plateau(m["t"], m["n"])
    return m
