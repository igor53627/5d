import sys; _R = __import__("pathlib").Path(__file__).resolve().parents[2]; [sys.path.insert(0, str(p)) for p in [_R, _R / "src"] + list((_R / "experiments").glob("part*"))]
"""Длинный CMA-ES в 5D: жёсткий самодвижущийся солитон Flow-Lenia+χ∇φ при asym=0.
Метрики считаются на GPU (без scipy): масса, круговой центр и локализация по осям, корреляция формы в сопутствующей
системе (центрированный, усреднённый 2×-пулингом снимок) с лагами 50 и 250 шагов.
Фитнес = (смещение/R) · прямолинейность · f_rigid · f_mass · f_loc, агрегируется по 2 зёрнам: 0.5·(mean+min).
Чекпойнт после каждого поколения (CMA-состояние + история) → можно продолжить."""
import time, os, pickle, numpy as np

GENES = [("mu", 0.10, 0.50), ("sigma", 0.05, 0.25), ("chi", 0.0, 2.0), ("beta", -0.8, 0.8), ("theta", 0.3, 2.5),
         ("gam", 0.0, 0.5), ("log_tau", 0.0, 2.0), ("h1", 0.05, 1.0), ("h2", 0.05, 1.0), ("h3", 0.05, 1.0),
         ("dtA", 0.1, 0.35), ("eps", 0.03, 0.3), ("c_hi", 3.5, 6.0)]

def decode(z, R=6.0):
    z = np.clip(z, 0, 1); d = {g: lo + (hi - lo) * float(z[i]) for i, (g, lo, hi) in enumerate(GENES)}
    hs = [d["h1"], d["h2"], d["h3"]]; m = max(hs)
    return dict(mode="flow", R=R, mu=d["mu"], sigma=d["sigma"], chi=d["chi"], beta=d["beta"], theta=d["theta"], gam=d["gam"],
                tau_c=10 ** d["log_tau"], peaks=[h / m for h in hs], dtA=d["dtA"], eps=d["eps"], c_hi=d["c_hi"], asym=0.0, density=0.6)

def encode(c):
    v = dict(DEFS); v.update(c); v["log_tau"] = np.log10(v.get("tau_c", 5.0)); pk = list(c.get("peaks", [1.0]))
    pk = (pk + [pk[-1]] * 3)[:3]
    for i in range(3): v[f"h{i+1}"] = pk[i]
    return np.array([np.clip((v[g] - lo) / (hi - lo), 0, 1) for g, lo, hi in GENES])
DEFS = dict(gam=0.0, theta=1.5, dtA=0.2, eps=0.1, c_hi=5.0, tau_c=5.0)

class CMA:
    """Минимальный (μ/μ_w, λ)-CMA-ES (Hansen), максимизация, ограничения [0,1] через отражение."""
    def __init__(self, x0, sigma, lam, seed=0):
        n = len(x0); self.n = n; self.lam = lam; self.mu = lam // 2; self.rng = np.random.default_rng(seed)
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1)); self.w = w / w.sum(); self.mueff = 1 / (self.w ** 2).sum()
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n); self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff); self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs; self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.m = np.array(x0, float); self.sigma = sigma; self.C = np.eye(n); self.pc = np.zeros(n); self.ps = np.zeros(n); self.gen = 0
    def ask(self):
        D2, B = np.linalg.eigh(self.C); D = np.sqrt(np.maximum(D2, 1e-20)); self._B, self._D = B, D
        Z = self.rng.standard_normal((self.lam, self.n)); Y = Z @ np.diag(D) @ B.T; X = self.m + self.sigma * Y
        X = np.abs(X); X = 1 - np.abs(1 - X)            # отражение в [0,1]
        self._Y = (X - self.m) / self.sigma
        return X
    def tell(self, X, f):
        idx = np.argsort(-np.asarray(f))[: self.mu]; Y = self._Y[idx]; yw = self.w @ Y
        self.m = self.m + self.sigma * yw
        Cinvsqrt = self._B @ np.diag(1 / self._D) @ self._B.T
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * Cinvsqrt @ yw
        hs = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) / self.chiN < 1.4 + 2 / (self.n + 1)
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - hs) * self.cc * (2 - self.cc) * self.C) \
                 + self.cmu * (Y.T * self.w) @ Y
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1)); self.sigma = float(np.clip(self.sigma, 0.01, 0.4))
        self.m = np.clip(self.m, 0, 1); self.gen += 1

def _metric_fns(N, L, pool=2):
    import jax, jax.numpy as jnp
    th = 2 * np.pi * np.arange(L) / L
    def one(A):
        M = A.sum(); out_c, out_r = [], []
        for k in range(N):
            sh = [1] * N; sh[k] = L
            cs = (A * jnp.cos(th).reshape(sh)).sum(); sn = (A * jnp.sin(th).reshape(sh)).sum()
            out_c.append((jnp.arctan2(sn, cs) % (2 * np.pi)) * L / (2 * np.pi)); out_r.append(jnp.sqrt(cs * cs + sn * sn) / (M + 1e-9))
        cen = jnp.stack(out_c)
        shift = tuple((L // 2 - jnp.round(cen[k]).astype(jnp.int32)) for k in range(N))
        Ac = jnp.roll(A, shift, axis=tuple(range(N)))
        P = Ac.reshape(sum(((L // pool, pool) for _ in range(N)), ())).mean(axis=tuple(range(1, 2 * N, 2))).ravel()
        return M, cen, jnp.stack(out_r), P
    vone = jax.jit(jax.vmap(one))
    def corr(P, Q):
        P = P - P.mean(-1, keepdims=True); Q = Q - Q.mean(-1, keepdims=True)
        return (P * Q).sum(-1) / (jnp.sqrt((P * P).sum(-1) * (Q * Q).sum(-1)) + 1e-12)
    return vone, jax.jit(corr)

def evaluate(cands, seeds, N=5, L=20, steps=2000, every=50, t0=600, flip=False, keep=False, slice_every=0, init=None):
    """Каждый кандидат × каждое зерно — отдельная решётка (vmap). Возвращает метрики и фитнес на кандидата."""
    import jax, jax.numpy as jnp, fl_glider as FG
    R = cands[0]["R"]; jobs = [(i, s) for i in range(len(cands)) for s in seeds]; B = len(jobs)
    KA = jnp.stack([jnp.asarray(FG.ring_kernel_np(L, N, cands[i]["R"], tuple(cands[i]["peaks"]))) for i, _ in jobs])
    Ko = jnp.asarray(FG.ring_kernel_np(L, N, 3.0))
    P = jax.tree.map(lambda *a: jnp.stack(a), *[FG.params(cands[i]) for i, _ in jobs])
    sts = []
    for i, s in jobs:
        if init is not None: st = init(i, s)
        else:
            st = FG.init_state(jax.random.PRNGKey(s), L, N, R, n_blobs=1, density=cands[i].get("density", 0.6))
            A = np.asarray(st["A"]); cs = np.argwhere(A > 0).mean(0) if (A > 0).any() else np.full(N, L / 2)
            sh = tuple(int(L // 2 - round(x)) for x in cs); st = {k: np.roll(np.asarray(v), sh, axis=tuple(range(N))) for k, v in st.items()}
            st["c"] = np.asarray(cands[i].get("c_lo", 2.5) + (cands[i].get("c_hi", 5.0) - cands[i].get("c_lo", 2.5)) * np.clip(st["A"], 0, 1), np.float32)
        if flip: st = {k: np.flip(v, 0).copy() for k, v in st.items()}
        sts.append(st)
    st = jax.tree.map(lambda *a: jnp.stack([jnp.asarray(x) for x in a]), *sts)
    vone, corr = _metric_fns(N, L)
    Ms, Cs, Rs, c50, c250, slices = [], [], [], [], [], []
    buf = []
    for k in range(steps // every + 1):
        if k: st = FG.run_chunk(st, P, KA, Ko, every, mode=cands[0].get("mode", "flow"), use_chi=True)
        M, cen, rr, Pp = vone(st["A"])
        buf.append(Pp); buf = buf[-6:]
        c50.append(np.asarray(corr(buf[-1], buf[-2])) if len(buf) >= 2 else np.full(B, np.nan))
        c250.append(np.asarray(corr(buf[-1], buf[0])) if len(buf) >= 6 else np.full(B, np.nan))
        Ms.append(np.asarray(M)); Cs.append(np.asarray(cen)); Rs.append(np.asarray(rr))
        if slice_every and k % slice_every == 0:
            A = np.asarray(st["A"][:, (slice(None),) * 2 + (L // 2,) * (N - 2)]) if False else None
            Ac = np.asarray(jnp.stack([jnp.roll(st["A"][b], tuple(int(L // 2 - round(float(x))) for x in np.asarray(cen[b])), axis=tuple(range(N)))
                                       [(slice(None), slice(None)) + (L // 2,) * (N - 2)] for b in range(B)]))
            slices.append(Ac.astype(np.float16))
    Ms = np.array(Ms); Cs = np.array(Cs); Rs = np.array(Rs); c50 = np.array(c50); c250 = np.array(c250)
    T = np.arange(len(Ms)) * every; w = T >= t0
    res = []
    for b in range(B):
        M = Ms[:, b]; ok = np.isfinite(M).all() and M[w].min() > 0.3 * M[0]
        d = (np.diff(Cs[:, b], axis=0) + L / 2) % L - L / 2; dw = d[w[1:]]
        net = float(np.linalg.norm(dw.sum(0))); path = float(np.linalg.norm(dw, axis=1).sum()) + 1e-9
        r50 = float(np.nanmean(c50[w, b])); r250 = float(np.nanmean(c250[w, b])); rmin = float(np.nanpercentile(c50[w, b], 10))
        mw = M[w]; mcv = float(mw.std() / (mw.mean() + 1e-9)); drift = float(abs(mw[-1] / mw[0] - 1))
        loc0 = float(Rs[w, b].min(-1)[0]); locw = float(np.percentile(Rs[w, b].min(-1), 10))
        rig = 0.5 * (r50 + r250)
        f_r = float(np.clip((rig - 0.4) / 0.5, 0, 1) ** 2) * (1.0 + 0.5 * float(rig > 0.8))
        f_m = float(np.exp(-(mcv / 0.05) ** 2) * np.exp(-(drift / 0.1) ** 2))
        f_l = float(np.clip(locw / max(loc0, 1e-6), 0, 1) ** 2) * float(loc0 > 0.25)
        fit = (net / R) * (net / path) * f_r * f_m * f_l if ok else 0.0
        res.append(dict(cand=jobs[b][0], seed=jobs[b][1], fit=fit, net=net, straight=net / path, r50=r50, r250=r250, r50_p10=rmin,
                        mcv=mcv, drift=drift, loc0=loc0, locw=locw, ok=bool(ok), vel=(dw.sum(0) / max(T[-1] - t0, 1)).tolist(),
                        traj=Cs[:, b].tolist() if keep else None, mass=M.tolist() if keep else None,
                        c50=c50[:, b].tolist() if keep else None, c250=c250[:, b].tolist() if keep else None,
                        slices=[s[b] for s in slices] if slice_every else None))
    per = []
    for i in range(len(cands)):
        f = [r["fit"] for r in res if r["cand"] == i]
        per.append(0.5 * (np.mean(f) + np.min(f)))
    return per, res

def run_cma(seed_cands, gens=60, lam=32, sigma0=0.12, N=5, L=20, steps=2000, n_seeds=2, max_wall=5400, seed=0,
            ckpt="/vol/es5_ckpt.pkl", save_cb=None, final_seeds=(901, 902, 903)):
    t_start = time.time()
    if os.path.exists(ckpt):
        S = pickle.load(open(ckpt, "rb")); es, hist, best = S["es"], S["hist"], S["best"]
    else:
        es = CMA(encode(seed_cands[0]), sigma0, lam, seed=seed); hist = []; best = []
    while es.gen < gens and time.time() - t_start < max_wall:
        tg = time.time(); X = es.ask()
        if es.gen == 0:
            for j, c in enumerate(seed_cands[: lam]): X[j] = encode(c)   # инъекция стартовых особей (C, B …)
            es._Y = (X - es.m) / es.sigma
        cands = [decode(x) for x in X]
        sd = [1000 * es.gen + s for s in range(n_seeds)]
        f, res = evaluate(cands, sd, N=N, L=L, steps=steps)
        es.tell(X, f)
        for j in range(lam):
            rr = [r for r in res if r["cand"] == j]
            hist.append(dict(gen=es.gen - 1, fit=float(f[j]), z=X[j].tolist(), runs=[{k: v for k, v in r.items() if k not in ("traj", "mass", "c50", "c250", "slices")} for r in rr]))
        best = sorted(best + [(float(f[j]), X[j].tolist()) for j in range(lam)], key=lambda t: -t[0])[:8]
        gi = int(np.argmax(f)); rb = [r for r in res if r["cand"] == gi]
        print(f"gen {es.gen-1}: best {f[gi]:.3f} mean {np.mean(f):.3f} sigma {es.sigma:.3f} | net {np.mean([r['net'] for r in rb]):.1f} r50 {np.mean([r['r50'] for r in rb]):.2f} r250 {np.mean([r['r250'] for r in rb]):.2f} mcv {np.mean([r['mcv'] for r in rb]):.3f} | {time.time()-tg:.0f}s", flush=True)
        pickle.dump(dict(es=es, hist=hist, best=best), open(ckpt + ".tmp", "wb")); os.replace(ckpt + ".tmp", ckpt)
        if save_cb: save_cb(dict(hist=hist, best=best, gen=es.gen, sigma=es.sigma, mean=es.m.tolist(), partial=True))
    done = es.gen >= gens
    # финальная проверка топ-5 (различных) на новых зёрнах
    top = []; 
    for fb, z in best:
        if all(np.abs(np.array(z) - np.array(t)).max() > 1e-6 for t in top): top.append(z)
    top = top[:5] + [es.m.tolist()]
    fin, fres = evaluate([decode(np.array(z)) for z in top], list(final_seeds), N=N, L=L, steps=steps)
    out = dict(hist=hist, best=best, gen=es.gen, sigma=es.sigma, mean=es.m.tolist(), final_top=top, final_fit=list(map(float, fin)),
               final_runs=[{k: v for k, v in r.items() if k not in ("traj", "mass", "c50", "c250", "slices")} for r in fres], partial=not done,
               summary=[round(float(x), 3) for x in fin], t_total=time.time() - t_start)
    return out

def _single_state(c, N, L, seed):
    import jax, fl_glider as FG
    st = FG.init_state(jax.random.PRNGKey(seed), L, N, c["R"], n_blobs=1, density=c.get("density", 0.6))
    A = np.asarray(st["A"]); cs = np.argwhere(A > 0).mean(0)
    sh = tuple(int(L // 2 - round(x)) for x in cs); st = {k: np.roll(np.asarray(v), sh, axis=tuple(range(N))) for k, v in st.items()}
    st["c"] = np.asarray(c.get("c_lo", 2.5) + (c.get("c_hi", 5.0) - c.get("c_lo", 2.5)) * np.clip(st["A"], 0, 1), np.float32)
    return st

def collision(c, N=5, L=32, pre=1500, steps=4000, seed=7, gap=None, every=50):
    """Два пловца лоб в лоб: разгоняем одного (pre шагов), берём его и зеркальную копию вдоль главной оси скорости,
    ставим на расстоянии gap навстречу друг другу. Исход по компонентам связности (scipy) каждые 250 шагов."""
    import jax, jax.numpy as jnp, fl_glider as FG
    from scipy import ndimage
    R = c["R"]; KA = jnp.asarray(FG.ring_kernel_np(L, N, R, tuple(c["peaks"])))[None]; Ko = jnp.asarray(FG.ring_kernel_np(L, N, 3.0))
    P = jax.tree.map(lambda a: a[None], FG.params(c))
    st = jax.tree.map(lambda a: jnp.asarray(a)[None], _single_state(c, N, L, seed))
    vone, _ = _metric_fns(N, L)
    cens = []
    for k in range(pre // every):
        st = FG.run_chunk(st, P, KA, Ko, every, mode=c.get("mode", "flow"), use_chi=True); cens.append(np.asarray(vone(st["A"])[1][0]))
    d = (np.diff(np.array(cens[len(cens) // 2:]), axis=0) + L / 2) % L - L / 2; v = d.sum(0); ax = int(np.argmax(np.abs(v))); sgn = np.sign(v[ax])
    S = {k: np.asarray(x[0]) for k, x in st.items()}
    cen = cens[-1]; sh = tuple(int(L // 2 - round(x)) for x in cen)
    S = {k: np.roll(x, sh, axis=tuple(range(N))) for k, x in S.items()}          # пловец в центре
    gap = gap or int(L // 2)
    a = {k: np.roll(x, -int(sgn) * gap // 2, axis=ax) for k, x in S.items()}      # позади центра, едет вперёд (sgn)
    b = {k: np.roll(np.flip(x, ax), int(sgn) * gap // 2, axis=ax) for k, x in S.items()}   # зеркальный, едет назад
    mA = a["A"] > 0.02; mB = b["A"] > 0.02
    two = {k: np.where(mB & ~mA, b[k], a[k]) for k in S}; two["A"] = a["A"] + b["A"]
    st2 = jax.tree.map(lambda x: jnp.asarray(x)[None], two)
    M0 = float(two["A"].sum()); log = []; slices = []
    for k in range(steps // every + 1):
        if k: st2 = FG.run_chunk(st2, P, KA, Ko, every, mode=c.get("mode", "flow"), use_chi=True)
        if k % 5 == 0:
            A = np.asarray(st2["A"][0]); comps, _ = FG.components(A, 0.1, min_mass=0.03 * M0 / 2)
            log.append(dict(t=k * every, n=len(comps), masses=sorted([round(cc["mass"], 1) for cc in comps], reverse=True)[:4],
                            cens=[list(map(float, cc["cen"])) for cc in sorted(comps, key=lambda q: -q["mass"])[:4]]))
            idx = (slice(None),) * N; sl = [L // 2] * N; sl[ax] = slice(None); other = [i for i in range(N) if i != ax][0]; sl[other] = slice(None)
            slices.append(A[tuple(sl)].astype(np.float16))
    return dict(axis=ax, sign=float(sgn), gap=gap, M0=M0, log=log, slices=slices, v_pre=(v / (len(d) * every)).tolist())

def run_verify5(cands, save_cb=None, seeds=(11, 12, 13), steps=4000):
    """Проверка лучших: 5D 32⁵ (3 зерна) + зеркало + χ=0; 4D 64⁴ (2 зерна); столкновение двух пловцов в 5D 32⁵."""
    out = dict(runs={}, partial=True)
    def save():
        if save_cb: save_cb(out)
    for name, c in cands.items():
        c0 = dict(c, chi=0.0)
        f, r = evaluate([c, c0], list(seeds), N=5, L=32, steps=steps, keep=True, slice_every=2); out["runs"][f"{name}_5D32"] = r; save()
        f, r = evaluate([c], [seeds[0]], N=5, L=32, steps=steps, keep=True, flip=True); out["runs"][f"{name}_5D32_flip"] = r; save()
        f, r = evaluate([c, c0], list(seeds[:2]), N=4, L=64, steps=steps, keep=True); out["runs"][f"{name}_4D64"] = r; save()
        out["runs"][f"{name}_collision"] = collision(c, N=5, L=32); save()
    out["partial"] = False
    out["summary"] = {k: ([(round(x["net"], 1), round(x["r50"], 2), round(x["r250"], 2)) for x in v] if isinstance(v, list) else [(l["t"], l["n"]) for l in v["log"]][-3:]) for k, v in out["runs"].items()}
    return out

def run_ctrl5(c, save_cb=None, seeds=(21, 22), steps=3000):
    """Контроли механизма (5D 24⁵) + столкновения двух пловцов (4D 64⁴ gap 32, 5D 40⁵ gap 20)."""
    var = {"base": {}, "chi0": dict(chi=0.0), "beta0": dict(beta=0.0), "gam0": dict(gam=0.0), "beta0_gam0": dict(beta=0.0, gam=0.0),
           "pure_flow(b0,g0,chi0)": dict(beta=0.0, gam=0.0, chi=0.0), "eps0": dict(eps=0.0), "dtA×0.7": dict(dtA=c["dtA"] * 0.7),
           "tau_c×10": dict(tau_c=c["tau_c"] * 10), "c_hi=c_lo(no genome contrast)": dict(c_hi=2.5001)}
    names = list(var); cands = [dict(c, **var[n]) for n in names]
    out = dict(names=names, partial=True)
    f, r = evaluate(cands, list(seeds), N=5, L=24, steps=steps, keep=False); out["ctrl"] = r
    if save_cb: save_cb(out)
    out["coll4"] = collision(c, N=4, L=64, pre=1500, steps=4000, gap=32)
    if save_cb: save_cb(out)
    out["coll5"] = collision(c, N=5, L=40, pre=1500, steps=3000, gap=20)
    out["partial"] = False
    out["summary"] = [(n, round(float(np.mean([x["net"] for x in r if x["cand"] == i])), 1), round(float(np.mean([x["r50"] for x in r if x["cand"] == i])), 2)) for i, n in enumerate(names)]
    return out

def run_eval5(cands, seeds=(31, 32, 33), N=5, L=32, steps=3000, save_cb=None):
    """Короткая финальная проверка нескольких особей + зеркало + χ=0 для каждой."""
    names = list(cands); cs = []
    for n in names: cs += [cands[n], dict(cands[n], chi=0.0)]
    f, r = evaluate(cs, list(seeds), N=N, L=L, steps=steps)
    f2, r2 = evaluate([cands[n] for n in names], [seeds[0]], N=N, L=L, steps=steps, flip=True)
    return dict(names=names, runs=r, flip=r2, fit=list(map(float, f)), partial=False,
                summary=[(n, round(float(f[2 * i]), 2), round(float(f[2 * i + 1]), 2)) for i, n in enumerate(names)])

def run_iso(cands, modes, n_seeds=48, N=5, L=20, steps=2000, seed0=500, save_cb=None):
    """Распределение направлений движения по многим зёрнам для каждой (особь, схема переноса)."""
    out = dict(runs={}, partial=True)
    for name, c in cands.items():
        for m in modes:
            f, r = evaluate([dict(c, mode=m)], list(range(seed0, seed0 + n_seeds)), N=N, L=L, steps=steps)
            out["runs"][f"{name}|{m}"] = [{k: v for k, v in x.items() if k not in ("traj", "mass", "c50", "c250", "slices")} for x in r]
            if save_cb: save_cb(out)
    out["partial"] = False
    out["summary"] = {k: (round(float(np.mean([np.linalg.norm(x["vel"]) for x in v])), 4), round(float(np.mean([x["r50"] for x in v])), 3)) for k, v in out["runs"].items()}
    return out
