"""Ограниченный эволюционный поиск (элитарная ES) спонтанных глайдеров в Flow-Lenia(+χ∇φ), asym = 0.
Особь: изолированное пятно в центре чистой решётки. Фитнес = (смещение главного объекта / R) × прямолинейность × corr²,
обнуляется, если объект распался/расползся (не один доминирующий компонент) или занял > 5% решётки."""
import time, numpy as np

GENES = [("mu", 0.10, 0.60), ("sigma", 0.03, 0.25), ("chi", 0.0, 3.0), ("beta", -0.6, 0.6), ("theta", 0.5, 3.0),
         ("gam", 0.0, 0.6), ("log_tau", 0.0, 2.0), ("h1", 0.05, 1.0), ("h2", 0.05, 1.0), ("h3", 0.05, 1.0)]

def decode(z, n_rings, R, base):
    z = np.clip(z, 0, 1); d = {g: lo + (hi - lo) * z[i] for i, (g, lo, hi) in enumerate(GENES)}
    hs = [d["h1"], d["h2"], d["h3"]][:n_rings]; m = max(hs)
    c = dict(base, mode="flow", R=R, mu=d["mu"], sigma=d["sigma"], chi=d["chi"], beta=d["beta"], theta=d["theta"], gam=d["gam"],
             tau_c=10 ** d["log_tau"], peaks=[h / m for h in hs], asym=0.0)
    return c

def encode(c, n_rings):
    v = dict(c); v["log_tau"] = np.log10(c.get("tau_c", 5.0)); pk = list(c.get("peaks", [1.0])) + [1.0] * 3
    for i in range(3): v[f"h{i+1}"] = pk[i] if i < n_rings else 1.0
    v.setdefault("gam", 0.0); v.setdefault("theta", 1.5)
    return np.array([np.clip((v[g] - lo) / (hi - lo), 0, 1) for g, lo, hi in GENES])

def evaluate(cands, N=4, L=32, steps=1500, every=50, thr=0.1, seed=0, flip=False, traj=False, slice_every=0):
    import jax, jax.numpy as jnp, fl_glider as FG
    from glider_search import ball_vol
    R = cands[0]["R"]; B = len(cands)
    KA = jnp.stack([jnp.asarray(FG.ring_kernel_np(L, N, c["R"], tuple(c["peaks"]))) for c in cands])
    Ko = jnp.asarray(FG.ring_kernel_np(L, N, 3.0))
    P = jax.tree.map(lambda *a: jnp.stack(a), *[FG.params(c) for c in cands])
    sts = []
    for i, c in enumerate(cands):
        s = FG.init_state(jax.random.PRNGKey(seed), L, N, R, n_blobs=1, blob_r=c.get("blob_r", 1.0), density=c.get("density", 0.6))
        # перецентрировать пятно в центр решётки
        A = np.asarray(s["A"]); cs = np.argwhere(A > 0).mean(0) if (A > 0).any() else np.full(N, L / 2)
        sh = tuple(int(L // 2 - round(x)) for x in cs)
        s = {k: np.roll(np.asarray(v), sh, axis=tuple(range(N))) for k, v in s.items()}
        if flip: s = {k: np.flip(v, 0).copy() for k, v in s.items()}
        sts.append(s)
    st = jax.tree.map(lambda *a: jnp.stack([jnp.asarray(x) for x in a]), *sts)
    mm = 0.03 * ball_vol(N, R)
    cen = [[] for _ in range(B)]; corr = [[] for _ in range(B)]; ok = [True] * B; prevp = [None] * B; mass = [[] for _ in range(B)]
    cenC = [[] for _ in range(B)]; slices = [[] for _ in range(B)]
    for k in range(steps // every + 1):
        if k: st = FG.run_chunk(st, P, KA, Ko, every, mode="flow", use_chi=True)
        A = np.asarray(st["A"]); Cg = np.asarray(st["c"]) if traj else None
        for b in range(B):
            if not ok[b]: continue
            if not np.isfinite(A[b]).all(): ok[b] = False; continue
            comps, _ = FG.components(A[b], thr, min_mass=mm, C=Cg[b] if traj else None)
            if not comps: ok[b] = False; continue
            tot = sum(c["mass"] for c in comps); c = max(comps, key=lambda c: c["mass"])
            if c["mass"] < 0.7 * tot or c["size"] > 0.05 * L ** N:
                if k * every >= steps // 3: ok[b] = False
                continue
            p = FG.patch(A[b], c["cen"], int(R))
            if prevp[b] is not None and p.std() > 0 and prevp[b].std() > 0: corr[b].append(float(np.corrcoef(p.ravel(), prevp[b].ravel())[0, 1]))
            prevp[b] = p; cen[b].append((k * every, c["cen"])); mass[b].append(c["mass"])
            if traj:
                cenC[b].append(list(map(float, c["cenC"])))
                if slice_every and k % slice_every == 0:
                    idx = tuple(int(round(x)) % L for x in c["cen"][2:])
                    slices[b].append((A[b][(slice(None), slice(None)) + idx].astype(np.float16), Cg[b][(slice(None), slice(None)) + idx].astype(np.float16)))
    out = []
    for b in range(B):
        pts = [(t, x) for t, x in cen[b] if t >= steps // 3]
        if not ok[b] or len(pts) < 5:
            out.append(dict(fit=0.0, net=0.0, straight=0.0, corr=0.0, vel=[0.0] * N)); continue
        X = np.array([x for _, x in pts]); dts = np.array([FG.pdelta(X[i], X[i + 1], L) for i in range(len(X) - 1)])
        net = float(np.linalg.norm(dts.sum(0))); path = float(np.linalg.norm(dts, axis=1).sum()) + 1e-9
        cr = float(np.mean(corr[b][-len(dts):])) if corr[b] else 0.0
        T = pts[-1][0] - pts[0][0]
        fit = (net / R) * (net / path) * max(cr, 0) ** 2
        out.append(dict(fit=fit, net=net, straight=net / path, corr=cr, vel=(dts.sum(0) / max(T, 1)).tolist(),
                        mass_cv=float(np.std(mass[b]) / (np.mean(mass[b]) + 1e-9))))
    if traj:
        for b in range(B):
            out[b]["traj"] = [(t, list(map(float, x))) for t, x in cen[b]]; out[b]["corrs"] = corr[b]; out[b]["masses"] = mass[b]; out[b]["alive"] = ok[b]
            out[b]["cenC"] = cenC[b]; out[b]["slices"] = slices[b]
    return out

def run_es(seed_cand, n_rings=1, R=6.0, gens=20, lam=16, n_elite=4, sigma0=0.12, N=4, L=32, steps=1500, every=50, seed=0, save_cb=None):
    rng = np.random.default_rng(seed); t0 = time.time()
    z0 = encode(seed_cand, n_rings); base = {k: v for k, v in seed_cand.items() if k in ("dtA", "eps", "c_lo", "c_hi", "density", "s0")}
    pop = [z0] + [np.clip(z0 + sigma0 * rng.standard_normal(len(z0)), 0, 1) for _ in range(lam - 1)]
    hist = []; elite = []; sig = sigma0
    for g in range(gens):
        cands = [decode(z, n_rings, R, base) for z in pop]
        res = evaluate(cands, N=N, L=L, steps=steps, every=every, seed=seed * 100 + g % 3)
        for z, c, r in zip(pop, cands, res): hist.append(dict(gen=g, z=z.tolist(), cand=c, **r))
        allc = sorted(elite + [(r["fit"], z.tolist(), c, r) for z, c, r in zip(pop, cands, res)], key=lambda x: -x[0])
        elite = allc[:n_elite]
        best = elite[0]
        frac_better = np.mean([r["fit"] > elite[-1][0] for r in res])
        sig = float(np.clip(sig * (1.15 if frac_better > 0.2 else 0.9), 0.02, 0.3))
        pop = [np.array(elite[i % n_elite][1]) for i in range(2)] + \
              [np.clip(np.array(elite[rng.integers(n_elite)][1]) + sig * rng.standard_normal(len(z0)), 0, 1) for _ in range(lam - 2)]
        summ = dict(gen=g, best_fit=best[0], best=best[3], sigma=sig, t=time.time() - t0)
        if save_cb: save_cb(dict(hist=hist, elite=elite, summ=summ, partial=True))
    # проверка лучших: повтор с другим зерном и зеркальная копия (артефакты схемы адвекции)
    top = [e[2] for e in elite]
    chk = evaluate(top, N=N, L=L, steps=2 * steps, every=every, seed=seed * 100 + 77)
    chkf = evaluate(top, N=N, L=L, steps=2 * steps, every=every, seed=seed * 100 + 77, flip=True)
    out = dict(hist=hist, elite=elite, check=chk, check_flip=chkf, partial=False, t_total=time.time() - t0,
               summary=[(round(e[0], 3), round(c["fit"], 3), round(cf["fit"], 3), [round(v, 4) for v in c["vel"]], [round(v, 4) for v in cf["vel"]])
                        for e, c, cf in zip(elite, chk, chkf)])
    return out

def run_verify(jobs, save_cb=None):
    """jobs: dict(type='iso', cands, N, L, steps, seeds, flip) или dict(type='soup', kwargs для glider_search.run_candidates)."""
    import glider_search as GS
    res = []
    for j in jobs:
        if j["type"] == "iso":
            for sd in j["seeds"]:
                for fl in ([False, True] if j.get("flip") else [False]):
                    r = evaluate(j["cands"], N=j["N"], L=j["L"], steps=j["steps"], every=j.get("every", 50), seed=sd, flip=fl, traj=True, slice_every=j.get("slice_every", 0))
                    res.append(dict(type="iso", N=j["N"], L=j["L"], seed=sd, flip=fl, cands=j["cands"], out=r))
                    if save_cb: save_cb(dict(res=res, partial=True))
        else:
            r = GS.run_candidates(**j["kw"]); r["type"] = "soup"; res.append(r)
            if save_cb: save_cb(dict(res=res, partial=True))
    summ = []
    for r in res:
        if r["type"] == "iso": summ.append(("iso", r["N"], r["seed"], r["flip"], [(round(o["net"], 1), round(o["straight"], 2), round(o["corr"], 2)) for o in r["out"]]))
        else: summ.append(("soup", r["N"], r["L"], [o["n_glider"] for o in r["out"]]))
    return dict(res=res, partial=False, summary=summ)
