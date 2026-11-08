"""Часть 9: эксперименты экологии — свип параметров пищи, длинные прогоны популяции, контроли."""
import time, numpy as np
import glider_coll as GC

def _pool(A, N, L, pool=2):
    sh = A.shape[:-N] + sum(((L // pool, pool) for _ in range(N)), ())
    return A.reshape(sh).mean(axis=tuple(range(A.ndim - N + 1, A.ndim - N + 2 * N + 1, 2)))

def _objects(Ap, Qp, M1, pool, N):
    """Объекты на пулинговой сетке: (масса/M1, центр (в клетках), средний p массо-взвешенно)."""
    import fl_glider as FG
    from scipy import ndimage
    _, lab = FG.components(Ap, 0.05, min_mass=0.0)
    ids = np.unique(lab); ids = ids[ids > 0]; out = []
    if not len(ids): return out
    ms = np.asarray(ndimage.sum(Ap, lab, ids)); keep = ms * pool ** N >= 0.15 * M1
    ids, ms = ids[keep], ms[keep]
    if not len(ids): return out
    qs = np.stack([ndimage.sum(Qp[k], lab, ids) for k in range(Qp.shape[0])], 1) / ms[:, None]
    Lp = Ap.shape[0]; cen = np.zeros((len(ids), N))
    for k in range(N):
        sh = [1] * N; sh[k] = Lp; th = (2 * np.pi * np.arange(Lp) / Lp).reshape(sh)
        cs = ndimage.sum(Ap * np.cos(th), lab, ids); sn = ndimage.sum(Ap * np.sin(th), lab, ids)
        cen[:, k] = (np.arctan2(sn, cs) % (2 * np.pi)) / (2 * np.pi) * Lp
    for j in range(len(ids)):
        out.append((float(ms[j]) * pool ** N / M1, (cen[j] * pool).tolist(), qs[j].tolist()))
    return out

def run_eco(c, ecos, N=4, L=48, seed=3, steps=10000, every=100, K=2, p0=(0.0, 0.0), pre=600, proj_every=500, transport=None,
            init="single", n_init=1, p_spread=0.0, save_cb=None, batch=6, tag="", model="eco", min_sep=None, async_log=True, opts=None, legacy_log=False):
    """ecos: список словарей эко-параметров (keat, dec, rF, F0, mut, ...); каждый — отдельный член батча.
    transport: список bool той же длины (по умолчанию True). init='single' — один солитон; 'multi' — n_init копий с p ~ N(p0, p_spread)."""
    import jax, jax.numpy as jnp, fl_glider as FG
    E = __import__({"eco2": "fl_eco2", "eco3": "fl_eco3", "brain": "fl_brain", "poison": "fl_poison", "season": "fl_season"}.get(model, "fl_eco")); opts = opts or {}; e2 = model in ("eco2", "eco3", "brain", "poison", "season"); e3 = model in ("eco3", "brain", "poison", "season")
    t0 = time.time(); out = dict(runs=[], partial=True, tag=tag)
    st, v = GC.soliton(c, N, min(L, 64) if e2 else L, seed, pre=pre); M1 = float(st["A"].sum()); out.update(M1=M1, v0=float(np.linalg.norm(v)))
    if st["A"].shape[0] != L:      # вложить солитон из 64^N в большое поле
        Ls = st["A"].shape[0]; pad = (L - Ls) // 2; sl = tuple(slice(pad, pad + Ls) for _ in range(N))
        big = {}
        for k, x in st.items():
            y = np.tile(x, (int(np.ceil(L / Ls)),) * N)[tuple(slice(0, L) for _ in range(N))].copy() if k != "A" else np.zeros((L,) * N, np.float32)
            if k == "A": y[sl] = x
            elif k == "c": y[:] = c.get("c_lo", 2.5); y[sl] = x
            big[k] = y.astype(np.float32)
        st = big
    if e2:
        U1 = E.U1_of(st["A"], L, N, **opts) if e3 else E.U1_of(st["A"], L, N); out["U1"] = U1
        ecos = [dict(e, U1=U1) for e in ecos]
    rng = np.random.default_rng(seed)
    if init == "multi":
        A = np.zeros_like(st["A"]); Q = np.zeros((K,) + st["A"].shape, np.float32); base = {k: x.copy() for k, x in st.items()}
        perms = list(GC.signed_perms(N)); pinit = []
        for i in range(n_init):
            p, sg = perms[rng.integers(len(perms))]; S = GC.transform(st, p, sg)
            for _try in range(200):     # перф: при подборе сдвига крутим только A, остальные поля — один раз
                sh = tuple(int(x) for x in rng.integers(0, L, N)); A2 = np.roll(S["A"], sh, axis=tuple(range(N)))
                if min_sep is None or not ((A2 > 0.02) & (A > 0.02)).any(): break
            S = {k: (A2 if k == "A" else np.roll(x, sh, axis=tuple(range(N)))) for k, x in S.items()}
            m = (S["A"] > 0.02) & (A < 0.02); pi = np.asarray(p0) + p_spread * rng.standard_normal(K); pinit.append(pi.tolist())
            for k in base:
                if k != "A": base[k] = np.where(m, S[k], base[k])
            A = A + np.where(m, S["A"], 0); Q += pi.reshape((K,) + (1,) * N) * np.where(m, S["A"], 0)[None]
        base["A"] = A.astype(np.float32); out["p_init"] = pinit
    transport = transport or [True] * len(ecos)
    for i0 in range(0, len(ecos), batch):
        EC = ecos[i0:i0 + batch]; TR = transport[i0:i0 + batch]
        # контроль без переноса нельзя смешать в одном vmap (static) — группируем
        for tr in sorted(set(TR)):
            idx = [j for j in range(len(EC)) if TR[j] == tr]; B = len(idx)
            cs = [dict(c, **EC[j]) for j in idx]
            if init == "multi":
                sts = []
                for j, cj in enumerate(cs):
                    s0 = E.make_state({k: base[k] for k in ("A", "x", "y", "z", "c")}, cj, K=K, seed=seed * 100 + i0 + j); s0["Q"] = Q.copy(); sts.append(s0)
            else:
                sts = [E.make_state(st, cj, K=K, p0=np.asarray(p0), seed=seed * 100 + i0 + j) for j, cj in enumerate(cs)]
            KA = jnp.stack([jnp.asarray(FG.ring_kernel_np(L, N, c["R"], tuple(c["peaks"])))] * B); Ko = jnp.asarray(FG.ring_kernel_np(L, N, 3.0))
            KX = (E.kernels(L, N, **opts) if e3 else E.kernels(L, N)) if e2 else None
            P = jax.tree.map(lambda *a: jnp.stack(a), *[E.params(cj) for cj in cs])
            s = jax.tree.map(lambda *a: jnp.stack([jnp.asarray(x) for x in a]), *sts)
            logs = [[] for _ in range(B)]; projs = [[] for _ in range(B)]; tot = [[] for _ in range(B)]
            adv = (lambda s_: E.run_chunk(s_, P, KA, Ko, KX, every, transport=tr, **opts)) if e2 else (lambda s_: E.run_chunk(s_, P, KA, Ko, every, transport=tr))
            nK = steps // every + 1; s_next = None
            for k in range(nK):
                if k: s = s_next
                Ap = np.asarray(_pool(s["A"], N, L)); Qp = np.asarray(_pool(jnp.concatenate([s["Q"], s["D"]], 1) if "D" in s else s["Q"], N, L)); Fm = np.asarray(s["F"].mean(axis=tuple(range(1, N + 1))))
                pj = (np.asarray(s["A"]).mean(axis=tuple(range(3, N + 1))) if legacy_log else np.asarray(s["A"].mean(axis=tuple(range(3, N + 1))))) if (k * every) % proj_every == 0 else None   # проекция на GPU (Часть 10, перф); legacy — как раньше, через хост
                if not np.all(np.isfinite(Ap)): break
                if async_log and k + 1 < nK: s_next = adv(s)       # асинхронно: GPU считает следующий кусок, пока CPU ищет объекты
                for b in range(B):
                    logs[b].append((k * every, _objects(Ap[b], Qp[b], M1, 2, N)))
                    tot[b].append((float(Ap[b].sum() * 2 ** N / M1), float(Fm[b])))
                    if pj is not None:
                        projs[b].append(pj[b].astype(np.float16))
                if not async_log and k + 1 < nK: s_next = adv(s)
                if save_cb and k % 20 == 0:
                    save_cb(dict(out, runs=out["runs"] + [dict(eco=EC[idx[b]], transport=tr, log=logs[b], tot=tot[b], proj=projs[b], partial=True) for b in range(B)]))
            for b in range(B):
                out["runs"].append(dict(eco=EC[idx[b]], transport=tr, log=logs[b], tot=tot[b], proj=projs[b]))
            if save_cb: save_cb(out)
    out["partial"] = False; out["t_total"] = time.time() - t0
    out["summary"] = [(r["eco"], r["transport"], len(r["log"][-1][1]), round(r["tot"][-1][0], 2)) for r in out["runs"]]
    return out

def run_eco_list(runs, save_cb=None):
    """Часть 12: несколько прогонов run_eco подряд в одном контейнере; out = {tag: результат}."""
    res = {}
    for kw in runs:
        kw = dict(kw); tag = kw.get("tag", str(len(res)))
        cb = (lambda r, tag=tag: save_cb(dict(res, **{tag: r}))) if save_cb else None
        res[tag] = run_eco(**kw, save_cb=cb)
        if save_cb: save_cb(dict(res))
    return dict(res, summary=[(t, r.get("summary")) for t, r in res.items()])
