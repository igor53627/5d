import sys; _R = __import__("pathlib").Path(__file__).resolve().parents[2]; [sys.path.insert(0, str(p)) for p in [_R, _R / "src"] + list((_R / "experiments").glob("part*"))]
"""Численная континуация по asym: вырастить подвижную «палочку» в супе, вырезать одну на чистую решётку,
затем медленно уводить asym → 0 (с релаксацией на каждом шаге) в нескольких ветках сразу (vmap):
вариации μ/σ/β/χ, зеркальная копия (flip), резкий сброс (quench), реверс asym.
Возвращает временные ряды центра, массы, размера, корреляции формы для каждой ветки."""
import time, numpy as np

def _fresh_osc(key, L, N):
    import jax, jax.numpy as jnp
    k = jax.random.split(key, 3); shape = (L,) * N
    th = jax.random.uniform(k[0], shape, maxval=2 * np.pi); rad = jax.random.uniform(k[1], shape, minval=2, maxval=6)
    return np.array(rad * jnp.cos(th)), np.array(rad * jnp.sin(th)), np.array(jax.random.uniform(k[2], shape, maxval=0.5))

def grow_and_pick(cand, N=4, L_soup=32, soup_steps=2000, track_steps=600, every=25, thr=0.1, nseed=4, seed=0):
    """Суп → трек → выбрать лучший подвижный локализованный объект. Возвращает dict полей, центрированных в L_soup//2, и маску."""
    import jax, jax.numpy as jnp, fl_glider as FG
    from glider_search import ball_vol
    from scipy import ndimage
    R = cand["R"]; mode = cand.get("mode", "lenia"); use_chi = cand.get("chi", 0) != 0
    cands = [dict(cand, seed=cand.get("seed", 0) + 7919 * s) for s in range(nseed)]
    KA = jnp.stack([jnp.asarray(FG.ring_kernel_np(L_soup, N, R, tuple(c.get("peaks", (1.0,))), shift=c.get("asym", 0.0))) for c in cands])
    Ko = jnp.asarray(FG.ring_kernel_np(L_soup, N, 3.0))
    P = jax.tree.map(lambda *a: jnp.stack(a), *[FG.params(c) for c in cands])
    st = jax.tree.map(lambda *a: jnp.stack(a), *[FG.init_state(jax.random.PRNGKey(seed * 1000 + c["seed"]), L_soup, N, R, n_blobs=c.get("n_blobs", 6),
                                                                  density=c.get("density", 0.5)) for c in cands])
    st = FG.run_chunk(st, P, KA, Ko, soup_steps, mode=mode, use_chi=use_chi)
    frames = [[] for _ in cands]; last = None
    mm = 0.03 * ball_vol(N, R)
    for ch in range(track_steps // every + 1):
        if ch: st = FG.run_chunk(st, P, KA, Ko, every, mode=mode, use_chi=use_chi)
        A = np.asarray(st["A"])
        for b in range(len(cands)):
            comps, lab = FG.components(A[b], thr, min_mass=mm)
            for c in comps: c["patch"] = FG.patch(A[b], c["cen"], int(R))
            frames[b].append(comps[:64])
    A = np.asarray(st["A"]); best = None
    for b in range(len(cands)):
        tr = FG.track(frames[b], L_soup, R, every)
        for t in tr:
            if t["t1"] != len(frames[b]) - 1 or t["size_max"] > 0.05 * L_soup ** N or t["life"] < 0.5 * track_steps: continue
            sc = t["net"] * max(t["corr"], 0) * (1 - min(t["mass_cv"], 1)) + 1e-3 * t["mass_mean"]
            if best is None or sc > best[0]: best = (sc, b, t)
    if best is None: return None
    _, b, t = best
    comps, lab = FG.components(A[b], thr, min_mass=mm)
    j = int(np.argmin([FG.pdist(np.array(t["cens"][-1]), c["cen"], L_soup) for c in comps]))
    cen = comps[j]["cen"]; mask = lab == comps[j]["id"]
    sh = tuple(int(L_soup // 2 - round(x)) for x in cen)
    roll = lambda F: np.roll(F, sh, axis=tuple(range(N)))
    mask = ndimage.binary_dilation(roll(mask), iterations=2)
    out = {k: roll(np.asarray(st[k][b])) for k in ("A", "x", "y", "z", "c")}
    out["A"] = out["A"] * mask
    info = dict(soup_seed=b, net=t["net"], speed=t["speed"], corr=t["corr"], mass=t["mass_mean"], size=t["size_max"], vel=t["vel"],
                n_tracks=sum(len(f) for f in frames[b][-1:]))
    return out, mask, info

def embed(fields, mask, L, N, cand, key):
    import jax
    Ls = fields["A"].shape[0]; n = min(Ls, L)
    src = (slice(Ls // 2 - n // 2, Ls // 2 - n // 2 + n),) * N
    dst = (slice(L // 2 - n // 2, L // 2 - n // 2 + n),) * N
    x, y, z = _fresh_osc(key, L, N)
    A = np.zeros((L,) * N, np.float32); c = np.full((L,) * N, cand.get("c_lo", 2.5), np.float32)
    A[dst] = fields["A"][src]
    m = mask[src]
    for F, k in ((x, "x"), (y, "y"), (z, "z")):
        F[dst] = np.where(ndimage_dil(m), fields[k][src], F[dst])
    c[dst] = np.where(m, fields["c"][src], c[dst])
    return dict(A=A, x=x.astype(np.float32), y=y.astype(np.float32), z=z.astype(np.float32), c=c)

def ndimage_dil(m, it=3):
    from scipy import ndimage
    return ndimage.binary_dilation(m, iterations=it)

def stage_stats(frames, L, R, every, ns, axis, vol):
    """Треки внутри стадии → статистика популяции подвижных объектов."""
    import fl_glider as FG
    tr = FG.track(frames, L, R, every)
    ncomp = [len(f) for f in frames]
    long = [t for t in tr if t["life"] >= 0.5 * ns and t["size_max"] < 0.05 * vol and t["mass_cv"] < 0.5]
    if not long:
        return dict(n_long=0, ncomp=float(np.mean(ncomp)) if ncomp else 0.0)
    V = np.array([t["vel"] for t in long]); sp = np.linalg.norm(V, axis=1)
    U = V / (sp[:, None] + 1e-12)
    mov = [t for t in long if t["net"] >= 0.5 * R and t["straight"] > 0.7 and t["corr"] > 0.6]
    return dict(n_long=len(long), ncomp=float(np.mean(ncomp)), n_moving=len(mov), frac_moving=len(mov) / len(long),
                speed_med=float(np.median(sp)), v_axis_med=float(np.median(V[:, axis])), v_perp_med=float(np.median(np.sqrt(np.maximum(sp ** 2 - V[:, axis] ** 2, 0)))),
                pol=float(np.linalg.norm(U.mean(0))), pol_axis=float(U[:, axis].mean()), corr_med=float(np.median([t["corr"] for t in long])),
                straight_med=float(np.median([t["straight"] for t in long])), mass_med=float(np.median([t["mass_mean"] for t in long])),
                size_med=float(np.median([t["size_max"] for t in long])), net_max=float(max(t["net"] for t in long)),
                moving_speed_med=float(np.median([t["speed"] for t in mov])) if mov else 0.0)

def continuation(cand, branches, n_stage_steps, N=4, L=40, soup_steps=2000, soup_asym=None, every=50, thr=0.1, seed=0, proj_every=2,
                 save_cb=None):
    """Континуация на ПОПУЛЯЦИИ (суп после soup_steps при asym=soup_asym) + опционально изолированная «палочка» (iso=True).
    branches: dict(name, asyms[len(n_stage_steps)], axis=0, flip=False, iso=False, over={param: value})."""
    import jax, jax.numpy as jnp, fl_glider as FG
    from glider_search import ball_vol
    t0 = time.time()
    R = cand["R"]; mode = cand.get("mode", "lenia"); a0 = cand.get("asym", 0.0) if soup_asym is None else soup_asym
    use_chi = cand.get("chi", 0) != 0 or any(b.get("over", {}).get("chi", 0) != 0 for b in branches)
    Ko = jnp.asarray(FG.ring_kernel_np(L, N, 3.0))
    # суп на самой решётке L
    c0 = dict(cand, asym=a0)
    st0 = FG.init_state(jax.random.PRNGKey(seed * 1000 + cand.get("seed", 0)), L, N, R, n_blobs=cand.get("n_blobs", 6) * max(1, (L // 32) ** N), density=cand.get("density", 0.5))
    st0 = jax.tree.map(lambda a: a[None], st0)
    st0 = FG.run_chunk(st0, jax.tree.map(lambda a: a[None], FG.params(c0)), jnp.asarray(FG.ring_kernel_np(L, N, R, tuple(cand.get("peaks", (1.0,))), shift=a0))[None],
                       Ko, soup_steps, mode=mode, use_chi=use_chi)
    soup = {k: np.asarray(v[0]) for k, v in st0.items()}
    info = dict(soup_mass_frac=float(soup["A"].mean()))
    iso = None
    if any(b.get("iso") for b in branches):
        g = grow_and_pick(c0, N=N, L_soup=min(L, 32), soup_steps=soup_steps, thr=thr, seed=seed)
        if g is not None:
            fields, mask, inf2 = g; info["iso_pick"] = inf2
            iso = embed(fields, mask, L, N, cand, jax.random.PRNGKey(seed + 99))
    branches = [b for b in branches if not b.get("iso") or iso is not None]
    sts = []
    for br in branches:
        base = iso if br.get("iso") else soup
        s = {k: (np.flip(v, axis=0).copy() if br.get("flip") else v) for k, v in base.items()}
        if br.get("axis", 0) != 0:
            s = {k: np.swapaxes(v, 0, br["axis"]).copy() for k, v in s.items()}
        sts.append(s)
    st = jax.tree.map(lambda *a: jnp.stack([jnp.asarray(x) for x in a]), *sts)
    P = jax.tree.map(lambda *a: jnp.stack(a), *[FG.params(dict(cand, **br.get("over", {}))) for br in branches])
    B = len(branches); mm = 0.03 * ball_vol(N, R); vol = L ** N
    stats = [[] for _ in range(B)]; proj = [[] for _ in range(B)]; massT = [[] for _ in range(B)]
    tstep = 0; t_gpu = 0.0; nsnap = 0
    for si, ns in enumerate(n_stage_steps):
        asy = [br["asyms"][si] for br in branches]
        KA = jnp.stack([jnp.asarray(FG.ring_kernel_np(L, N, R, tuple(cand.get("peaks", (1.0,))), shift=float(a), axis=br.get("axis", 0)))
                        for a, br in zip(asy, branches)])
        frames = [[] for _ in range(B)]
        for k in range(ns // every + 1):
            if k:
                tt = time.time(); st = FG.run_chunk(st, P, KA, Ko, every, mode=mode, use_chi=use_chi); jax.block_until_ready(st["A"]); t_gpu += time.time() - tt
                tstep += every
            A = np.asarray(st["A"]); nsnap += 1
            for b in range(B):
                if not np.isfinite(A[b]).all(): frames[b].append([]); continue
                comps, _ = FG.components(A[b], thr, min_mass=mm)
                for c in comps: c["patch"] = FG.patch(A[b], c["cen"], int(R))
                frames[b].append(comps[:96]); massT[b].append(float(A[b].mean()))
                if proj_every and nsnap % proj_every == 0:
                    proj[b].append(A[b].mean(axis=tuple(range(2, N))).astype(np.float16))
        for b, br in enumerate(branches):
            s = stage_stats(frames[b], L, R, every, ns, br.get("axis", 0), vol)
            s.update(stage=si, asym=asy[b], t_end=tstep, mass_frac=massT[b][-1] if massT[b] else np.nan)
            stats[b].append(s)
        res = dict(cand=cand, branches=branches, info=info, stats=stats, n_stage_steps=n_stage_steps, N=N, L=L, every=every,
                   t_gpu=t_gpu, t_total=time.time() - t0, partial=si < len(n_stage_steps) - 1)
        if save_cb: save_cb(res)
    res["proj"] = [np.array(p) for p in proj]; res["massT"] = massT
    res["summary"] = [(br["name"], round(stats[b][-1].get("speed_med", 0), 4), round(stats[b][-1].get("frac_moving", 0), 2), stats[b][-1].get("n_long", 0))
                      for b, br in enumerate(branches)]
    return res
