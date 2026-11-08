import _path  # noqa: F401
_root = _path._root
_archive = str(_root / "archive") if _root is not None else ""
"""Батч-поиск движущихся структур: прогон кандидатов (vmap) + детекторы + треки."""
import time, json, numpy as np

def ball_vol(N, R):
    from math import gamma, pi
    return pi ** (N / 2) / gamma(N / 2 + 1) * R ** N

def run_candidates(cands, N=4, L=32, steps=3000, every=25, thr=0.1, keep_frames=True, proj_every=1, seed=0, slice_every=0, vol_every=0, vol_ds=2):
    import jax, jax.numpy as jnp, fl_glider as FG
    t0 = time.time()
    mode = cands[0].get("mode", "lenia"); use_chi = any(c.get("chi", 0) != 0 for c in cands)
    assert all(c.get("mode", "lenia") == mode for c in cands)
    KA = jnp.stack([jnp.asarray(FG.ring_kernel_np(L, N, c["R"], tuple(c.get("peaks", (1.0,))), shift=c.get("asym", 0.0))) for c in cands])
    Ko = jnp.asarray(FG.ring_kernel_np(L, N, 3.0))
    P = jax.tree.map(lambda *a: jnp.stack(a), *[FG.params(c) for c in cands])
    sts = [FG.init_state(jax.random.PRNGKey(seed * 1000 + c.get("seed", i)), L, N, c["R"], n_blobs=c.get("n_blobs", 6),
                         blob_r=c.get("blob_r", 1.0), density=c.get("density", 0.5)) for i, c in enumerate(cands)]
    st = jax.tree.map(lambda *a: jnp.stack(a), *sts)
    B = len(cands)
    frames = [[] for _ in range(B)]; mproj = [[] for _ in range(B)]; proj = [[] for _ in range(B)]; cproj = [[] for _ in range(B)]; massT = [[] for _ in range(B)]
    slices = [[] for _ in range(B)]; vols = [[] for _ in range(B)]
    nchunk = steps // every
    t_gpu = 0.0
    for ch in range(nchunk + 1):
        if ch > 0:
            tt = time.time(); st = FG.run_chunk(st, P, KA, Ko, every, mode=mode, use_chi=use_chi); jax.block_until_ready(st["A"]); t_gpu += time.time() - tt
        A = np.asarray(st["A"]); C = np.asarray(st["c"])
        for b in range(B):
            R = cands[b]["R"]
            comps, _ = FG.components(A[b], thr, min_mass=0.03 * ball_vol(N, R), C=C[b],
                                     c_lo=cands[b].get("c_lo", 2.5), c_hi=cands[b].get("c_hi", 5.0))
            for c in comps: c["patch"] = FG.patch(A[b], c["cen"], int(R))
            frames[b].append(comps[:64])
            massT[b].append(float(A[b].sum()))
            if keep_frames and ch % proj_every == 0:
                ax = tuple(range(2, N))
                proj[b].append(A[b].max(axis=ax).astype(np.float16)); mproj[b].append(A[b].mean(axis=ax).astype(np.float16)); cproj[b].append(C[b].mean(axis=ax).astype(np.float16))
            if vol_every and ch % vol_every == 0:   # прореженный полный N-D объём (для срезов через треки)
                vols[b].append(A[b][(slice(None, None, vol_ds),) * N].astype(np.float16))
            if slice_every and ch % slice_every == 0:
                slices[b].append(A[b][(slice(None), slice(None)) + (L // 2,) * (N - 2)].astype(np.float16))
        if not np.isfinite(A).all(): break
    out = []
    for b in range(B):
        R = cands[b]["R"]
        tr = FG.track(frames[b], L, R, every)
        g, stat = FG.score_tracks(tr, R, steps, vol_total=L ** N)
        ncomp = [len(f) for f in frames[b]]
        out.append(dict(cand=cands[b], n_glider=len(g), n_static=len(stat), ncomp_end=ncomp[-1], ncomp=ncomp,
                        mass=massT[b], gliders=[{k: v for k, v in t.items()} for t in g],
                        tracks=[t for t in sorted(tr, key=lambda t: -t["life"])[:40]],
                        proj=np.array(proj[b]) if keep_frames else None, cproj=np.array(cproj[b]) if keep_frames else None, mproj=np.array(mproj[b]) if keep_frames else None,
                        slices=np.array(slices[b]) if slice_every else None, vols=np.array(vols[b]) if vol_every else None,
                        tracks_all=tr if len(tr) < 3000 else None,
                        speeds=[t["speed"] for t in g]))
    return dict(N=N, L=L, steps=steps, every=every, B=B, t_gpu=t_gpu, t_total=time.time() - t0, out=out)
