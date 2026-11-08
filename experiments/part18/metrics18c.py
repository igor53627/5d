"""Part 18c metrics. Pure functions on run_eco logs.

A season switch can move the population mean of g = g_A - g_B in two ways:
the same bodies change g, or different bodies make up the mean. decompose_switch
splits those. It does not decide whether a policy was learned.
"""
import numpy as np
import analyze_eco as AE
import fl_season as Z

MATCH_RADIUS = 8.0


def g_of(obj, i_gA, i_gB):
    q = obj[2]
    return float(q[i_gA] - q[i_gB])


def match_bodies(prev, nxt, L, radius=MATCH_RADIUS):
    """One-to-one greedy match of body centers closer than radius. Returns (i_prev, j_next, dist)."""
    if not prev or not nxt:
        return []
    cands = []
    for i, a in enumerate(prev):
        for j, b in enumerate(nxt):
            d = AE.wd(a[1], b[1], L)
            if d <= radius:
                cands.append((d, i, j))
    cands.sort()
    used_i, used_j, out = set(), set(), []
    for d, i, j in cands:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        out.append((i, j, d))
    return out


def _mean(objs, i_gA, i_gB):
    if not objs:
        return np.nan, 0.0
    m = np.array([o[0] for o in objs], float)
    g = np.array([g_of(o, i_gA, i_gB) for o in objs], float)
    if m.sum() <= 0:
        return np.nan, float(m.sum())
    return float(np.average(g, weights=m)), float(m.sum())


def decompose_switch(log, L, sigmas, i_gA, i_gB, radius=MATCH_RADIUS):
    """One record per sign change between consecutive snapshots.

    within is the mass-weighted change of g in matched bodies (old masses).
    delta_mean is the change of the population mean. replacement = delta_mean - within.
    When matched_mass_frac is 0, within is nan and replacement is not a split.
    """
    if len(sigmas) != len(log):
        raise ValueError("sigmas must align with log")
    out = []
    for k in range(1, len(log)):
        if sigmas[k] == sigmas[k - 1]:
            continue
        prev, nxt = log[k - 1][1], log[k][1]
        pairs = match_bodies(prev, nxt, L, radius)
        mean0, mass0 = _mean(prev, i_gA, i_gB)
        mean1, mass1 = _mean(nxt, i_gA, i_gB)
        if pairs and mass0 > 0:
            m = np.array([prev[i][0] for i, _, _ in pairs], float)
            dg = np.array([g_of(nxt[j], i_gA, i_gB) - g_of(prev[i], i_gA, i_gB) for i, j, _ in pairs], float)
            g0 = np.array([g_of(prev[i], i_gA, i_gB) for i, _, _ in pairs], float)
            g1 = g0 + dg
            within = float(np.average(dg, weights=m))
            flip = float(np.average(np.sign(g0) != np.sign(g1), weights=m))
            matched_frac = float(m.sum() / mass0)
        else:
            within, flip, matched_frac = np.nan, np.nan, 0.0
        matched_j = {j for _, j, _ in pairs}
        entrants = [o for j, o in enumerate(nxt) if j not in matched_j]
        e_mean, e_mass = _mean(entrants, i_gA, i_gB)
        delta = mean1 - mean0
        out.append(dict(
            t_before=float(log[k - 1][0]), t_after=float(log[k][0]),
            sigma_before=float(sigmas[k - 1]), sigma_after=float(sigmas[k]),
            mean_before=mean0, mean_after=mean1, delta_mean=float(delta),
            within=within, replacement=(float(delta - within) if within == within else np.nan),
            sign_flip_mass_frac=flip, matched_mass_frac=matched_frac,
            n_before=len(prev), n_after=len(nxt), n_matched=len(pairs),
            entrant_mean=e_mean, entrant_mass_frac=(e_mass / mass1 if mass1 else np.nan),
        ))
    return out


def population_series(log):
    rows = []
    for t, objs in log:
        if not objs:
            rows.append(dict(t=float(t), n=0, mass=0.0, max_frac=np.nan))
            continue
        m = np.array([o[0] for o in objs], float)
        rows.append(dict(t=float(t), n=int(len(objs)), mass=float(m.sum()), max_frac=float(m.max() / m.sum())))
    return rows


def fusion_time(series, frac=0.5):
    """First t at which the largest body holds more than `frac` of the mass, else None.

    A two-body tie (exactly 0.5) is not fusion. Part 16 used a strict majority.
    """
    for row in series:
        if row["max_frac"] == row["max_frac"] and row["max_frac"] > frac:
            return row["t"]
    return None


def diet_around(log, sigmas, i_eA, i_eB, half=2):
    """Poison fraction in a snapshot window on each side of a sign change.

    Poison is B when sigma > 0 and A when sigma < 0. Objects contribute mass * per-mass intake.
    """
    out = []
    for k in range(1, len(log)):
        if sigmas[k] == sigmas[k - 1]:
            continue
        def window(a, b, sigma):
            eG = eP = 0.0
            for t, objs in log[a:b]:
                for o in objs:
                    eA, eB = o[0] * float(o[2][i_eA]), o[0] * float(o[2][i_eB])
                    if sigma > 0:
                        eG += eA; eP += eB
                    else:
                        eG += eB; eP += eA
            return eP / (eG + eP) if (eG + eP) > 0 else np.nan
        lo0, hi0 = max(0, k - half), k
        lo1, hi1 = k, min(len(log), k + half)
        out.append(dict(
            t_after=float(log[k][0]),
            poison_before=window(lo0, hi0, sigmas[k - 1]),
            poison_after=window(lo1, hi1, sigmas[k]),
        ))
    return out


def channel_stats(log, indices, every=1):
    """Mass-weighted mean and between-body std of the mean |weight| on `indices`, over snapshots."""
    idx = np.array(list(indices), int)
    rows = []
    for n, (t, objs) in enumerate(log):
        if n % every or not objs:
            continue
        m = np.array([o[0] for o in objs], float)
        vals = np.array([np.mean(np.abs(np.asarray(o[2], float)[idx])) for o in objs])
        rows.append(dict(
            t=float(t),
            mean=float(np.average(vals, weights=m)),
            std=float(np.sqrt(np.average((vals - np.average(vals, weights=m)) ** 2, weights=m))) if len(vals) > 1 else 0.0,
        ))
    return rows


def init_diversity(p_init):
    """p_init is the list of founder genomes stored by run_eco (length KB, no D channels)."""
    if not p_init:
        return {}
    rows = np.asarray(p_init, float)
    w = rows[:, Z.K0:Z.iM]
    new = np.array(Z.new_channel_flat())
    old = np.ones(w.shape[1], dtype=bool)
    old[new] = False
    return dict(
        n=int(len(rows)),
        std_new=float(w[:, new].std()),
        std_old=float(w[:, old].std()),
        mean_abs_new=float(np.abs(w[:, new]).mean()),
        max_abs=float(np.abs(rows).max()),
    )


def taxis_repulsion_ratio(N=4, L=24, sep=8, krep=10.0, coarse=True, dtA=0.33, amp=0.7):
    """Median |krep ∇U_far| versus |∇Ã| on one of two blobs, in cells/step before the speed cap.

    Part 18b runs with krep=10 and coarse kernels. Food taxis uses kb=1 and |g|~1.
    """
    import jax.numpy as jnp
    import fl_eco3 as E3
    ax = np.arange(L)
    grids = np.meshgrid(*[ax] * N, indexing="ij")

    def blob(center, s=1.8):
        r2 = sum(((g - c + L / 2) % L - L / 2) ** 2 for g, c in zip(grids, center))
        return amp * np.exp(-r2 / (2 * s * s)).astype(np.float32)

    c0 = (L // 2,) * N
    c1 = list(c0)
    c1[0] = (c0[0] + sep) % L
    A = blob(c0) + blob(tuple(c1))
    KX = E3.kernels(L, N, coarse=coarse)
    if coarse:
        Ufar = np.asarray(E3.up2(jnp.fft.irfftn(jnp.fft.rfftn(E3.pool2(jnp.asarray(A))) * KX["far"], s=tuple(s // 2 for s in A.shape))))
    else:
        Ufar = np.asarray(jnp.fft.irfftn(jnp.fft.rfftn(jnp.asarray(A)) * KX["far"], s=A.shape))
    x0, x1 = grids[0], grids[1]
    At = (1 + np.cos(2 * np.pi * x0 / L) * np.cos(2 * np.pi * x1 / L)) / 2

    def grad(F):
        return [0.5 * (np.roll(F, -1, k) - np.roll(F, 1, k)) for k in range(N)]

    mask = blob(c0) > 0.3

    def mag(gs):
        return np.sqrt(sum(np.asarray(g)[mask] ** 2 for g in gs))

    gU, gF = float(np.median(mag(grad(Ufar)))), float(np.median(mag(grad(At))))
    scale = float(dtA) * 10.0
    v_rep, v_food = krep * gU * scale, gF * scale
    return dict(v_rep=v_rep, v_food=v_food, ratio=(v_rep / v_food if v_food else np.nan), cap=0.45 / np.sqrt(N), krep=float(krep), coarse=bool(coarse))
