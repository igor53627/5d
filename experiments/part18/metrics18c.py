"""Part 18c metrics. Pure functions on run_eco logs.

A season switch can move the population mean of g = g_A - g_B because the same
body changed, because bodies fused or divided, or because the cast of bodies changed.
decompose_switch splits those. It does not decide whether a policy was learned.
"""
import numpy as np
import analyze_eco as AE
import fl_season as Z

MATCH_RADIUS = 8.0


def g_of(obj, i_gA, i_gB):
    q = obj[2]
    return float(q[i_gA] - q[i_gB])


def _links(prev, nxt, L, radius):
    """old->news and new->olds whose centers are within radius, plus the distance of each pair."""
    old_n = [[] for _ in prev]
    new_n = [[] for _ in nxt]
    dist = {}
    for i, a in enumerate(prev):
        for j, b in enumerate(nxt):
            d = AE.wd(a[1], b[1], L)
            if d <= radius:
                old_n[i].append(j)
                new_n[j].append(i)
                dist[(i, j)] = d
    return old_n, new_n, dist


def match_bodies(prev, nxt, L, radius=MATCH_RADIUS):
    """Exclusive one-to-one matches. A body with two neighbors is a fusion or a division, not a match."""
    if not prev or not nxt:
        return []
    old_n, new_n, dist = _links(prev, nxt, L, radius)
    pairs = []
    for i, js in enumerate(old_n):
        if len(js) == 1 and len(new_n[js[0]]) == 1:
            j = js[0]
            pairs.append((i, j, dist[(i, j)]))
    return pairs


def _fusion_groups(prev, nxt, old_n, new_n, paired_old, paired_new):
    """Connected clusters that are not exclusive one-to-one matches."""
    seen_o = set(paired_old)
    seen_n = set(paired_new)
    groups = []
    for start in range(len(prev)):
        if start in seen_o or not old_n[start]:
            continue
        olds, news = [], []
        stack = [("o", start)]
        seen_o.add(start)
        while stack:
            kind, k = stack.pop()
            if kind == "o":
                olds.append(k)
                for j in old_n[k]:
                    if j not in seen_n:
                        seen_n.add(j)
                        stack.append(("n", j))
            else:
                news.append(k)
                for i in new_n[k]:
                    if i not in seen_o:
                        seen_o.add(i)
                        stack.append(("o", i))
        groups.append((olds, news))
    return groups


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

    Terms are contributions to the change in the population mean, each divided
    by the old population mass:
    within + fusion + uncertain + replacement = delta_mean.
    within uses only exclusive one-to-one matches. fusion is only clusters whose
    old/new cardinality differs (a confirmed merge or split). uncertain is a
    same-cardinality connected cluster of nearby bodies that never merged or
    split (mass reweighting among neighbors inside the match radius). replacement
    is births, deaths, and the residual mass reweighting.
    """
    if len(sigmas) != len(log):
        raise ValueError("sigmas must align with log")
    out = []
    for k in range(1, len(log)):
        if sigmas[k] == sigmas[k - 1]:
            continue
        prev, nxt = log[k - 1][1], log[k][1]
        mean0, mass0 = _mean(prev, i_gA, i_gB)
        mean1, mass1 = _mean(nxt, i_gA, i_gB)
        delta = mean1 - mean0
        old_n, new_n, _ = _links(prev, nxt, L, radius) if prev and nxt else ([[] for _ in prev], [[] for _ in nxt], {})
        pairs = match_bodies(prev, nxt, L, radius)
        if pairs and mass0 > 0:
            m = np.array([prev[i][0] for i, _, _ in pairs], float)
            g0 = np.array([g_of(prev[i], i_gA, i_gB) for i, _, _ in pairs], float)
            g1 = np.array([g_of(nxt[j], i_gA, i_gB) for _, j, _ in pairs], float)
            within = float(np.sum(m * (g1 - g0)) / mass0)
            flip = float(np.sum(m * (np.sign(g0) != np.sign(g1))) / m.sum()) if m.sum() else 0.0
            matched_frac = float(m.sum() / mass0)
        else:
            within, flip, matched_frac = (0.0, 0.0, 0.0) if mass0 > 0 else (np.nan, np.nan, 0.0)
        paired_old = {i for i, _, _ in pairs}
        paired_new = {j for _, j, _ in pairs}
        fusion = 0.0 if mass0 > 0 else np.nan
        uncertain = 0.0 if mass0 > 0 else np.nan
        groups = _fusion_groups(prev, nxt, old_n, new_n, paired_old, paired_new) if mass0 > 0 else []
        fusion_groups = []
        uncertain_groups = []
        for olds, news in groups:
            mo = np.array([prev[i][0] for i in olds], float)
            mn = np.array([nxt[j][0] for j in news], float)
            if mo.sum() <= 0 or mn.sum() <= 0:
                continue
            go = np.average([g_of(prev[i], i_gA, i_gB) for i in olds], weights=mo)
            gn = np.average([g_of(nxt[j], i_gA, i_gB) for j in news], weights=mn)
            contrib = float(mo.sum() / mass0 * (gn - go))
            # Same cardinality: nearby bodies that never merged or split.
            # Mass reweighting among them is not fusion (Greptile P1).
            if len(olds) == len(news):
                uncertain += contrib
                uncertain_groups.append((olds, news))
            else:
                fusion += contrib
                fusion_groups.append((olds, news))
        entrants = [nxt[j] for j in range(len(nxt)) if j not in paired_new and not new_n[j]]
        e_mean, e_mass = _mean(entrants, i_gA, i_gB)
        replacement = (float(delta - within - fusion - uncertain) if mass0 > 0 and delta == delta else np.nan)
        out.append(dict(
            t_before=float(log[k - 1][0]), t_after=float(log[k][0]),
            sigma_before=float(sigmas[k - 1]), sigma_after=float(sigmas[k]),
            mean_before=mean0, mean_after=mean1, delta_mean=float(delta) if delta == delta else delta,
            within=within, fusion=fusion, uncertain=uncertain, replacement=replacement,
            sign_flip_mass_frac=flip, matched_mass_frac=matched_frac,
            n_before=len(prev), n_after=len(nxt), n_matched=len(pairs),
            n_fusion_old=int(sum(len(a) for a, _ in fusion_groups)),
            n_fusion_new=int(sum(len(b) for _, b in fusion_groups)),
            n_uncertain_old=int(sum(len(a) for a, _ in uncertain_groups)),
            n_uncertain_new=int(sum(len(b) for _, b in uncertain_groups)),
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
