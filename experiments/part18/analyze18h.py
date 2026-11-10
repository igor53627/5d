"""Lineage fractions and within_aligned per tag for Part 18h logs (K=KB+1)."""
import glob
import pickle
import numpy as np
import _path  # noqa: F401
import fl_season as Z
import metrics18c as M
import analyze18c as A18

ND = Z.ND


def k_of(objs):
    if not objs:
        return Z.KB + 1
    return int(len(objs[0][2]) - ND)


def tag_of(obj, iL=Z.iL):
    return float(obj[2][iL])


def lineage(obj, iL=Z.iL):
    t = tag_of(obj, iL)
    if t > 0:
        return "plus"   # planted in mixed; tag+ half in control
    if t < 0:
        return "minus"  # bias_cue in mixed; tag- half in control
    return "zero"


def fraction_series(log, iL=Z.iL):
    rows = []
    for t, objs in log:
        if not objs:
            rows.append(dict(t=float(t), mass=0.0, frac_plus=np.nan, n_plus=0, n_minus=0, n=0, mean_tag=np.nan))
            continue
        m = np.array([o[0] for o in objs], float)
        tags = np.array([tag_of(o, iL) for o in objs], float)
        plus = tags > 0
        mass = float(m.sum())
        rows.append(dict(
            t=float(t), mass=mass,
            frac_plus=float(m[plus].sum() / mass) if mass else np.nan,
            n_plus=int(plus.sum()), n_minus=int((~plus).sum()), n=len(objs),
            mean_tag=float(np.average(tags, weights=m)) if mass else np.nan,
        ))
    return rows


def fixation(series, lo=0.05, hi=0.95):
    """First time frac_plus exits (lo, hi), and end state."""
    end = series[-1]["frac_plus"] if series else np.nan
    t_fix = None
    state = "polymorphic"
    if end == end:
        if end >= hi:
            state = "plus_fixed"
        elif end <= lo:
            state = "plus_extinct"
    for r in series:
        f = r["frac_plus"]
        if f != f:
            continue
        if f >= hi or f <= lo:
            t_fix = r["t"]
            break
    return dict(end_frac_plus=end, t_fix=t_fix, state=state)


def within_by_lineage(log, L, sigmas, i_gA, i_gB, iL=Z.iL):
    """decompose_switch on the full pop, then split exclusive matches by old-body tag."""
    sws = M.decompose_switch(log, L, sigmas, i_gA, i_gB)
    out = []
    # map switch index to log frames
    k_sw = 0
    for k in range(1, len(log)):
        if sigmas[k] == sigmas[k - 1]:
            continue
        rec = dict(sws[k_sw])
        k_sw += 1
        prev, nxt = log[k - 1][1], log[k][1]
        pairs = M.match_bodies(prev, nxt, L)
        sgn = 1.0 if rec["sigma_after"] > 0 else -1.0
        rec["within_aligned"] = rec["within"] * sgn
        rec["replacement_aligned"] = rec["replacement"] * sgn
        rec["direction"] = "BtoA" if rec["sigma_after"] > 0 else "AtoB"
        # mass at t_before
        mass0 = sum(o[0] for o in prev) or 0.0
        per = {}
        for name, pred in (("plus", lambda o: tag_of(o, iL) > 0), ("minus", lambda o: tag_of(o, iL) < 0)):
            sub_pairs = [(i, j, d) for i, j, d in pairs if pred(prev[i])]
            m = np.array([prev[i][0] for i, _, _ in sub_pairs], float)
            if len(sub_pairs) and m.sum() > 0 and mass0 > 0:
                g0 = np.array([M.g_of(prev[i], i_gA, i_gB) for i, _, _ in sub_pairs], float)
                g1 = np.array([M.g_of(nxt[j], i_gA, i_gB) for _, j, _ in sub_pairs], float)
                within = float(np.sum(m * (g1 - g0)) / mass0)  # share of pop mass, like decompose
                within_own = float(np.sum(m * (g1 - g0)) / m.sum())  # within that lineage
            else:
                within, within_own = float("nan"), float("nan")
            lin_mass = sum(o[0] for o in prev if pred(o))
            per[name] = dict(
                within=within, within_own=within_own,
                within_aligned=within * sgn if within == within else float("nan"),
                within_own_aligned=within_own * sgn if within_own == within_own else float("nan"),
                n_matched=len(sub_pairs), mass_frac=lin_mass / mass0 if mass0 else float("nan"),
            )
        rec["per_lineage"] = per
        out.append(rec)
    return out


def summarize_run(tag, log, eco, L=48, p_init=None):
    K = k_of(log[-1][1] if log else []) or (Z.KB + 1)
    i_gA, i_gB = K + 3, K + 4
    Ts = float(eco.get("Ts", 8000.0))
    sigmas = [Z.snapshot_season(t, Ts) for t, _ in log]
    series = fraction_series(log)
    fix = fixation(series)
    sws = within_by_lineage(log, L, sigmas, i_gA, i_gB)
    # overall metrics with correct I_GA
    base = A18.summarize(dict(tag=tag, log=log, eco=eco, p_init=p_init), L) if K == Z.KB else None
    # when K!=KB, compute diet/n_end ourselves via series-like
    n_end = len(log[-1][1]) if log else 0
    mass_end = float(sum(o[0] for o in log[-1][1])) if log else 0.0
    def mean_g(idx):
        objs = log[-1][1]
        if not objs:
            return float("nan")
        m = np.array([o[0] for o in objs], float)
        g = np.array([float(o[2][idx]) for o in objs], float)
        return float(np.average(g, weights=m)) if m.sum() else float("nan")
    gA, gB = mean_g(i_gA), mean_g(i_gB)
    wals = [s["within_aligned"] for s in sws if s["within_aligned"] == s["within_aligned"]]
    rals = [s["replacement_aligned"] for s in sws if s["replacement_aligned"] == s["replacement_aligned"]]
    by_dir = {}
    for d in ("AtoB", "BtoA"):
        xs = [s for s in sws if s["direction"] == d]
        if not xs:
            continue
        by_dir[d] = dict(
            within_aligned=float(np.nanmean([s["within_aligned"] for s in xs])),
            replacement_aligned=float(np.nanmean([s["replacement_aligned"] for s in xs])),
        )
    lin_wal = {}
    for name in ("plus", "minus"):
        xs = [s["per_lineage"][name]["within_own_aligned"] for s in sws]
        xs = [x for x in xs if x == x]
        lin_wal[name] = float(np.mean(xs)) if xs else float("nan")
    return dict(
        tag=tag, arm=tag.rsplit("_s", 1)[0], seed=int(tag.rsplit("_s", 1)[1]),
        K=K, n_end=n_end, mass_end=mass_end, mean_g_A=gA, mean_g_B=gB,
        crossed=bool(gA == gA and gB == gB and gA > 0 and gB < 0),
        frac_plus_init=series[0]["frac_plus"] if series else None,
        frac_plus_end=fix["end_frac_plus"], t_fix=fix["t_fix"], state=fix["state"],
        series=series, switches=sws, fixation=fix,
        mean_within_aligned=float(np.mean(wals)) if wals else float("nan"),
        mean_replacement_aligned=float(np.mean(rals)) if rals else float("nan"),
        by_direction=by_dir,
        within_own_aligned_plus=lin_wal["plus"],
        within_own_aligned_minus=lin_wal["minus"],
    )
