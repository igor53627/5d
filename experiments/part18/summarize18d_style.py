"""Build per-run / per-arm summaries matching full18d_summary.json schema."""
import glob
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import _path  # noqa: F401
import fl_season as Z
import metrics18c as M
import analyze18c as A18
import init18c as I18c

I_EA, I_EB = A18.I_EA, A18.I_EB
I_GA, I_GB = A18.I_GA, A18.I_GB


def _mean_g(objs, idx):
    if not objs:
        return float("nan")
    m = np.array([o[0] for o in objs], float)
    g = np.array([float(o[2][idx]) for o in objs], float)
    return float(np.average(g, weights=m)) if m.sum() > 0 else float("nan")


def _poison_by_season(log, sigmas):
    """Mass-weighted poison fraction of diet in A vs B seasons (excluding t=0)."""
    pa, pb, ma, mb = [], [], [], []
    for (t, objs), sg in zip(log, sigmas):
        if t <= 0 or not objs:
            continue
        # D0=eaten A, D1=eaten B stored on objects? In eco logs, objs carry phenotype;
        # diet uses metrics18c helpers. Fall back to series fields if present.
        pass
    # Use M.diet-style aggregation from per-snapshot object D channels if available.
    # Object layout: o = (mass, cen, p_or_pheno...). Season poison from analyze path:
    eat_a = {1.0: [], -1.0: []}
    eat_b = {1.0: [], -1.0: []}
    for (t, objs), sg in zip(log, sigmas):
        if t <= 0 or not objs:
            continue
        # Prefer D channels appended after KB in phenotype vector when present.
        for o in objs:
            vec = np.asarray(o[2], float)
            if vec.size <= I_EB:
                continue
            m = float(o[0])
            eat_a[sg].append(m * float(vec[I_EA]))
            eat_b[sg].append(m * float(vec[I_EB]))
    out = {}
    for sg, name in ((1.0, "A"), (-1.0, "B")):
        ea, eb = sum(eat_a[sg]), sum(eat_b[sg])
        tot = ea + eb
        # poison = B when A-season (sg>0), A when B-season
        poi = (eb if sg > 0 else ea) / tot if tot > 0 else float("nan")
        out[f"poison_{name}"] = float(poi) if poi == poi else None
    return out


def _crossed(gA, gB):
    return bool(gA == gA and gB == gB and gA > 0 and gB < 0)


def _switcher_stats(objs):
    if not objs:
        return 0, 0.0
    m = np.array([o[0] for o in objs], float)
    flags = np.array([float(I18c.is_switcher(np.asarray(o[2][Z.K0:Z.iM], float))) for o in objs])
    return int(flags.sum()), float(np.average(flags, weights=m)) if m.sum() else 0.0


def _bias_d(objs):
    """Mean g_A - g_B."""
    if not objs:
        return float("nan")
    m = np.array([o[0] for o in objs], float)
    g = np.array([M.g_of(o, I_GA, I_GB) for o in objs], float)
    return float(np.average(g, weights=m)) if m.sum() else float("nan")


def enrich(base, run):
    """Add 18d-style fields onto an analyze18c summary."""
    log, eco = run["log"], run["eco"]
    Ts = float(eco.get("Ts", 8000.0))
    sigmas = [Z.snapshot_season(t, Ts) for t, _ in log]
    tag = base["tag"]
    # arm = tag without _sSEED
    arm = tag.rsplit("_s", 1)[0]
    seed = int(tag.rsplit("_s", 1)[1])
    gA = _mean_g(log[-1][1], I_GA) if log else float("nan")
    gB = _mean_g(log[-1][1], I_GB) if log else float("nan")
    # sign flips of mean (gA-gB) across log
    diffs = []
    for _, objs in log:
        diffs.append(_bias_d(objs))
    flips = 0
    for a, b in zip(diffs, diffs[1:]):
        if a == a and b == b and np.sign(a) != np.sign(b) and abs(a) > 1e-6 and abs(b) > 1e-6:
            flips += 1
    poi = _poison_by_season(log, sigmas)
    sw0 = _switcher_stats(log[0][1]) if log else (0, 0.0)
    sw1 = _switcher_stats(log[-1][1]) if log else (0, 0.0)
    bd0 = _bias_d(log[0][1]) if log else float("nan")
    bd1 = _bias_d(log[-1][1]) if log else float("nan")
    withins = [sw["within"] for sw in base["switches"] if sw["within"] == sw["within"]]
    flips_f = [sw["sign_flip_mass_frac"] for sw in base["switches"] if sw["sign_flip_mass_frac"] == sw["sign_flip_mass_frac"]]
    return dict(
        tag=tag, arm=arm, seed=seed,
        mut_bias=eco.get("mut_bias"), cue=float(eco.get("cue", 0.0)),
        cue_lead=float(eco.get("cue_lead", 0.0)), dec=float(eco.get("dec", 0.0)),
        n_end=base["n_end"], n_min=base["n_min"], mass_end=base["mass_end"],
        max_frac_end=base["max_frac_end"], fusion_t=base["fusion_t"],
        divisions=base["divisions"], n_switch=base["n_switch"],
        switches=base["switches"], diet=base["diet"],
        mean_g_A=gA, mean_g_B=gB, mean_g_sign_flips=flips,
        crossed=_crossed(gA, gB),
        poison_A=poi.get("poison_A"), poison_B=poi.get("poison_B"),
        switchers_init=sw0[0], switchers_end=sw1[0],
        switcher_mass_frac_end=sw1[1],
        bias_d_init=bd0, bias_d_end=bd1,
        mean_within=float(np.mean(withins)) if withins else float("nan"),
        mean_replacement=float(np.mean([sw["replacement"] for sw in base["switches"] if sw["replacement"] == sw["replacement"]])) if base["switches"] else float("nan"),
        mean_fusion=float(np.mean([sw["fusion"] for sw in base["switches"] if sw["fusion"] == sw["fusion"]])) if base["switches"] else float("nan"),
        mean_uncertain=float(np.mean([sw.get("uncertain", 0.0) for sw in base["switches"]])) if base["switches"] else float("nan"),
        mean_flip_frac=float(np.mean(flips_f)) if flips_f else float("nan"),
        within_last=withins[-1] if withins else float("nan"),
    )


def aggregate(runs):
    by = defaultdict(list)
    for r in runs:
        by[r["arm"]].append(r)
    out = {}
    for arm, rs in by.items():
        def mean(key):
            xs = [r[key] for r in rs if r.get(key) is not None and r[key] == r[key]]
            return float(np.mean(xs)) if xs else None
        out[arm] = dict(
            n_seeds=len(rs),
            n_end_mean=mean("n_end"),
            mass_mean=mean("mass_end"),
            max_frac_mean=mean("max_frac_end"),
            fusion_any=any(r["fusion_t"] is not None for r in rs),
            gA_mean=mean("mean_g_A"),
            gB_mean=mean("mean_g_B"),
            crossed_frac=float(np.mean([1.0 if r["crossed"] else 0.0 for r in rs])),
            poison_B_mean=mean("poison_B"),
            poison_A_mean=mean("poison_A"),
            within_mean=mean("mean_within"),
            replacement_mean=mean("mean_replacement"),
            fusion_mean=mean("mean_fusion"),
            uncertain_mean=mean("mean_uncertain"),
            within_last_mean=mean("within_last"),
            flip_frac_mean=mean("mean_flip_frac"),
            msw_end_mean=mean("switcher_mass_frac_end"),
            sw_end_mean=mean("switchers_end"),
            bias_d_end_mean=mean("bias_d_end"),
            per_seed={str(r["seed"]): dict(
                n_end=r["n_end"], mass=r["mass_end"], gA=r["mean_g_A"], gB=r["mean_g_B"],
                crossed=r["crossed"], poison_B=r["poison_B"], within_mean=r["mean_within"],
                within_last=r["within_last"], replacement_mean=r["mean_replacement"],
                sw_end=r["switchers_end"], msw=r["switcher_mass_frac_end"],
                fusion_t=r["fusion_t"], bias_d_end=r["bias_d_end"],
                cue_lead=r.get("cue_lead"), dec=r.get("dec"),
            ) for r in rs},
        )
    return out


def summarize_glob(pat, L=48):
    raw = A18.load(pat)
    # Attach L from first pickle if needed
    runs = []
    for r in raw:
        base = A18.summarize(r, L)
        runs.append(enrich(base, r))
    return runs


def json_ready(x):
    return A18.json_ready(x)


if __name__ == "__main__":
    pat = sys.argv[1]
    dest = sys.argv[2]
    L = int(sys.argv[3]) if len(sys.argv) > 3 else 48
    runs = summarize_glob(pat, L)
    agg = aggregate(runs)
    out = dict(runs=runs, aggregates=agg)
    Path(dest).write_text(json.dumps(json_ready(out), indent=2) + "\n")
    print("wrote", dest, "n=", len(runs))
