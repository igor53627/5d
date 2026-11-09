"""Part 18c summary. analyze18c.py [glob] reads run_eco pickles and prints switch splits.

Does not interpret a positive within-body change as learned adaptation.
Labels use the 18c half-period rule (season_rule=1). Part 18/18b logs stay with analyze18.py.
"""
import glob
import json
import pickle
import sys
import numpy as np
import _path  # noqa: F401
import analyze_eco as AE
import fl_season as Z
import metrics18c as M

I_EA, I_EB = Z.KB, Z.KB + 1
I_GA, I_GB = Z.KB + 3, Z.KB + 4


def load(pat):
    runs = []
    for f in sorted(glob.glob(pat)):
        blob = pickle.load(open(f, "rb"))
        out = blob["out"] if isinstance(blob, dict) and "out" in blob else blob
        if isinstance(out, list):
            out = out[0] if out else {}
        # ecolist saves {tag: run_eco result}; a bare run_eco result has "runs".
        if isinstance(out, dict) and "runs" in out and isinstance(out["runs"], list):
            tagged = {out.get("tag", f): out}
        else:
            tagged = {k: v for k, v in out.items() if isinstance(v, dict) and "runs" in v}
        for tag, result in tagged.items():
            rr = result["runs"][0]
            runs.append(dict(tag=tag, log=rr["log"], eco=rr["eco"], p_init=result.get("p_init"), L=None))
    return runs


def summarize(run, L):
    log, eco = run["log"], run["eco"]
    if float(eco.get("season_rule", 0.0)) <= 0.0:
        raise ValueError("analyze18c labels the half-period season; this log has season_rule=0 (use analyze18.py)")
    Ts = float(eco.get("Ts", 8000.0))
    sigmas = [Z.snapshot_season(t, Ts) for t, _ in log]
    series = M.population_series(log)
    new = [Z.K0 + i for i in Z.new_channel_flat()]
    switches = M.decompose_switch(log, L, sigmas, I_GA, I_GB)
    # Price on g = g_A - g_B. Positive S means higher g leaves more mass.
    price_log = []
    for (t, objs), sg in zip(log, sigmas):
        price_log.append((t, [(o[0], o[1], [M.g_of(o, I_GA, I_GB)]) for o in objs]))
    S, corr, n_pairs = AE.selection_diff(price_log, L, lag=5, k=0)
    ev, _ = AE.lineage(log, L)
    return dict(
        tag=run["tag"], Ts=Ts, n_switch=len(switches), switches=switches,
        diet=M.diet_around(log, sigmas, I_EA, I_EB),
        fusion_t=M.fusion_time(series), n_end=series[-1]["n"] if series else 0,
        mass_end=series[-1]["mass"] if series else 0.0,
        max_frac_end=series[-1]["max_frac"] if series else None,
        n_min=min((r["n"] for r in series), default=0),
        divisions=len(ev),
        init=M.init_diversity(run.get("p_init")),
        new_weights=M.channel_stats(log, new, every=max(1, len(log) // 8)),
        price_S_g=S, price_corr_g=corr, price_pairs=n_pairs,
        sigma_end=sigmas[-1] if sigmas else None,
    )


def main(pat, L):
    runs = load(pat)
    if not runs:
        print("no runs in", pat)
        return 1
    summaries = [summarize(r, L) for r in runs]
    for s in summaries:
        print(f"{s['tag']}: switches {s['n_switch']}  n_end {s['n_end']}  mass_end {s['mass_end']:.2f}  "
              f"fusion_t {s['fusion_t']}  divisions {s['divisions']}  Price S(g) {s['price_S_g']}")
        for sw, diet in zip(s["switches"], s["diet"]):
            print(f"  t {sw['t_before']:.0f}->{sw['t_after']:.0f} σ {sw['sigma_before']:+.0f}->{sw['sigma_after']:+.0f} "
                  f"Δmean {sw['delta_mean']:+.3f} within {sw['within']:+.3f} fusion {sw['fusion']:+.3f} "
                  f"uncertain {sw.get('uncertain', 0):+.3f} replacement {sw['replacement']:+.3f} "
                  f"flip {sw['sign_flip_mass_frac']} matched {sw['matched_mass_frac']:.2f} "
                  f"poison {diet['poison_before']:.2f}->{diet['poison_after']:.2f}")
        if s["init"]:
            print(f"  init std_new {s['init']['std_new']:.3f} std_old {s['init']['std_old']:.3f} |w_new| {s['init']['mean_abs_new']:.3f}")
    return summaries


def json_ready(x):
    """NaN and inf become null. NumPy scalars become Python numbers. Strict JSON can then encode the tree."""
    if isinstance(x, dict):
        return {str(k): json_ready(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_ready(v) for v in x]
    if isinstance(x, np.ndarray):
        return json_ready(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def dump_summary(out, dest):
    with open(dest, "w") as f:
        json.dump(json_ready(out), f, allow_nan=False)


if __name__ == "__main__":
    pat = sys.argv[1] if len(sys.argv) > 1 else "results/eco18c/b18c_*.pkl"
    L = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    out = main(pat, L)
    if isinstance(out, list):
        dest = sys.argv[3] if len(sys.argv) > 3 else ""
        if dest:
            dump_summary(out, dest)
