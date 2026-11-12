"""Summarize Part 19d kpred=0.005 replicates and pool with 19c k005 seeds."""
import json
import math
import pickle
import sys
from pathlib import Path
import _path  # noqa: F401
import metrics19 as M
import run19d as R
import analyze19 as A19
import analyze19c as C

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "part19" / "results" / "full19d_summary.json"

ARM = R.ARM
KPRED = float(R.KPRED)
BOOT_SEED = 19
MIN_T_FINAL = R.STEPS


def load_job(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def iter_k005(job, include_partial=False, min_t=None):
    """Yield completed (or all, if include_partial) kpred=0.005 runs."""
    out = job.get("out")
    payload = out[0] if isinstance(out, list) and out else out
    if not isinstance(payload, dict):
        return
    base_meta = dict(gpu=job.get("gpu"), wall=job.get("wall"), kind=job.get("kind"))
    for tag, res in payload.items():
        if tag == "summary" or not isinstance(res, dict) or "runs" not in res:
            continue
        n_pred = int(res.get("n_pred") or 0)
        for run in res["runs"]:
            reason = A19._incomplete_reason(res, run, min_t=min_t)
            eco = run.get("eco") or {}
            kpred = float(eco.get("kpred", float("nan")))
            if int(n_pred or 0) == 0 or abs(kpred - KPRED) > 1e-12:
                continue
            seed = None
            for part in tag.split("_"):
                if part.startswith("s") and part[1:].isdigit():
                    seed = int(part[1:])
            meta = dict(base_meta, incomplete=reason, tag=tag)
            if reason and not include_partial:
                continue
            yield tag, seed, run, meta


def _arm_block(sub):
    return C._arm_block(sub)


def binomial_ci(k, n, alpha=0.05):
    """Clopper–Pearson 95% CI for a binomial proportion (k successes in n)."""
    k, n = int(k), int(n)
    if n <= 0:
        return dict(k=0, n=0, p=None, lo=None, hi=None, method="clopper-pearson")
    p = k / n
    # beta quantiles: lo = BetaInv(alpha/2; k, n-k+1), hi = BetaInv(1-alpha/2; k+1, n-k)
    # Use scipy when available; otherwise a Wilson interval as a fallback.
    lo = hi = None
    try:
        from scipy.stats import beta
        lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
        hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
        method = "clopper-pearson"
    except Exception:
        z = 1.959963984540054
        den = 1.0 + z * z / n
        center = (p + z * z / (2 * n)) / den
        half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
        lo, hi = max(0.0, center - half), min(1.0, center + half)
        method = "wilson"
    return dict(k=k, n=n, p=p, lo=lo, hi=hi, method=method)


def summarize_paths(paths, min_t=MIN_T_FINAL):
    rows, skipped, walls, gpus = [], [], [], set()
    for path in paths:
        job = load_job(path)
        name = Path(path).name
        is_19d = name.startswith("b19d_")
        if is_19d and job.get("wall") is not None:
            walls.append(float(job["wall"]))
        if is_19d and job.get("gpu"):
            gpus.add(str(job["gpu"]))
        for tag, seed, run, meta in iter_k005(job, include_partial=True, min_t=min_t):
            rec = dict(
                file=name, tag=tag, arm=ARM, seed=seed,
                kpred=float((run.get("eco") or {}).get("kpred", float("nan"))),
                gpu=meta.get("gpu"), wall=meta.get("wall"),
                incomplete=meta.get("incomplete"),
                source="19d" if is_19d else "19c",
            )
            if meta.get("incomplete"):
                skipped.append(rec)
                continue
            rec.update(M.summarize_run(run))
            rows.append(rec)
    rows19d = [r for r in rows if r["source"] == "19d"]
    rows19c = [r for r in rows if r["source"] == "19c"]
    all_block = _arm_block(rows) if rows else None
    pooled = None
    if all_block is not None:
        n = all_block["n"]
        k = all_block["n_held_bodies"]
        pooled = dict(
            n=n,
            n_19d=len(rows19d),
            n_19c=len(rows19c),
            seeds=all_block["seeds"],
            n_held_pred=all_block["n_held_pred"],
            n_held_prey=all_block["n_held_prey"],
            n_held_bodies=k,
            frac_held_pred=all_block["frac_held_pred"],
            frac_held_prey=all_block["frac_held_prey"],
            frac_held_bodies=all_block["frac_held_bodies"],
            both_held_binomial=binomial_ci(k, n),
            status_bodies=all_block["status_bodies"],
            pred_loss_t=all_block["pred_loss_t"],
            ci=all_block["ci"],
        )
    return dict(
        n_files=len(paths), n_runs=len(rows),
        n_skipped_incomplete=len(skipped),
        skipped_incomplete=skipped,
        walls_s=walls, gpu_hours=sum(walls) / 3600.0 if walls else 0.0,
        gpus=sorted(gpus),
        by_source=dict(
            k005_19d=_arm_block(rows19d) if rows19d else {},
            k005_19c=_arm_block(rows19c) if rows19c else {},
        ),
        pooled_k005=pooled,
        runs=rows,
    )


def main(argv):
    paths = [Path(p) for p in argv] if argv else []
    if not paths:
        raise SystemExit("analyze19d.py pickle [pickle ...]")
    summary = summarize_paths(paths)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(M.jsonable(summary), indent=1, allow_nan=False) + "\n")
    print("wrote", OUT)
    print("gpu_hours", round(summary["gpu_hours"], 3),
          "n_runs", summary["n_runs"],
          "skipped", summary["n_skipped_incomplete"])
    for src, rec in summary["by_source"].items():
        if not rec:
            print(f"  {src}: (none)")
            continue
        ci_t = rec["ci"]["coexist_bodies_t"]
        ci_loss = rec["ci"]["time_to_pred_loss"]
        cens = "censored" if ci_loss.get("right_censored") else "observed"
        print(f"  {src} held_pred={rec['n_held_pred']}/{rec['n']} "
              f"held_prey={rec['n_held_prey']}/{rec['n']} "
              f"held_both={rec['n_held_bodies']}/{rec['n']} "
              f"bodies={rec['status_bodies']} "
              f"coexist_t={_fmt_ci(ci_t)} pred_loss_t={_fmt_ci(ci_loss)} ({cens})")
    pool = summary.get("pooled_k005") or {}
    if pool:
        b = pool["both_held_binomial"]
        print(f"  pooled k005 both_held={b['k']}/{b['n']} "
              f"p={b['p']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}] ({b['method']})")
    return summary


def _fmt_ci(c):
    if not c or c.get("mean") is None:
        return "na"
    if c.get("lo") is None:
        return f"{c['mean']:.0f}"
    return f"{c['mean']:.0f} [{c['lo']:.0f}, {c['hi']:.0f}]"


if __name__ == "__main__":
    main(sys.argv[1:])
