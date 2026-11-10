"""Summarize Part 19 predator–prey ecology logs (Modal ecolist pickles)."""
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import _path  # noqa: F401
import metrics19 as M
import run19 as R

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "part19" / "results" / "pilot19_summary.json"

ARMS = list(R.ARMS)
KPRED = dict(R.KPRED)
# Default: only count finished run_eco results that reached the planned horizon.
MIN_T_FINAL = R.STEPS


def load_job(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d


def _incomplete_reason(res, run, min_t=None):
    """Why a run_eco result or one of its ecos should not count as finished."""
    if res.get("partial") or run.get("partial"):
        return "partial"
    if min_t is not None:
        t = run["log"][-1][0] if run.get("log") else None
        if t is None or float(t) < float(min_t) - 1e-6:
            return "short"
    return None


def iter_runs(job, include_partial=False, min_t=None):
    """Yield (tag, arm, seed, run_dict, meta). Skips unfinished checkpoints by default."""
    out = job.get("out")
    if isinstance(out, list) and out:
        payload = out[0]
    else:
        payload = out
    if not isinstance(payload, dict):
        return
    base_meta = dict(gpu=job.get("gpu"), wall=job.get("wall"), kind=job.get("kind"))
    # ecolist: {tag: run_eco_result, summary: ...}
    for tag, res in payload.items():
        if tag == "summary" or not isinstance(res, dict) or "runs" not in res:
            continue
        for i, run in enumerate(res["runs"]):
            reason = _incomplete_reason(res, run, min_t=min_t)
            eco = run.get("eco") or {}
            kpred = float(eco.get("kpred", float("nan")))
            arm = None
            for name, val in KPRED.items():
                if abs(kpred - val) < 1e-12:
                    arm = name
                    break
            if arm is None:
                arm = f"kpred_{kpred}"
            # tag is like pp_s191; seed from the run_eco kwargs stored? not always.
            seed = None
            if tag.startswith("pp_s"):
                try:
                    seed = int(tag.split("s")[-1])
                except ValueError:
                    seed = None
            meta = dict(base_meta, incomplete=reason, tag=tag)
            if reason and not include_partial:
                continue
            yield tag, arm, seed, run, meta


def summarize_paths(paths, min_t=MIN_T_FINAL):
    rows = []
    skipped = []
    walls = []
    gpus = set()
    for path in paths:
        job = load_job(path)
        if job.get("wall") is not None:
            walls.append(float(job["wall"]))
        if job.get("gpu"):
            gpus.add(str(job["gpu"]))
        for tag, arm, seed, run, meta in iter_runs(job, include_partial=True, min_t=min_t):
            rec = dict(file=Path(path).name, tag=tag, arm=arm, seed=seed,
                       kpred=float((run.get("eco") or {}).get("kpred", float("nan"))),
                       gpu=meta.get("gpu"), wall=meta.get("wall"),
                       incomplete=meta.get("incomplete"))
            if meta.get("incomplete"):
                skipped.append(rec)
                continue
            rec.update(M.summarize_run(run))
            rows.append(rec)
    by_arm = {}
    for arm in sorted({r["arm"] for r in rows}, key=lambda a: KPRED.get(a, 99)):
        sub = [r for r in rows if r["arm"] == arm]
        def mean(key):
            xs = [r[key] for r in sub if r.get(key) is not None]
            return float(np.mean(xs)) if xs else None
        coexist = sum(1 for r in sub if r["status"] == "coexist")
        by_arm[arm] = dict(
            n=len(sub),
            kpred=sub[0]["kpred"] if sub else None,
            n_coexist=coexist,
            n_prey_only=sum(1 for r in sub if r["status"] == "prey_only"),
            n_pred_only=sum(1 for r in sub if r["status"] == "pred_only"),
            n_both_extinct=sum(1 for r in sub if r["status"] == "both_extinct"),
            n_coexist_bodies=sum(1 for r in sub if r.get("status_bodies") == "coexist"),
            status_bodies=[r.get("status_bodies") for r in sub],
            coexist_bodies_t=[r.get("coexist_bodies_t") for r in sub],
            mean_coexist_t=mean("coexist_t"),
            mean_mA=mean("mA_mean"),
            mean_mP=mean("mP_mean"),
            mean_mA_final=mean("mA_final"),
            mean_mP_final=mean("mP_final"),
            mean_peaks_A=mean("peaks_A"),
            mean_peaks_P=mean("peaks_P"),
            mean_period_A=mean("period_A"),
            mean_pred_lag=mean("pred_lag"),
            mean_kill=mean("kill_mean"),
            seeds=[r["seed"] for r in sub],
            status=[r["status"] for r in sub],
            coexist_t=[r["coexist_t"] for r in sub],
        )
    any_coexist = any(r["status"] == "coexist" for r in rows)
    any_bodies = any(r.get("status_bodies") == "coexist" for r in rows)
    return dict(
        n_files=len(paths), n_runs=len(rows),
        n_skipped_incomplete=len(skipped),
        skipped_incomplete=skipped,
        walls_s=walls, gpu_hours=sum(walls) / 3600.0 if walls else 0.0,
        gpus=sorted(gpus),
        by_arm=by_arm,
        runs=rows,
        any_coexist=any_coexist,
        any_bodies=any_bodies,
        verdict=("mass coexistence found; see status_bodies for intact prey/predator bodies"
                 if any_coexist else
                 "no arm coexisted to the end; retune kpred/krep_x/eff before a full run"),
    )


def main(argv):
    paths = [Path(p) for p in argv] if argv else []
    if not paths:
        raise SystemExit("analyze19.py pickle [pickle ...]")
    summary = summarize_paths(paths)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(M.jsonable(summary), indent=1, allow_nan=False) + "\n")
    print("wrote", OUT)
    print("gpu_hours", round(summary["gpu_hours"], 3), "verdict:", summary["verdict"])
    for arm, rec in summary["by_arm"].items():
        print(f"  {arm} kpred={rec['kpred']} coexist={rec['n_coexist']}/{rec['n']} "
              f"status={rec['status']} coexist_t={rec['coexist_t']}")
    return summary


if __name__ == "__main__":
    main(sys.argv[1:])
