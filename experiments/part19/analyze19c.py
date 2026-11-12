"""Summarize Part 19c low-side kpred scan, plus the 19b kpred=0.015 arm."""
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import _path  # noqa: F401
import metrics19 as M
import run19c as R
import analyze19 as A19

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "part19" / "results" / "full19c_summary.json"

# 19c arms plus the 19b low comparison arm.
KPRED = dict(R.KPRED)
KPRED[R.COMPARE_ARM] = R.COMPARE_KPRED
ARMS = list(R.ARMS) + [R.COMPARE_ARM]
BOOT_SEED = 19
MIN_T_FINAL = R.STEPS


def load_job(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _arm_of(kpred, n_pred):
    if int(n_pred or 0) == 0 or abs(float(kpred)) < 1e-15:
        return None  # prey-only control, ignored in 19c
    for name, val in KPRED.items():
        if abs(float(kpred) - val) < 1e-12:
            return name
    return None


def iter_runs(job, include_partial=False, min_t=None):
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
            arm = _arm_of(kpred, n_pred)
            if arm is None:
                continue
            seed = None
            for part in tag.split("_"):
                if part.startswith("s") and part[1:].isdigit():
                    seed = int(part[1:])
            meta = dict(base_meta, incomplete=reason, tag=tag)
            if reason and not include_partial:
                continue
            yield tag, arm, seed, run, meta


def _ci(rows, key, seed=BOOT_SEED):
    return M.bootstrap_ci([r.get(key) for r in rows], seed=seed)


def _arm_block(sub):
    # Time-to-predator-loss CI: held-pred seeds enter as t_final.
    # right_censored only when at least one seed still holds predator bodies.
    loss_for_ci = []
    for r in sub:
        if r.get("pred_loss_t") is not None:
            loss_for_ci.append(r["pred_loss_t"])
        elif r.get("held_pred"):
            loss_for_ci.append(r["t_final"])
    n_held_pred = sum(1 for r in sub if r.get("held_pred"))
    n_held_prey = sum(1 for r in sub if r.get("held_prey"))
    n_held_bodies = sum(1 for r in sub if r.get("held_bodies"))
    loss_ci = M.bootstrap_ci(loss_for_ci, seed=BOOT_SEED)
    loss_ci["right_censored"] = n_held_pred > 0
    loss_ci["censor_at"] = float(MIN_T_FINAL)
    return dict(
        n=len(sub),
        kpred=sub[0]["kpred"] if sub else None,
        seeds=[r["seed"] for r in sub],
        status_bodies=[r.get("status_bodies") for r in sub],
        n_held_pred=n_held_pred,
        n_held_prey=n_held_prey,
        n_held_bodies=n_held_bodies,
        frac_held_pred=n_held_pred / len(sub) if sub else 0.0,
        frac_held_prey=n_held_prey / len(sub) if sub else 0.0,
        frac_held_bodies=n_held_bodies / len(sub) if sub else 0.0,
        pred_loss_t=[r.get("pred_loss_t") for r in sub],
        prey_loss_t=[r.get("prey_loss_t") for r in sub],
        body_loss_who=[r.get("body_loss_who") for r in sub],
        body_loss_t=[r.get("body_loss_t") for r in sub],
        coexist_bodies_t=[r.get("coexist_bodies_t") for r in sub],
        frac_bodies=[r.get("frac_bodies") for r in sub],
        mA_final=[r.get("mA_final") for r in sub],
        mP_final=[r.get("mP_final") for r in sub],
        nA_final=[r.get("nA_final") for r in sub],
        nP_final=[r.get("nP_final") for r in sub],
        period_A=[r.get("period_A") for r in sub],
        period_P=[r.get("period_P") for r in sub],
        period_A_acf=[r.get("period_A_acf") for r in sub],
        period_P_acf=[r.get("period_P_acf") for r in sub],
        pred_lag=[r.get("pred_lag") for r in sub],
        pred_lag_r=[r.get("pred_lag_r") for r in sub],
        ci=dict(
            coexist_bodies_t=_ci(sub, "coexist_bodies_t"),
            frac_bodies=_ci(sub, "frac_bodies"),
            time_to_pred_loss=loss_ci,
            mA_final=_ci(sub, "mA_final"),
            mP_final=_ci(sub, "mP_final"),
            nA_final=_ci(sub, "nA_final"),
            nP_final=_ci(sub, "nP_final"),
            nA_mean=_ci(sub, "nA_mean"),
            nP_mean=_ci(sub, "nP_mean"),
            period_A=_ci(sub, "period_A"),
            period_P=_ci(sub, "period_P"),
            period_A_acf=_ci(sub, "period_A_acf"),
            period_P_acf=_ci(sub, "period_P_acf"),
            pred_lag=_ci(sub, "pred_lag"),
            kill_mean=_ci(sub, "kill_mean"),
        ),
    )


def summarize_paths(paths, min_t=MIN_T_FINAL):
    rows = []
    skipped = []
    walls, gpus = [], set()
    # Only 19c job walls count toward this run's cost; 19b walls are comparison-only.
    for path in paths:
        job = load_job(path)
        is_19c = Path(path).name.startswith("b19c_")
        if is_19c and job.get("wall") is not None:
            walls.append(float(job["wall"]))
        if is_19c and job.get("gpu"):
            gpus.add(str(job["gpu"]))
        for tag, arm, seed, run, meta in iter_runs(job, include_partial=True, min_t=min_t):
            rec = dict(file=Path(path).name, tag=tag, arm=arm, seed=seed,
                       kpred=float((run.get("eco") or {}).get("kpred", float("nan"))),
                       gpu=meta.get("gpu"), wall=meta.get("wall"),
                       incomplete=meta.get("incomplete"), source="19c" if is_19c else "19b")
            if meta.get("incomplete"):
                skipped.append(rec)
                continue
            rec.update(M.summarize_run(run))
            rows.append(rec)
    order = {a: i for i, a in enumerate(ARMS)}
    by_arm = {}
    for arm in sorted({r["arm"] for r in rows}, key=lambda a: order.get(a, 99)):
        by_arm[arm] = _arm_block([r for r in rows if r["arm"] == arm])
    return dict(
        n_files=len(paths), n_runs=len(rows),
        n_skipped_incomplete=len(skipped),
        skipped_incomplete=skipped,
        walls_s=walls, gpu_hours=sum(walls) / 3600.0 if walls else 0.0,
        gpus=sorted(gpus),
        by_arm=by_arm,
        runs=rows,
    )


def main(argv):
    paths = [Path(p) for p in argv] if argv else []
    if not paths:
        raise SystemExit("analyze19c.py pickle [pickle ...]")
    summary = summarize_paths(paths)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(M.jsonable(summary), indent=1, allow_nan=False) + "\n")
    print("wrote", OUT)
    print("gpu_hours", round(summary["gpu_hours"], 3),
          "n_runs", summary["n_runs"],
          "skipped", summary["n_skipped_incomplete"])
    for arm, rec in summary["by_arm"].items():
        ci_t = rec["ci"]["coexist_bodies_t"]
        ci_loss = rec["ci"]["time_to_pred_loss"]
        cens = "censored" if ci_loss.get("right_censored") else "observed"
        print(f"  {arm} kpred={rec['kpred']} "
              f"held_pred={rec['n_held_pred']}/{rec['n']} "
              f"held_prey={rec['n_held_prey']}/{rec['n']} "
              f"held_both={rec['n_held_bodies']}/{rec['n']} "
              f"bodies={rec['status_bodies']} "
              f"coexist_t={_fmt_ci(ci_t)} pred_loss_t={_fmt_ci(ci_loss)} ({cens})")
    return summary


def _fmt_ci(c):
    if not c or c.get("mean") is None:
        return "na"
    if c.get("lo") is None:
        return f"{c['mean']:.0f}"
    return f"{c['mean']:.0f} [{c['lo']:.0f}, {c['hi']:.0f}]"


if __name__ == "__main__":
    main(sys.argv[1:])
