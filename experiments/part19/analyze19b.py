"""Summarize Part 19b confirmatory logs: body coexistence, periods, lags, CIs."""
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import _path  # noqa: F401
import metrics19 as M
import run19b as R

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "part19" / "results" / "full19b_summary.json"

ARMS = list(R.ARMS)
KPRED = dict(R.KPRED)
BOOT_SEED = 19


def load_job(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def iter_runs(job):
    out = job.get("out")
    payload = out[0] if isinstance(out, list) and out else out
    if not isinstance(payload, dict):
        return
    meta = dict(gpu=job.get("gpu"), wall=job.get("wall"), kind=job.get("kind"))
    for tag, res in payload.items():
        if tag == "summary" or not isinstance(res, dict) or "runs" not in res:
            continue
        n_pred = int(res.get("n_pred") or 0)
        for run in res["runs"]:
            eco = run.get("eco") or {}
            kpred = float(eco.get("kpred", float("nan")))
            arm = None
            if n_pred == 0 or abs(kpred) < 1e-15:
                arm = "preyonly"
            else:
                for name, val in KPRED.items():
                    if name == "preyonly":
                        continue
                    if abs(kpred - val) < 1e-12:
                        arm = name
                        break
            if arm is None:
                arm = f"kpred_{kpred}"
            seed = None
            for part in tag.split("_"):
                if part.startswith("s") and part[1:].isdigit():
                    seed = int(part[1:])
            yield tag, arm, seed, run, meta


def _ci(rows, key, seed=BOOT_SEED):
    return M.bootstrap_ci([r.get(key) for r in rows], seed=seed)


def _arm_block(sub):
    # For time-to-loss CI, treat "held to the end" as t_final (right-censored at 20k).
    loss_for_ci = []
    for r in sub:
        if r.get("body_loss_t") is not None:
            loss_for_ci.append(r["body_loss_t"])
        elif r.get("held_bodies"):
            loss_for_ci.append(r["t_final"])
    n_held = sum(1 for r in sub if r.get("held_bodies"))
    return dict(
        n=len(sub),
        kpred=sub[0]["kpred"] if sub else None,
        seeds=[r["seed"] for r in sub],
        status_bodies=[r.get("status_bodies") for r in sub],
        n_held_bodies=n_held,
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
            time_to_loss=M.bootstrap_ci(loss_for_ci, seed=BOOT_SEED),
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


def summarize_paths(paths):
    rows = []
    walls, gpus = [], set()
    for path in paths:
        job = load_job(path)
        if job.get("wall") is not None:
            walls.append(float(job["wall"]))
        if job.get("gpu"):
            gpus.add(str(job["gpu"]))
        for tag, arm, seed, run, meta in iter_runs(job):
            rec = M.summarize_run(run)
            rec.update(file=Path(path).name, tag=tag, arm=arm, seed=seed,
                       kpred=float((run.get("eco") or {}).get("kpred", float("nan"))),
                       gpu=meta.get("gpu"), wall=meta.get("wall"))
            rows.append(rec)
    order = {a: i for i, a in enumerate(ARMS)}
    by_arm = {}
    for arm in sorted({r["arm"] for r in rows}, key=lambda a: order.get(a, 99)):
        by_arm[arm] = _arm_block([r for r in rows if r["arm"] == arm])
    # Held-body window among pred arms
    pred_rows = [r for r in rows if r["arm"] in ("low", "mid")]
    n_held = sum(1 for r in pred_rows if r.get("held_bodies"))
    return dict(
        n_files=len(paths), n_runs=len(rows),
        walls_s=walls, gpu_hours=sum(walls) / 3600.0 if walls else 0.0,
        gpus=sorted(gpus),
        by_arm=by_arm,
        runs=rows,
        n_pred_arm_held=n_held,
        n_pred_arm=len(pred_rows),
    )


def py(x):
    if isinstance(x, dict):
        return {k: py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [py(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


def main(argv):
    paths = [Path(p) for p in argv] if argv else []
    if not paths:
        raise SystemExit("analyze19b.py pickle [pickle ...]")
    summary = summarize_paths(paths)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(py(summary), indent=1) + "\n")
    print("wrote", OUT)
    print("gpu_hours", round(summary["gpu_hours"], 3),
          "held_bodies", summary["n_pred_arm_held"], "/", summary["n_pred_arm"])
    for arm, rec in summary["by_arm"].items():
        ci_t = rec["ci"]["coexist_bodies_t"]
        ci_loss = rec["ci"]["time_to_loss"]
        print(f"  {arm} kpred={rec['kpred']} held={rec['n_held_bodies']}/{rec['n']} "
              f"bodies={rec['status_bodies']} "
              f"coexist_t={_fmt_ci(ci_t)} loss_t={_fmt_ci(ci_loss)}")
    return summary


def _fmt_ci(c):
    if not c or c.get("mean") is None:
        return "na"
    if c.get("lo") is None:
        return f"{c['mean']:.0f}"
    return f"{c['mean']:.0f} [{c['lo']:.0f}, {c['hi']:.0f}]"


if __name__ == "__main__":
    main(sys.argv[1:])
