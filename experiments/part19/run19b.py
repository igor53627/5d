"""Part 19b confirmatory predator–prey: low, mid, and a prey-only control.

Writes archive/specs/b19b_full.json. Does not call Modal.
Four new seeds × 20k steps on the Part 19 pilot world (4D L=48, Part 15 ecology).
n_pred is a run_eco-level flag, so each seed is one ecolist container with two
runs: low+mid batched (n_pred=4) then prey-only (n_pred=0, kpred=0).
"""
import json
import sys
from pathlib import Path
import _path  # noqa: F401
import fl_pred as Z
import run19 as R19

ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = ROOT / "archive" / "specs"
REPORT = ROOT / "experiments" / "part19" / "results" / "b19b_init_report.json"
MODAL = "modal run src/modal_app.py::spawn_h100 --kind ecolist --spec {spec} --prefix {prefix}"

STEPS = 20000
EVERY = R19.EVERY
PROJ_EVERY = R19.PROJ_EVERY
MIN_GAP = R19.MIN_GAP
# Fresh seeds, not the 19 pilot (191, 192).
SEEDS = (193, 194, 195, 196)
N_PREY = R19.N_PREY
N_PRED = R19.N_PRED
KPRED = dict(low=0.015, mid=0.05, preyonly=0.0)
PRED_ARMS = ("low", "mid")
ARMS = ("low", "mid", "preyonly")
EFF = R19.EFF
DEC_P = R19.DEC_P
KA = R19.KA
KP = R19.KP
KREP_X = R19.KREP_X


def _py(x):
    return R19._py(x)


def base_world():
    return R19.base_world()


def arm_eco(base, arm):
    e = dict(base)
    e["kpred"] = KPRED[arm]
    return e


def _run_kw(c, ecos, seed, tag, n_pred, batch, opts):
    return dict(
        c=c, ecos=ecos, seed=int(seed), tag=tag, N=4, L=48,
        steps=STEPS, every=EVERY, proj_every=PROJ_EVERY, init="multi",
        n_init=N_PREY, n_pred=int(n_pred), p_spread=0.5, p_spread_pred=0.5,
        min_sep=1, min_gap=MIN_GAP, batch=int(batch),
        model="pred", opts=dict(opts), K=Z.KB,
        p0=[0.0] * Z.KB, p0_pred=[0.0] * Z.KB,
    )


def build():
    c, eco, opts = base_world()
    payloads = []
    for seed in SEEDS:
        pred_runs = [_run_kw(c, [arm_eco(eco, a) for a in PRED_ARMS],
                             seed, f"pp_s{seed}", N_PRED, 2, opts)]
        only = [_run_kw(c, [arm_eco(eco, "preyonly")],
                        seed, f"preyonly_s{seed}", 0, 1, opts)]
        payloads.append(dict(runs=pred_runs + only))
    return payloads


def report():
    import numpy
    c, eco, opts = base_world()
    return dict(
        numpy=numpy.__version__, steps=STEPS, every=EVERY, seeds=list(SEEDS),
        n_prey=N_PREY, n_pred=N_PRED, L=48, N=4, K=Z.KB,
        kpred=dict(KPRED), eff=EFF, dec_p=DEC_P, ka=KA, kp=KP, krep_x=KREP_X,
        arms=list(ARMS), opts=opts,
        keat=eco["keat"], dec=eco["dec"], rF=eco["rF"], F0=eco["F0"],
        notes=[
            "Part 19b confirmatory: low (0.015) and mid (0.05) vs prey-only, 4 new seeds, 20k.",
            "Same Part 15 world as the 19 pilot (4D L=48, 8 prey, 4 predators on pred arms).",
            "Primary readout is body-level coexistence (n>=1 body of mass >=0.15 M1).",
            "prey-only is kpred=0 and n_pred=0, a separate run_eco in the same container.",
            f"Budget target ~$15–20 (hard cap $28) for {len(SEEDS)} H100 ecolist containers × {STEPS} steps.",
        ],
    )


def write():
    payloads = build()
    path = SPEC_DIR / "b19b_full.json"
    path.write_text(json.dumps(_py(payloads)))
    print("wrote", path, "n=", len(payloads),
          "runs_per=", len(payloads[0]["runs"]))
    rep = _py(report())
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=1) + "\n")
    print("wrote", REPORT)
    return rep


def smoke():
    write()
    payloads = build()
    assert len(payloads) == len(SEEDS)
    assert set(SEEDS).isdisjoint({191, 192})
    for item, seed in zip(payloads, SEEDS):
        assert len(item["runs"]) == 2
        pred, only = item["runs"]
        assert pred["tag"] == f"pp_s{seed}"
        assert only["tag"] == f"preyonly_s{seed}"
        assert pred["n_pred"] == N_PRED and only["n_pred"] == 0
        assert pred["n_init"] == N_PREY and only["n_init"] == N_PREY
        assert pred["steps"] == STEPS and only["steps"] == STEPS
        assert pred["seed"] == only["seed"] == seed
        assert pred["L"] == 48 and pred["model"] == "pred" and pred["K"] == Z.KB
        assert [e["kpred"] for e in pred["ecos"]] == [KPRED["low"], KPRED["mid"]]
        assert only["ecos"][0]["kpred"] == 0.0
        assert only["batch"] == 1 and pred["batch"] == 2
        # Same world knobs as the 19 pilot (except kpred / n_pred).
        for e in pred["ecos"] + only["ecos"]:
            assert e["ka"] == KA and e["kp"] == KP
            assert e["eff"] == EFF and e["dec_p"] == DEC_P
            assert e["krep_x"] == KREP_X
            assert e["keat"] == 0.01 and e["dec"] == 0.0003
    print("full:", MODAL.format(spec="archive/specs/b19b_full.json", prefix="b19b"))
    print("smoke ok")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "write"
    if cmd == "write":
        write()
    elif cmd == "smoke":
        write(); smoke()
    else:
        raise SystemExit(cmd)
