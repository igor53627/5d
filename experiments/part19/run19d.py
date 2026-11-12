"""Part 19d: replicate 19c kpred=0.005 on 8 new seeds, 20k, batch=1.

Writes archive/specs/b19d_full.json. Does not call Modal.
Same Part 15 world as the 19c k005 arm (4D L=48, coarse/fnoise/m=5,
8 prey + 4 predators, pred_pursuit_evolve default 1). One ecolist
container per seed, a single kpred=0.005 eco (batch=1, n_pred=4).
The 19c seeds 197–200 at kpred=0.005 are pooled in analyze19d, not rerun.
"""
import json
import sys
from pathlib import Path
import _path  # noqa: F401
import fl_pred as Z
import run19 as R19
import run19b as R19b
import run19c as R19c

ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = ROOT / "archive" / "specs"
REPORT = ROOT / "experiments" / "part19" / "results" / "b19d_init_report.json"
MODAL = "modal run src/modal_app.py::spawn_h100 --kind ecolist --spec {spec} --prefix {prefix}"

STEPS = R19c.STEPS
EVERY = R19c.EVERY
PROJ_EVERY = R19c.PROJ_EVERY
MIN_GAP = R19c.MIN_GAP
# Fresh seeds, disjoint from the 19 pilot (191–192), 19b (193–196), and 19c (197–200).
SEEDS = (201, 202, 203, 204, 205, 206, 207, 208)
PROBE_SEED = 201
N_PREY = R19.N_PREY
N_PRED = R19.N_PRED
KPRED = 0.005
ARM = "k005"
POOL_SEEDS = tuple(R19c.SEEDS)  # 19c k005 comparison, not launched here
EFF = R19.EFF
DEC_P = R19.DEC_P
KA = R19.KA
KP = R19.KP
KREP_X = R19.KREP_X


def _py(x):
    return R19._py(x)


def base_world():
    return R19.base_world()


def arm_eco(base):
    e = dict(base)
    e["kpred"] = KPRED
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
        payloads.append(dict(runs=[
            _run_kw(c, [arm_eco(eco)], seed, f"pp_s{seed}", N_PRED, 1, opts),
        ]))
    return payloads


def report():
    import numpy
    c, eco, opts = base_world()
    return dict(
        numpy=numpy.__version__, steps=STEPS, every=EVERY, seeds=list(SEEDS),
        probe_seed=PROBE_SEED, n_prey=N_PREY, n_pred=N_PRED, L=48, N=4, K=Z.KB,
        kpred=KPRED, arm=ARM, pool_seeds=list(POOL_SEEDS),
        eff=EFF, dec_p=DEC_P, ka=KA, kp=KP, krep_x=KREP_X,
        opts=opts, keat=eco["keat"], dec=eco["dec"], rF=eco["rF"], F0=eco["F0"],
        notes=[
            "Part 19d: replicate 19c kpred=0.005 on 8 new seeds (201–208), 20k, batch=1.",
            "Same Part 15 world as the 19c k005 arm (4D L=48, 8 prey, 4 predators, coarse/fnoise/m=5).",
            "pred_pursuit_evolve defaults to 1, as in Parts 19/19b/19c (not set in the eco dict).",
            "One container per seed, single eco (batch=1, n_pred=4). Seed 201 is the timing probe.",
            "19c seeds 197–200 at kpred=0.005 are pooled in analyze19d, not rerun.",
            "Hard cap $15 at $3.95/H100-h; remaining seeds launch only if the probe wall leaves room.",
        ],
    )


def write():
    payloads = build()
    path = SPEC_DIR / "b19d_full.json"
    path.write_text(json.dumps(_py(payloads)))
    print("wrote", path, "n=", len(payloads),
          "runs_per=", len(payloads[0]["runs"]),
          "ecos_per=", len(payloads[0]["runs"][0]["ecos"]))
    rep = _py(report())
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=1) + "\n")
    print("wrote", REPORT)
    return rep


def smoke():
    write()
    payloads = build()
    assert len(payloads) == len(SEEDS) == 8
    assert set(SEEDS).isdisjoint({191, 192, *R19b.SEEDS, *R19c.SEEDS})
    assert PROBE_SEED == SEEDS[0]
    c, eco, opts = base_world()
    assert opts.get("coarse") is True
    assert opts.get("fnoise") is True and opts.get("m") == 5
    eco_19c = R19c.arm_eco(eco, "k005")
    for item, seed in zip(payloads, SEEDS):
        assert len(item["runs"]) == 1
        pred = item["runs"][0]
        assert pred["tag"] == f"pp_s{seed}"
        assert pred["n_pred"] == N_PRED and pred["n_init"] == N_PREY
        assert pred["steps"] == STEPS and pred["seed"] == seed
        assert pred["L"] == 48 and pred["model"] == "pred" and pred["K"] == Z.KB
        assert pred["batch"] == 1 and len(pred["ecos"]) == 1
        assert pred["ecos"][0]["kpred"] == KPRED
        for k, v in eco_19c.items():
            assert pred["ecos"][0][k] == v, (k, pred["ecos"][0][k], v)
    print("full:", MODAL.format(spec="archive/specs/b19d_full.json", prefix="b19d"))
    print("smoke ok")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "write"
    if cmd == "write":
        write()
    elif cmd == "smoke":
        write(); smoke()
    else:
        raise SystemExit(cmd)
