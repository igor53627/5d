"""Part 19c low-side kpred scan: 0.005 and 0.01, 4 new seeds, 20k.

Writes archive/specs/b19c_full.json. Does not call Modal.
Same Part 15 world as the 19b low arm (4D L=48, coarse/fnoise/m=5,
8 prey + 4 predators). One ecolist container per seed with both kpred
values batched (batch=2, n_pred=4), matching the 19b pred-batch packing.
The 19b kpred=0.015 seeds are a comparison arm in analyze19c, not rerun.
"""
import json
import sys
from pathlib import Path
import _path  # noqa: F401
import fl_pred as Z
import run19 as R19
import run19b as R19b

ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = ROOT / "archive" / "specs"
REPORT = ROOT / "experiments" / "part19" / "results" / "b19c_init_report.json"
MODAL = "modal run src/modal_app.py::spawn_h100 --kind ecolist --spec {spec} --prefix {prefix}"

STEPS = R19b.STEPS
EVERY = R19b.EVERY
PROJ_EVERY = R19b.PROJ_EVERY
MIN_GAP = R19b.MIN_GAP
# Fresh seeds, disjoint from the 19 pilot (191, 192) and 19b (193–196).
SEEDS = (197, 198, 199, 200)
N_PREY = R19.N_PREY
N_PRED = R19.N_PRED
KPRED = dict(k005=0.005, k010=0.01)
ARMS = ("k005", "k010")
# Comparison arm in analyze19c only (19b low, not launched here).
COMPARE_ARM = "k015"
COMPARE_KPRED = 0.015
COMPARE_SEEDS = tuple(R19b.SEEDS)
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
        ecos = [arm_eco(eco, a) for a in ARMS]
        payloads.append(dict(runs=[
            _run_kw(c, ecos, seed, f"pp_s{seed}", N_PRED, 2, opts),
        ]))
    return payloads


def report():
    import numpy
    c, eco, opts = base_world()
    return dict(
        numpy=numpy.__version__, steps=STEPS, every=EVERY, seeds=list(SEEDS),
        n_prey=N_PREY, n_pred=N_PRED, L=48, N=4, K=Z.KB,
        kpred=dict(KPRED), compare_kpred=COMPARE_KPRED,
        compare_seeds=list(COMPARE_SEEDS),
        eff=EFF, dec_p=DEC_P, ka=KA, kp=KP, krep_x=KREP_X,
        arms=list(ARMS), opts=opts,
        keat=eco["keat"], dec=eco["dec"], rF=eco["rF"], F0=eco["F0"],
        notes=[
            "Part 19c low-side scan: kpred 0.005 and 0.01, 4 new seeds, 20k.",
            "Same Part 15 world as the 19b low arm (4D L=48, 8 prey, 4 predators, coarse/fnoise/m=5).",
            "pred_pursuit_evolve defaults to 1, as in Parts 19/19b (not set in the eco dict).",
            "One container per seed, both kpred values batched (batch=2, n_pred=4).",
            "19b seeds 193–196 at kpred=0.015 are the comparison arm in analyze19c, not rerun.",
            f"Budget target ~$14.10 (hard cap ~$16) for {len(SEEDS)} H100 ecolist containers × {STEPS} steps.",
        ],
    )


def write():
    payloads = build()
    path = SPEC_DIR / "b19c_full.json"
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
    assert len(payloads) == len(SEEDS)
    assert set(SEEDS).isdisjoint({191, 192, *R19b.SEEDS})
    c, eco, opts = base_world()
    assert opts.get("coarse") is True
    assert opts.get("fnoise") is True and opts.get("m") == 5
    low_19b = R19b.arm_eco(eco, "low")
    for item, seed in zip(payloads, SEEDS):
        assert len(item["runs"]) == 1
        pred = item["runs"][0]
        assert pred["tag"] == f"pp_s{seed}"
        assert pred["n_pred"] == N_PRED and pred["n_init"] == N_PREY
        assert pred["steps"] == STEPS and pred["seed"] == seed
        assert pred["L"] == 48 and pred["model"] == "pred" and pred["K"] == Z.KB
        assert pred["batch"] == 2 and len(pred["ecos"]) == 2
        assert [e["kpred"] for e in pred["ecos"]] == [KPRED[a] for a in ARMS]
        for e in pred["ecos"]:
            assert e["ka"] == KA and e["kp"] == KP
            assert e["eff"] == EFF and e["dec_p"] == DEC_P
            assert e["krep_x"] == KREP_X
            assert e["keat"] == 0.01 and e["dec"] == 0.0003
            # Same world as 19b low except kpred.
            for k, v in low_19b.items():
                if k == "kpred":
                    continue
                assert e[k] == v, (k, e[k], v)
    print("full:", MODAL.format(spec="archive/specs/b19c_full.json", prefix="b19c"))
    print("smoke ok")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "write"
    if cmd == "write":
        write()
    elif cmd == "smoke":
        write(); smoke()
    else:
        raise SystemExit(cmd)
