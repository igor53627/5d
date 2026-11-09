"""Part 18d specs: seasonal assay with mutable / noisy output biases g_A, g_B.

Writes archive/specs/b18d_pilot.json and b18d_full.json. Does not call Modal.
Pilot and full commands are printed by `python experiments/part18/run18d.py smoke`.
"""
import json
import sys
from pathlib import Path
import numpy as np
import _path  # noqa: F401
import fl_season as Z
import init18c as I18c
import init18d as I
import metrics18c as M
import run18c as R18c

ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = ROOT / "archive" / "specs"
REPORT = ROOT / "experiments" / "part18" / "results" / "b18d_init_report.json"
TS = R18c.TS
EVERY = R18c.EVERY
PROJ_EVERY = R18c.PROJ_EVERY
MIN_GAP = R18c.MIN_GAP
KREP = R18c.KREP_18C
MODAL = R18c.MODAL
STEPS = dict(pilot=12000, full=20000)
SEEDS = dict(pilot=(841,), full=(841, 842, 843, 844))
# Elevated bias mutation for bias_* arms only. Default mut_bias=-1 keeps mutw (18c path).
MUT_BIAS = 0.02
ARM_FLAGS = {
    "bias_cue": dict(mem=1.0, cue=1.0, kb=1.0, kr=10.0, mut_bias=MUT_BIAS),
    "bias_nocue": dict(mem=1.0, cue=0.0, kb=1.0, kr=10.0, mut_bias=MUT_BIAS),
    "inputs_cue": dict(mem=1.0, cue=1.0, kb=1.0, kr=10.0),  # mut_bias unset → BDEF -1
    "planted": dict(mem=1.0, cue=1.0, kb=1.0, kr=10.0),
    "nobrain": dict(mem=0.0, cue=0.0, kb=0.0, kr=0.0),
}


def _py(x):
    return R18c._py(x)


def base_world():
    return R18c.base_world()


def sign_changes(steps, every=EVERY, Ts=TS):
    return R18c.sign_changes(steps, every=every, Ts=Ts)


def populations():
    founders = I18c.load_founders()
    pops = {}
    for seed, founder_seed in I.SEED_OF_FOUNDER.items():
        p0 = founders[founder_seed]
        pops[seed] = {}
        for arm in ("bias_cue", "inputs_cue", "planted", "nobrain"):
            # nobrain uses lo stream inside population()
            rows, info = I.population(p0, seed, arm)
            pops[seed][arm] = (rows, info, p0)
        pops[seed]["bias_nocue"] = pops[seed]["bias_cue"]
    return pops


def build(kind, pops=None):
    c, eco, opts = base_world()
    pops = pops or populations()
    payloads = []
    for seed in SEEDS[kind]:
        for arm in I.ARMS:
            rows, _, p0 = pops[seed][arm]
            e = dict(eco)
            e.update(ARM_FLAGS[arm])
            payloads.append(dict(runs=[dict(
                c=c, ecos=[e], seed=int(seed), tag=f"{arm}_s{seed}", N=4, L=48,
                steps=STEPS[kind], every=EVERY, proj_every=PROJ_EVERY, init="multi",
                n_init=I18c.N_INIT, p_spread=0.0, min_sep=1, min_gap=MIN_GAP, batch=1,
                model="season", opts=dict(opts), K=Z.KB, p0=_py(p0), p_inits=_py(rows),
            )]))
    return payloads


def report(pops):
    import numpy
    arms = {}
    for seed, founder_seed in I.SEED_OF_FOUNDER.items():
        p0 = pops[seed]["bias_cue"][2]
        arms[str(seed)] = dict(founder_seed=founder_seed, arms={})
        for arm in ("bias_cue", "inputs_cue", "planted", "nobrain"):
            rows, info, _ = pops[seed][arm]
            rec = I.genome_report(rows, p0)
            if arm == "planted":
                rec["plant"] = [None if x is None else dict(rms=x["rms"], hits=x["hits"], trials=x["trials"]) for x in info]
            arms[str(seed)]["arms"][arm] = rec
        arms[str(seed)]["arms"]["bias_nocue"] = "same_genomes_as_bias_cue"
    n_pilot, _ = sign_changes(STEPS["pilot"])
    n_full, _ = sign_changes(STEPS["full"])
    return dict(
        numpy=numpy.__version__, Ts=TS, every=EVERY, min_gap=MIN_GAP,
        steps=STEPS, sign_changes=dict(pilot=n_pilot, full=n_full),
        taxis_repulsion=M.taxis_repulsion_ratio(krep=KREP),
        std_bias=I.STD_BIAS, mut_bias=MUT_BIAS, seeds=arms,
        shared_genomes={"bias_nocue": "bias_cue"},
        notes=[
            "18c already mutated all NW weights (incl. g_A/g_B) at mutw=0.004; biases were only frozen at init.",
            f"bias_* arms: new-channel N(0,0.5) + bias N(0,{I.STD_BIAS}) at init; mut_bias={MUT_BIAS} during run.",
            "inputs_cue: 18c cue baseline (new channels only; mut_bias default -1 → mutw).",
            "planted / nobrain: same positive / floor controls as 18c.",
            f"krep={KREP} (18c fusion fix). Seeds {list(I.SEED_OF_FOUNDER)} inherit founders 621–624.",
        ],
    )


def write():
    pops = populations()
    for kind in ("pilot", "full"):
        path = SPEC_DIR / f"b18d_{kind}.json"
        path.write_text(json.dumps(_py(build(kind, pops))))
        print("wrote", path)
    rep = _py(report(pops))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=1) + "\n")
    print("wrote", REPORT)
    return rep


def smoke():
    pops = populations()
    rep = report(pops)
    if rep["sign_changes"]["full"] < 4:
        raise SystemExit(f"full run has {rep['sign_changes']['full']} sign changes, need >= 4")
    ratio = M.taxis_repulsion_ratio(krep=KREP)
    print(f"sign changes pilot {rep['sign_changes']['pilot']}  full {rep['sign_changes']['full']}")
    print(f"taxis/repulsion at krep={KREP:g}: {ratio['ratio']:.3f}")
    print(f"std_bias={I.STD_BIAS}  mut_bias={MUT_BIAS}")
    for seed, rec in rep["seeds"].items():
        bc, ic, pl = rec["arms"]["bias_cue"], rec["arms"]["inputs_cue"], rec["arms"]["planted"]
        print(f"  seed {seed} founder {rec['founder_seed']}: bias_cue switch {bc['n_switch']}/{bc['n']} "
              f"Δbias {bc.get('std_delta_bias', 0):.3f}; inputs_cue switch {ic['n_switch']}/{ic['n']} "
              f"Δbias {ic.get('max_abs_delta_bias', 0):.3g}; planted switch {pl['n_switch']}/{pl['n']}")
        if ic["max_abs_delta_bias"] != 0.0:
            raise SystemExit("inputs_cue must not touch g_A/g_B at init")
        if bc["std_delta_bias"] < 0.15:
            raise SystemExit("bias_cue standing bias variation too small")
        if pl["n_switch"] != I18c.N_PLANTED:
            raise SystemExit("planted arm does not have 6 switchers")
        # Shared genomes
        assert np.allclose(pops[int(seed)]["bias_cue"][0], pops[int(seed)]["bias_nocue"][0])
    # mut_bias default leaves amp = mutw
    assert Z.BDEF["mut_bias"] < 0.0
    R18c.smoke_step()
    print("pilot:", MODAL.format(spec="archive/specs/b18d_pilot.json", prefix="b18dp"))
    print("full: ", MODAL.format(spec="archive/specs/b18d_full.json", prefix="b18d"))
    print("smoke ok")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "write"
    if cmd == "write":
        write()
    elif cmd == "smoke":
        write()
        smoke()
    else:
        raise SystemExit(f"unknown command {cmd}")
