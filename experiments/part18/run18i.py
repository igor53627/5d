"""Part 18i: more seeds for the 18g confirmatory assay.

Same world/arms as 18g (bias_cue, bias_nocue, planted; mut_bias=0.05; 20k).
New seeds 855–858 (founders 621–624) bring the pool with 18g (851–854) to 8.

Writes archive/specs/b18i_full.json. Does not call Modal.
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
import run18g as R18g

ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = ROOT / "archive" / "specs"
REPORT = ROOT / "experiments" / "part18" / "results" / "b18i_init_report.json"
TS = R18c.TS
EVERY = R18c.EVERY
PROJ_EVERY = R18c.PROJ_EVERY
MIN_GAP = R18c.MIN_GAP
KREP = R18c.KREP_18C
MODAL = R18c.MODAL
STEPS = R18g.STEPS
# Continue after 18g's 851–854; same founder map, fresh placement/genome draws.
SEED_OF_FOUNDER = {855: 621, 856: 622, 857: 623, 858: 624}
SEEDS = tuple(SEED_OF_FOUNDER)
MUT_BIAS = R18g.MUT_BIAS
HOM = R18g.HOM
DEC = R18g.DEC
ARMS = R18g.ARMS
GENOME_ARM = R18g.GENOME_ARM


def _py(x):
    return R18c._py(x)


def populations():
    founders = I18c.load_founders()
    pops = {}
    for seed, founder_seed in SEED_OF_FOUNDER.items():
        p0 = founders[founder_seed]
        pops[seed] = {}
        for garm in ("bias_cue", "planted"):
            rows, info = I.population(p0, seed, garm)
            pops[seed][garm] = (rows, info, p0)
    return pops


def build(pops=None):
    c, eco, opts = R18c.base_world()
    pops = pops or populations()
    payloads = []
    for seed in SEEDS:
        for arm, flags in ARMS.items():
            garm = GENOME_ARM[arm]
            rows, _, p0 = pops[seed][garm]
            e = dict(eco)
            e.update(flags)
            payloads.append(dict(runs=[dict(
                c=c, ecos=[e], seed=int(seed), tag=f"{arm}_s{seed}", N=4, L=48,
                steps=STEPS, every=EVERY, proj_every=PROJ_EVERY, init="multi",
                n_init=I18c.N_INIT, p_spread=0.0, min_sep=1, min_gap=MIN_GAP, batch=1,
                model="season", opts=dict(opts), K=Z.KB, p0=_py(p0), p_inits=_py(rows),
            )]))
    return payloads


def report(pops):
    import numpy
    n_sw, frames = R18c.sign_changes(STEPS, every=EVERY, Ts=TS)
    return dict(
        numpy=numpy.__version__, Ts=TS, every=EVERY, steps=STEPS, seeds=list(SEEDS),
        seed_of_founder=dict(SEED_OF_FOUNDER), pool_with="18g 851-854 (8 seeds total)",
        mut_bias=MUT_BIAS, hom=HOM, sign_changes=n_sw, sign_change_frames=frames,
        taxis_repulsion=M.taxis_repulsion_ratio(krep=KREP), arms=list(ARMS),
        notes=[
            "18i: four new seeds, same config as 18g. Pool with 18g → 8 seeds/arm.",
            f"Budget target ~$15–20 for {len(SEEDS)*len(ARMS)}×{STEPS} H100 ecolist runs.",
        ],
    )


def write():
    pops = populations()
    path = SPEC_DIR / "b18i_full.json"
    payloads = build(pops)
    path.write_text(json.dumps(_py(payloads)))
    print("wrote", path, "n=", len(payloads))
    rep = _py(report(pops))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=1) + "\n")
    print("wrote", REPORT)
    return rep


def smoke():
    pops = populations()
    rep = report(pops)
    if rep["sign_changes"] < 4:
        raise SystemExit(f"{rep['sign_changes']} sign changes, need >= 4")
    payloads = build(pops)
    by = {r["runs"][0]["tag"]: r["runs"][0] for r in payloads}
    assert by["bias_cue_s855"]["ecos"][0]["cue"] == 1.0
    assert by["bias_nocue_s855"]["ecos"][0]["cue"] == 0.0
    assert np.allclose(by["bias_cue_s855"]["p_inits"], by["bias_nocue_s855"]["p_inits"])
    assert set(SEEDS).isdisjoint({731, 732, 733, 734, 841, 842, 843, 844, 851, 852, 853, 854, 861, 862})
    assert all(r["runs"][0]["ecos"][0]["krep"] == KREP for r in payloads)
    R18c.smoke_step()
    print(f"sign changes in {STEPS} steps: {rep['sign_changes']}")
    print("full:", MODAL.format(spec="archive/specs/b18i_full.json", prefix="b18i"))
    print("smoke ok")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "write"
    if cmd == "write":
        write()
    elif cmd == "smoke":
        write(); smoke()
    else:
        raise SystemExit(cmd)
