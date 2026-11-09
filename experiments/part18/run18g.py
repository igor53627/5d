"""Part 18g confirmatory full: within_aligned for bias_cue vs bias_nocue vs planted.

Extends the 18f longer-horizon finding (bias_cue within_al≈planted) to 4 new seeds
and a nocue contrast. No new knobs. mut_bias=0.05 on bias_* only.

Writes archive/specs/b18g_full.json. Does not call Modal.
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
REPORT = ROOT / "experiments" / "part18" / "results" / "b18g_init_report.json"
TS = R18c.TS
EVERY = R18c.EVERY
PROJ_EVERY = R18c.PROJ_EVERY
MIN_GAP = R18c.MIN_GAP
KREP = R18c.KREP_18C
MODAL = R18c.MODAL
STEPS = 20000  # 4 switches at Ts=8000: A→B, B→A, A→B, B→A
# Fresh seeds (not 731–734 / 841–844); same founders 621–624.
SEED_OF_FOUNDER = {851: 621, 852: 622, 853: 623, 854: 624}
SEEDS = tuple(SEED_OF_FOUNDER)
MUT_BIAS = 0.05
HOM = 0.05
DEC = 1e-4

ARMS = {
    "bias_cue": dict(mem=1.0, cue=1.0, kb=1.0, kr=10.0, mut_bias=MUT_BIAS, hom=HOM, cue_lead=0.0, dec=DEC),
    "bias_nocue": dict(mem=1.0, cue=0.0, kb=1.0, kr=10.0, mut_bias=MUT_BIAS, hom=HOM, cue_lead=0.0, dec=DEC),
    "planted": dict(mem=1.0, cue=1.0, kb=1.0, kr=10.0, hom=HOM, cue_lead=0.0, dec=DEC),
}
GENOME_ARM = {
    "bias_cue": "bias_cue",
    "bias_nocue": "bias_cue",  # shared genomes with bias_cue
    "planted": "planted",
}


def _py(x):
    return R18c._py(x)


def base_world():
    return R18c.base_world()


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
    c, eco, opts = base_world()
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
        seed_of_founder=dict(SEED_OF_FOUNDER), mut_bias=MUT_BIAS, hom=HOM,
        sign_changes=n_sw, sign_change_frames=frames,
        taxis_repulsion=M.taxis_repulsion_ratio(krep=KREP),
        arms=list(ARMS),
        notes=[
            "18g confirmatory full: within_aligned cue vs nocue vs planted, 4 new seeds, 20k.",
            "No new knobs. bias_nocue shares genomes with bias_cue (cue=0 only).",
            f"Budget target ~$15–20 for {len(SEEDS)*len(ARMS)}×{STEPS} H100 ecolist runs.",
        ],
    )


def write():
    pops = populations()
    path = SPEC_DIR / "b18g_full.json"
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
        raise SystemExit(f"full has {rep['sign_changes']} sign changes, need >= 4")
    payloads = build(pops)
    by_tag = {r["runs"][0]["tag"]: r["runs"][0] for r in payloads}
    assert by_tag["bias_cue_s851"]["ecos"][0]["cue"] == 1.0
    assert by_tag["bias_nocue_s851"]["ecos"][0]["cue"] == 0.0
    assert by_tag["bias_cue_s851"]["ecos"][0]["mut_bias"] == MUT_BIAS
    assert by_tag["bias_nocue_s851"]["ecos"][0]["mut_bias"] == MUT_BIAS
    assert "mut_bias" not in by_tag["planted_s851"]["ecos"][0] or \
           by_tag["planted_s851"]["ecos"][0].get("mut_bias", -1) < 0
    assert np.allclose(by_tag["bias_cue_s851"]["p_inits"], by_tag["bias_nocue_s851"]["p_inits"])
    assert all(r["runs"][0]["ecos"][0]["krep"] == KREP for r in payloads)
    assert set(SEEDS).isdisjoint({731, 732, 733, 734, 841, 842, 843, 844})
    R18c.smoke_step()
    print(f"sign changes in {STEPS} steps: {rep['sign_changes']}")
    print("full:", MODAL.format(spec="archive/specs/b18g_full.json", prefix="b18g"))
    print("smoke ok")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "write"
    if cmd == "write":
        write()
    elif cmd == "smoke":
        write(); smoke()
    else:
        raise SystemExit(cmd)
