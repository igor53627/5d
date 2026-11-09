"""Part 18f pilot: longer horizon + mut_bias/hom variants + planted cue_lead sanity.

Follows 18e readout: drop lowdec; keep ≥2 season switches (A→B and B→A);
try higher mut_bias and weaker homogenization; sanity-check cue_lead on planted.

Writes archive/specs/b18f_pilot.json. Does not call Modal.
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
REPORT = ROOT / "experiments" / "part18" / "results" / "b18f_init_report.json"
TS = R18c.TS
EVERY = R18c.EVERY
PROJ_EVERY = R18c.PROJ_EVERY
MIN_GAP = R18c.MIN_GAP
KREP = R18c.KREP_18C
MODAL = R18c.MODAL
STEPS = 12000  # 2 sign changes at Ts=8000 (A→B ~4.2k, B→A ~8.2k)
SEEDS = (841, 842)
MUT_BIAS = 0.05
MUT_BIAS_HI = 0.10
HOM_BASE = 0.05
HOM_LOW = 0.0125  # 4× weaker homogenization
CUE_LEAD = 400.0
DEC = 1e-4

ARMS = {
    "bias_cue": dict(mem=1.0, cue=1.0, kb=1.0, kr=10.0, mut_bias=MUT_BIAS, hom=HOM_BASE, cue_lead=0.0, dec=DEC),
    "bias_cue_mb10": dict(mem=1.0, cue=1.0, kb=1.0, kr=10.0, mut_bias=MUT_BIAS_HI, hom=HOM_BASE, cue_lead=0.0, dec=DEC),
    "bias_cue_lowhom": dict(mem=1.0, cue=1.0, kb=1.0, kr=10.0, mut_bias=MUT_BIAS, hom=HOM_LOW, cue_lead=0.0, dec=DEC),
    "planted": dict(mem=1.0, cue=1.0, kb=1.0, kr=10.0, hom=HOM_BASE, cue_lead=0.0, dec=DEC),
    "planted_lead": dict(mem=1.0, cue=1.0, kb=1.0, kr=10.0, hom=HOM_BASE, cue_lead=CUE_LEAD, dec=DEC),
}
GENOME_ARM = {
    "bias_cue": "bias_cue",
    "bias_cue_mb10": "bias_cue",
    "bias_cue_lowhom": "bias_cue",
    "planted": "planted",
    "planted_lead": "planted",
}


def _py(x):
    return R18c._py(x)


def base_world():
    return R18c.base_world()


def populations():
    founders = I18c.load_founders()
    pops = {}
    for seed, founder_seed in I.SEED_OF_FOUNDER.items():
        if seed not in SEEDS:
            continue
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
    n_sw, _ = R18c.sign_changes(STEPS, every=EVERY, Ts=TS)
    return dict(
        numpy=numpy.__version__, Ts=TS, every=EVERY, steps=STEPS, seeds=list(SEEDS),
        mut_bias=MUT_BIAS, mut_bias_hi=MUT_BIAS_HI, hom_base=HOM_BASE, hom_low=HOM_LOW,
        cue_lead=CUE_LEAD, sign_changes=n_sw,
        taxis_repulsion=M.taxis_repulsion_ratio(krep=KREP),
        arms=list(ARMS),
        notes=[
            "18f pilot: ≥2 season switches; raise within_aligned for bias_cue.",
            f"Variants: mut_bias={MUT_BIAS_HI}, hom={HOM_LOW}; planted_lead cue_lead={CUE_LEAD} sanity.",
            "Dropped lowdec (18e: hurt within_aligned).",
            f"Budget target ~$5–8 for {len(SEEDS)*len(ARMS)}×{STEPS} H100 ecolist runs.",
        ],
    )


def write():
    pops = populations()
    path = SPEC_DIR / "b18f_pilot.json"
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
    if rep["sign_changes"] < 2:
        raise SystemExit(f"pilot has {rep['sign_changes']} sign changes, need >= 2")
    payloads = build(pops)
    by_tag = {r["runs"][0]["tag"]: r["runs"][0] for r in payloads}
    assert by_tag["bias_cue_s841"]["ecos"][0]["mut_bias"] == MUT_BIAS
    assert by_tag["bias_cue_mb10_s841"]["ecos"][0]["mut_bias"] == MUT_BIAS_HI
    assert by_tag["bias_cue_lowhom_s841"]["ecos"][0]["hom"] == HOM_LOW
    assert by_tag["planted_lead_s841"]["ecos"][0]["cue_lead"] == CUE_LEAD
    assert by_tag["planted_s841"]["ecos"][0].get("cue_lead", 0.0) == 0.0
    assert np.allclose(by_tag["bias_cue_s841"]["p_inits"], by_tag["bias_cue_mb10_s841"]["p_inits"])
    assert np.allclose(by_tag["planted_s841"]["p_inits"], by_tag["planted_lead_s841"]["p_inits"])
    assert all(r["runs"][0]["ecos"][0]["krep"] == KREP for r in payloads)
    R18c.smoke_step()
    print(f"sign changes in {STEPS} steps: {rep['sign_changes']}")
    print("pilot:", MODAL.format(spec="archive/specs/b18f_pilot.json", prefix="b18fp"))
    print("smoke ok")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "write"
    if cmd == "write":
        write()
    elif cmd == "smoke":
        write(); smoke()
    else:
        raise SystemExit(cmd)
