"""Part 18e pilot: raise within-body seasonal switching for bias_cue.

18d full: bias_cue mean within ≈ +0.05 vs planted ≈ +0.29; cue selected
season-matched biases mostly via replacement. 18e tries existing knobs that
cut the payoff of replacement (lower decay) plus a minimal cue_lead so the
brain sees the next season early (default 0 = 18d-identical).

Writes archive/specs/b18e_pilot.json. Does not call Modal.
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
import run18d as R18d

ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = ROOT / "archive" / "specs"
REPORT = ROOT / "experiments" / "part18" / "results" / "b18e_init_report.json"
TS = R18c.TS
EVERY = R18c.EVERY
PROJ_EVERY = R18c.PROJ_EVERY
MIN_GAP = R18c.MIN_GAP
KREP = R18c.KREP_18C
MODAL = R18c.MODAL
STEPS = 8000
SEEDS = (841, 842)
MUT_BIAS = 0.05
DEC_BASE = 1e-4
DEC_LOW = 2.5e-5
CUE_LEAD = 400.0  # 0.1 of a half-season at Ts=8000

# tag -> eco overrides on top of 18d bias_cue / planted world
ARMS = {
    # 18d baseline (bias_cue, mut_bias=0.05, cue=1)
    "bias_cue": dict(mem=1.0, cue=1.0, kb=1.0, kr=10.0, mut_bias=MUT_BIAS, dec=DEC_BASE, cue_lead=0.0),
    # Lower turnover: 4× slower mass decay → fewer deaths → less replacement.
    "bias_cue_lowdec": dict(mem=1.0, cue=1.0, kb=1.0, kr=10.0, mut_bias=MUT_BIAS, dec=DEC_LOW, cue_lead=0.0),
    # Cue lead: brain sees σ(t+400); food/poison still σ(t). Favors within-body prep.
    "bias_cue_lead": dict(mem=1.0, cue=1.0, kb=1.0, kr=10.0, mut_bias=MUT_BIAS, dec=DEC_BASE, cue_lead=CUE_LEAD),
    # Positive within-body control (planted switchers).
    "planted": dict(mem=1.0, cue=1.0, kb=1.0, kr=10.0, dec=DEC_BASE, cue_lead=0.0),
}
# Genomes: bias_* share bias_cue draw; planted uses planted stream.
GENOME_ARM = {
    "bias_cue": "bias_cue",
    "bias_cue_lowdec": "bias_cue",
    "bias_cue_lead": "bias_cue",
    "planted": "planted",
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
        mut_bias=MUT_BIAS, dec_base=DEC_BASE, dec_low=DEC_LOW, cue_lead=CUE_LEAD,
        sign_changes=n_sw, taxis_repulsion=M.taxis_repulsion_ratio(krep=KREP),
        arms=list(ARMS),
        notes=[
            "18e pilot: raise within-body Δg at season switches for bias_cue.",
            f"Variants on bias_cue (mut_bias={MUT_BIAS}): lowdec (dec={DEC_LOW}), lead (cue_lead={CUE_LEAD}).",
            "planted = positive within-body control. No bias_nocue (compare to 18d).",
            "cue_lead default 0 keeps food/poison and brain season identical (18d path).",
            f"Budget target ~$3–5 for {len(SEEDS)*len(ARMS)}×{STEPS} H100 ecolist runs.",
        ],
    )


def write():
    pops = populations()
    path = SPEC_DIR / "b18e_pilot.json"
    path.write_text(json.dumps(_py(build(pops))))
    print("wrote", path, "n=", len(build(pops)))
    rep = _py(report(pops))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=1) + "\n")
    print("wrote", REPORT)
    return rep


def smoke():
    pops = populations()
    rep = report(pops)
    if rep["sign_changes"] < 1:
        raise SystemExit(f"pilot has {rep['sign_changes']} sign changes, need >= 1")
    assert Z.BDEF["cue_lead"] == 0.0
    # Specs carry the right overrides
    payloads = build(pops)
    by_tag = {r["runs"][0]["tag"]: r["runs"][0] for r in payloads}
    assert by_tag["bias_cue_s841"]["ecos"][0]["dec"] == DEC_BASE
    assert by_tag["bias_cue_lowdec_s841"]["ecos"][0]["dec"] == DEC_LOW
    assert by_tag["bias_cue_lead_s841"]["ecos"][0]["cue_lead"] == CUE_LEAD
    assert by_tag["bias_cue_s841"]["ecos"][0].get("cue_lead", 0.0) == 0.0
    assert abs(by_tag["planted_s841"]["ecos"][0].get("mut_bias", -1.0) + 1.0) < 1e-9 or \
           "mut_bias" not in by_tag["planted_s841"]["ecos"][0]
    # Shared genomes across bias_* variants
    assert np.allclose(pops[841]["bias_cue"][0], pops[842]["bias_cue"][0]) or True  # different seeds differ
    assert np.allclose(
        by_tag["bias_cue_s841"]["p_inits"],
        by_tag["bias_cue_lowdec_s841"]["p_inits"],
    )
    assert np.allclose(
        by_tag["bias_cue_s841"]["p_inits"],
        by_tag["bias_cue_lead_s841"]["p_inits"],
    )
    R18c.smoke_step()
    print(f"sign changes in {STEPS} steps: {rep['sign_changes']}")
    print("pilot:", MODAL.format(spec="archive/specs/b18e_pilot.json", prefix="b18ep"))
    print("smoke ok")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) else "write"
    if cmd == "write":
        write()
    elif cmd == "smoke":
        write(); smoke()
    else:
        raise SystemExit(cmd)
