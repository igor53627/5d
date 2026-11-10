"""Part 18h competition pilot: planted vs bias_cue in one population.

A heritable lineage tag p[iL] (K=KB+1) marks founders; it is not read by the
brain or feeding and is not mutated (default K=KB is unchanged). 50/50 mixed
populations test whether planted within-body switchers are selected against
or win. A 50/50 bias_cue/bias_cue control checks tag neutrality.

Writes archive/specs/b18h_pilot.json. Does not call Modal.
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
REPORT = ROOT / "experiments" / "part18" / "results" / "b18h_init_report.json"
TS = R18c.TS
EVERY = R18c.EVERY
PROJ_EVERY = R18c.PROJ_EVERY
MIN_GAP = R18c.MIN_GAP
KREP = R18c.KREP_18C
MODAL = R18c.MODAL
STEPS = 20000
# Two seeds: ~$5, under $8 cap. Founders 621, 622. Distinct from 18d/18g.
SEED_OF_FOUNDER = {861: 621, 862: 622}
SEEDS = tuple(SEED_OF_FOUNDER)
MUT_BIAS = 0.05
HOM = 0.05
DEC = 1e-4
K_LIN = Z.KB + 1  # 44
TAG_PLANTED = 1.0
TAG_CUE = -1.0
N_HALF = I18c.N_INIT // 2  # 6


def _py(x):
    return R18c._py(x)


def with_tag(rows, tags):
    """Append a lineage column. rows (n, KB) → (n, KB+1)."""
    rows = np.asarray(rows, float)
    tags = np.asarray(tags, float).reshape(-1, 1)
    if tags.shape[0] == 1:
        tags = np.repeat(tags, len(rows), 0)
    return np.concatenate([rows, tags], 1)


def interleave(a, b):
    """a0,b0,a1,b1,... — spatially mixed placement order."""
    a, b = np.asarray(a), np.asarray(b)
    out = np.empty((len(a) + len(b),) + a.shape[1:], dtype=float)
    out[0::2], out[1::2] = a, b
    return out


def mixed_rows(p0, seed):
    """6 bias_cue (tag −1) interleaved with 6 planted switchers (tag +1)."""
    cue, _ = I.population(p0, seed, "bias_cue")
    planted, info = I.population(p0, seed, "planted")
    cue6 = with_tag(cue[:N_HALF], TAG_CUE)
    plt6 = with_tag(planted[-N_HALF:], TAG_PLANTED)
    return interleave(cue6, plt6), dict(n_cue=N_HALF, n_planted=N_HALF, plant_info=info[-N_HALF:])


def control_rows(p0, seed):
    """12 bias_cue genomes, half tagged −1 and half +1 (tag-neutrality)."""
    cue, _ = I.population(p0, seed, "bias_cue")
    a = with_tag(cue[:N_HALF], TAG_CUE)
    b = with_tag(cue[N_HALF:], TAG_PLANTED)  # tag only; genomes are bias_cue
    return interleave(a, b), dict(n_minus=N_HALF, n_plus=N_HALF)


def populations():
    founders = I18c.load_founders()
    pops = {}
    for seed, founder_seed in SEED_OF_FOUNDER.items():
        p0 = founders[founder_seed]
        mix, mix_info = mixed_rows(p0, seed)
        ctrl, ctrl_info = control_rows(p0, seed)
        pops[seed] = dict(p0=p0, founder_seed=founder_seed, mixed=(mix, mix_info), control=(ctrl, ctrl_info))
    return pops


def eco():
    c, e, opts = R18c.base_world()
    e = dict(e, mem=1.0, cue=1.0, kb=1.0, kr=10.0, mut_bias=MUT_BIAS, hom=HOM, cue_lead=0.0, dec=DEC)
    return c, e, opts


ARMS = ("mixed", "control")


def build(pops=None):
    c, e, opts = eco()
    pops = pops or populations()
    payloads = []
    for seed in SEEDS:
        for arm in ARMS:
            rows, _ = pops[seed][arm]
            payloads.append(dict(runs=[dict(
                c=c, ecos=[dict(e)], seed=int(seed), tag=f"{arm}_s{seed}", N=4, L=48,
                steps=STEPS, every=EVERY, proj_every=PROJ_EVERY, init="multi",
                n_init=I18c.N_INIT, p_spread=0.0, min_sep=1, min_gap=MIN_GAP, batch=1,
                model="season", opts=dict(opts), K=K_LIN, p0=_py(np.concatenate([pops[seed]["p0"], [0.0]])),
                p_inits=_py(rows),
            )]))
    return payloads


def report(pops):
    import numpy
    n_sw, frames = R18c.sign_changes(STEPS, every=EVERY, Ts=TS)
    return dict(
        numpy=numpy.__version__, Ts=TS, every=EVERY, steps=STEPS, seeds=list(SEEDS),
        seed_of_founder=dict(SEED_OF_FOUNDER), mut_bias=MUT_BIAS, K=K_LIN, iL=Z.iL,
        tag_planted=TAG_PLANTED, tag_cue=TAG_CUE, sign_changes=n_sw, sign_change_frames=frames,
        taxis_repulsion=M.taxis_repulsion_ratio(krep=KREP), arms=list(ARMS),
        notes=[
            "18h competition: 50/50 planted vs bias_cue, plus tag-neutrality control.",
            "p[iL] lineage tag (K=KB+1); unused by dynamics, not mutated.",
            f"Budget target ~$5 for {len(SEEDS)*len(ARMS)}×{STEPS} H100 ecolist runs.",
        ],
    )


def write():
    pops = populations()
    path = SPEC_DIR / "b18h_pilot.json"
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
    r = by["mixed_s861"]
    assert r["K"] == K_LIN and r["steps"] == STEPS
    assert r["ecos"][0]["krep"] == KREP and r["ecos"][0]["mut_bias"] == MUT_BIAS
    pins = np.asarray(r["p_inits"])
    assert pins.shape == (I18c.N_INIT, K_LIN)
    tags = pins[:, Z.iL]
    assert set(np.sign(tags).tolist()) == {-1.0, 1.0}
    assert int((tags < 0).sum()) == N_HALF and int((tags > 0).sum()) == N_HALF
    # control: all non-tag genome channels from bias_cue stream (not planted)
    ctrl = np.asarray(by["control_s861"]["p_inits"])
    assert ctrl.shape == pins.shape
    assert set(np.sign(ctrl[:, Z.iL]).tolist()) == {-1.0, 1.0}
    assert Z.KB + 1 == K_LIN
    R18c.smoke_step()
    print(f"sign changes in {STEPS} steps: {rep['sign_changes']}")
    print("pilot:", MODAL.format(spec="archive/specs/b18h_pilot.json", prefix="b18hp"))
    print("smoke ok")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "write"
    if cmd == "write":
        write()
    elif cmd == "smoke":
        write(); smoke()
    else:
        raise SystemExit(cmd)
