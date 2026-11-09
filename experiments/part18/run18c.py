"""Part 18c specs and the CPU smoke check.

Writes archive/specs/b18c_pilot.json and b18c_full.json. Does not call Modal.
Pilot and full commands are printed by `python experiments/part18/run18c.py smoke`.
"""
import json
import sys
from pathlib import Path
import numpy as np
import _path  # noqa: F401
import fl_season as Z
import init18c as I
import metrics18c as M

ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = ROOT / "archive" / "specs"
REPORT = ROOT / "experiments" / "part18" / "results" / "b18c_init_report.json"
TS = 8000.0
EVERY = 200
PROJ_EVERY = 400
MIN_GAP = 4
# 18c only: raise repulsion to ~parity with food taxis (ratio≈1 at krep=90).
# Parts 18/18b keep krep=10 in their own specs; this is not a fl_season default.
KREP_18C = 90.0
MODAL = "modal run src/modal_app.py::spawn_h100 --kind ecolist --spec {spec} --prefix {prefix}"
STEPS = dict(pilot=12000, full=20000)
SEEDS = dict(pilot=(731,), full=(731, 732, 733, 734))
# Flags that differ across arms. Ecology other than these is the Part 18b world, with Ts halved.
ARM_FLAGS = {
    "lo": dict(mem=1.0, cue=0.0, kb=1.0, kr=10.0),
    "inputs": dict(mem=1.0, cue=0.0, kb=1.0, kr=10.0),
    "cue": dict(mem=1.0, cue=1.0, kb=1.0, kr=10.0),
    "planted": dict(mem=1.0, cue=1.0, kb=1.0, kr=10.0),
    "nobrain": dict(mem=0.0, cue=0.0, kb=0.0, kr=0.0),
}


def _py(x):
    if isinstance(x, np.ndarray):
        return [_py(v) for v in x.tolist()]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, dict):
        return {k: _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    return x


def base_world():
    """Soliton and ecology copied from the first Part 18b mem run, season period halved."""
    spec = json.loads(I.SPEC_18B.read_text())
    run = next(item["runs"][0] for item in spec if item["runs"][0]["tag"].startswith("mem_"))
    eco = dict(run["ecos"][0])
    eco["Ts"] = TS
    eco["season_rule"] = 1.0
    eco["krep"] = KREP_18C
    return dict(run["c"]), eco, list(run["opts"].items())


def sign_changes(steps, every=EVERY, Ts=TS):
    ts = list(range(0, steps + 1, every))
    sg = [float(Z.season_half(t, Ts, np)) for t in ts]
    return sum(sg[i] != sg[i - 1] for i in range(1, len(sg))), sg


def populations():
    founders = I.load_founders()
    pops = {}
    for seed, founder_seed in I.SEED_OF_FOUNDER.items():
        p0 = founders[founder_seed]
        pops[seed] = {}
        for arm in ("lo", "inputs", "planted"):
            rows, info = I.population(p0, seed, arm)
            pops[seed][arm] = (rows, info, p0)
        pops[seed]["cue"] = pops[seed]["inputs"]
        pops[seed]["nobrain"] = pops[seed]["lo"]
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
                n_init=I.N_INIT, p_spread=0.0, min_sep=1, min_gap=MIN_GAP, batch=1,
                model="season", opts=dict(opts), K=Z.KB, p0=_py(p0), p_inits=_py(rows),
            )]))
    return payloads


def report(pops):
    import numpy
    founders = {s: pops[s]["lo"][2] for s in pops}
    arms = {}
    for seed, founder_seed in I.SEED_OF_FOUNDER.items():
        p0 = founders[seed]
        arms[str(seed)] = dict(founder_seed=founder_seed, arms={})
        for arm in ("lo", "inputs", "planted"):
            rows, info, _ = pops[seed][arm]
            rec = I.genome_report(rows, p0)
            if arm == "planted":
                rec["plant"] = [None if x is None else dict(rms=x["rms"], hits=x["hits"], trials=x["trials"]) for x in info]
            arms[str(seed)]["arms"][arm] = rec
        arms[str(seed)]["arms"]["cue"] = "same_genomes_as_inputs"
        arms[str(seed)]["arms"]["nobrain"] = "same_genomes_as_lo"
    n_pilot, _ = sign_changes(STEPS["pilot"])
    n_full, _ = sign_changes(STEPS["full"])
    ratio = M.taxis_repulsion_ratio()
    return dict(
        numpy=numpy.__version__, Ts=TS, every=EVERY, min_gap=MIN_GAP,
        steps=STEPS, sign_changes=dict(pilot=n_pilot, full=n_full),
        taxis_repulsion=ratio, seeds=arms,
        shared_genomes={"cue": "inputs", "nobrain": "lo"},
        notes=[
            "p_spread is 0 because every founder genome is stored in p_inits.",
            "min_gap is Chebyshev cells between 0.02 contours at placement only.",
            "krep is KREP_18C (90): taxis/repulsion ratio ≈ 1. Parts 18/18b specs still use 10.",
            "Pilot fused by t=200 at krep=10; CPU L=32 probe kept n=4 through 150 steps at krep=90.",
        ],
    )


def write():
    pops = populations()
    for kind in ("pilot", "full"):
        path = SPEC_DIR / f"b18c_{kind}.json"
        path.write_text(json.dumps(_py(build(kind, pops))))
        print("wrote", path)
    rep = _py(report(pops))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=1) + "\n")
    print("wrote", REPORT)
    return rep


def smoke_step():
    """One feeding of pure A in each season, and a cue wired only into memory."""
    import jax.numpy as jnp
    import fl_glider as FG
    import fl_eco3 as E3
    L = 32
    yy, xx = np.mgrid[0:L, 0:L]

    def blob():
        dx = (xx - 16 + L / 2) % L - L / 2
        dy = (yy - 16 + L / 2) % L - L / 2
        return (0.8 * np.exp(-(dx ** 2 + dy ** 2) / 8)).astype(np.float32)

    A = blob()
    c, eco, _ = base_world()
    eco = dict(eco, keat=0.01, kb=0.0, kr=0.0, mem=1.0, cue=1.0, alpha=1.0, hom=0.0, mutn=0.0, mutw=0.0, rG=0.0, Ts=400.0)
    base = dict(A=A, x=0.1 * np.ones_like(A), y=0.1 * np.ones_like(A), z=np.zeros_like(A), c=np.full_like(A, 2.6))
    st = Z.make_state(base, {**c, **eco}, K=Z.KB, seed=0)
    w = np.zeros(Z.NW, np.float32)
    w[Z.w1_flat(0, Z.IN_CUE)] = 1.0
    w[Z.w2_flat(Z.OUT_H, 0)] = 1.0
    Q = np.zeros_like(st["Q"])
    Q[Z.K0:Z.iM] = w.reshape((-1, 1, 1)) * A
    st["Q"] = Q
    st["F"] = np.full_like(A, 0.08)
    st["F2"] = np.zeros_like(A)
    P = Z.params({**c, **eco})
    KA = jnp.asarray(FG.ring_kernel_np(L, 2, float(c["R"]), tuple(c["peaks"])))
    Ko = jnp.asarray(FG.ring_kernel_np(L, 2, 3.0))
    KX = E3.kernels(L, 2)

    def once(t):
        s = {k: jnp.asarray(v) for k, v in st.items()}
        s["t"] = jnp.float32(t)
        out = Z.step(s, P, KA, Ko, KX)
        A1 = np.asarray(out["A"])
        R = float(np.average(np.asarray(out["R"]), weights=np.maximum(A1, 1e-8)))
        h = np.asarray(out["Q"][Z.iM]) / np.maximum(A1, 1e-6)
        return R, float(np.average(h, weights=np.maximum(A1, 1e-8)))

    r_pos, h_pos = once(0.0)
    r_neg, h_neg = once(250.0)
    print(f"smoke step  R(A season) {r_pos:+.3f}  R(B season) {r_neg:+.3f}  h {h_pos:+.3f} -> {h_neg:+.3f}")
    if not (r_pos > 0.5 and r_neg < -0.2 and h_pos > 0.5 and h_neg < -0.5):
        raise SystemExit("smoke step failed the reward/cue sign check")


def smoke():
    pops = populations()
    rep = report(pops)
    if rep["sign_changes"]["full"] < 4:
        raise SystemExit(f"full run has {rep['sign_changes']['full']} sign changes, need >= 4")
    ratio = rep["taxis_repulsion"]["ratio"]
    print(f"sign changes pilot {rep['sign_changes']['pilot']}  full {rep['sign_changes']['full']}")
    ratio90 = M.taxis_repulsion_ratio(krep=KREP_18C)
    print(f"taxis/repulsion v_rep/v_food at krep=10: {ratio:.3f} (cap {rep['taxis_repulsion']['cap']:.3f})")
    print(f"taxis/repulsion at krep={KREP_18C:g} (18c): {ratio90['ratio']:.3f}")
    for seed, rec in rep["seeds"].items():
        inp, lo, pl = rec["arms"]["inputs"], rec["arms"]["lo"], rec["arms"]["planted"]
        print(f"  seed {seed} founder {rec['founder_seed']}: inputs switch {inp['n_switch']}/{inp['n']} "
              f"std_new {inp['std_new']:.3f} oldΔ {inp['max_abs_delta_old']:.3g}; "
              f"lo switch {lo['n_switch']}/{lo['n']}; planted switch {pl['n_switch']}/{pl['n']}")
        if inp["n_switch"] != 0 or inp["max_abs_delta_old"] != 0.0:
            raise SystemExit("inputs arm moved old weights or produced a cue-switcher")
        if pl["n_switch"] != I.N_PLANTED:
            raise SystemExit("planted arm does not have 6 switchers")
    smoke_step()
    print("pilot:", MODAL.format(spec="archive/specs/b18c_pilot.json", prefix="b18cp"))
    print("full: ", MODAL.format(spec="archive/specs/b18c_full.json", prefix="b18c"))
    print("smoke ok")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if cmd == "write":
        write()
    elif cmd == "smoke":
        smoke()
    else:
        raise SystemExit("usage: run18c.py [smoke|write]")
