"""Part 19 predator–prey pilot specs.

Writes archive/specs/b19_pilot.json. Does not call Modal.
Two seeds × three predation rates on the Part 15 micro-brain world
(4D L=48, 8 prey + 4 predators, 6000 steps).
"""
import json
import sys
from pathlib import Path
import numpy as np
import _path  # noqa: F401
import fl_pred as Z

ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = ROOT / "archive" / "specs"
SPEC_15 = SPEC_DIR / "b15p.json"
REPORT = ROOT / "experiments" / "part19" / "results" / "b19_init_report.json"
MODAL = "modal run src/modal_app.py::spawn_h100 --kind ecolist --spec {spec} --prefix {prefix}"

STEPS = 6000
EVERY = 200
PROJ_EVERY = 400
MIN_GAP = 4
SEEDS = (191, 192)
N_PREY = 8
N_PRED = 4
# Guessed window. mid aims at coexistence; low/high are the bracketing rates.
KPRED = dict(low=0.015, mid=0.05, high=0.15)
EFF = 0.4
DEC_P = 6e-4
KA = 1.0
KP = 1.0
KREP_X = 5.0
ARMS = ("low", "mid", "high")


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
    """Soliton and ecology from the first Part 15 4D pilot run, on a cheaper L=48 grid."""
    spec = json.loads(SPEC_15.read_text())
    run = spec[0]["runs"][0]
    eco = dict(run["ecos"][0])
    # Part 19 knobs (defaults in fl_pred are 0 / off; specs turn them on).
    eco.update(eff=EFF, dec_p=DEC_P, ka=KA, kp=KP, krep_x=KREP_X)
    return dict(run["c"]), eco, dict(run["opts"])


def arm_eco(base, arm):
    e = dict(base)
    e["kpred"] = KPRED[arm]
    return e


def build():
    c, eco, opts = base_world()
    payloads = []
    # One container per seed, three arms batched together to cut JIT cost.
    for seed in SEEDS:
        ecos = [arm_eco(eco, arm) for arm in ARMS]
        payloads.append(dict(runs=[dict(
            c=c, ecos=ecos, seed=int(seed), tag=f"pp_s{seed}", N=4, L=48,
            steps=STEPS, every=EVERY, proj_every=PROJ_EVERY, init="multi",
            n_init=N_PREY, n_pred=N_PRED, p_spread=0.5, p_spread_pred=0.5,
            min_sep=1, min_gap=MIN_GAP, batch=3,
            model="pred", opts=dict(opts), K=Z.KB,
            p0=[0.0] * Z.KB, p0_pred=[0.0] * Z.KB,
        )]))
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
            "Part 19 pilot: coexistence smoke on 2 seeds × 3 kpred values.",
            "World is Part 15 micro-brain ecology on 4D L=48 (not 56), 8 prey + 4 predators.",
            "Verdict is coexistence / immediate extinction, not an arms-race claim.",
            f"Budget target ~$5 (hard cap $8) for {len(SEEDS)} batched H100 ecolist runs × {STEPS} steps.",
        ],
    )


def write():
    payloads = build()
    path = SPEC_DIR / "b19_pilot.json"
    path.write_text(json.dumps(_py(payloads)))
    print("wrote", path, "n=", len(payloads), "ecos_per=", len(payloads[0]["runs"][0]["ecos"]))
    rep = _py(report())
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=1) + "\n")
    print("wrote", REPORT)
    return rep


def smoke_step():
    """Overlapping blobs: kpred=0 transfers no mass; kpred>0 transfers with gain≈eff·loss."""
    import jax.numpy as jnp
    import fl_glider as FG
    import fl_eco3 as E3
    L = 24
    yy, xx = np.mgrid[0:L, 0:L]

    def blob(cx, cy, sig=3.0):
        dx = (xx - cx + L / 2) % L - L / 2
        dy = (yy - cy + L / 2) % L - L / 2
        return (0.8 * np.exp(-(dx ** 2 + dy ** 2) / (2 * sig ** 2))).astype(np.float32)

    A = blob(12, 12)
    AP = blob(12, 12) * 0.5
    c, eco, _ = base_world()
    eco = dict(eco, keat=0.0, dec=0.0, dec_p=0.0, rF=0.0, kb=0.0, kr=0.0,
               ka=0.0, kp=0.0, krep=0.0, krep_x=0.0, chi=0.0, mutn=0.0, mutw=0.0,
               hom=0.0, kpred=0.0, eff=EFF, U1=0.1)
    # 2D base; drop 4D-only assumptions by using the same scalar c fields.
    base = dict(A=A, x=0.1 * np.ones_like(A), y=0.1 * np.ones_like(A),
                z=np.zeros_like(A), c=np.full_like(A, 2.6))
    st = Z.make_state(base, {**c, **eco}, K=Z.KB, seed=0)
    st["AP"] = AP
    P0 = Z.params({**c, **eco})
    KA = jnp.asarray(FG.ring_kernel_np(L, 2, float(c["R"]), tuple(c["peaks"])))
    Ko = jnp.asarray(FG.ring_kernel_np(L, 2, 3.0))
    KX = E3.kernels(L, 2)

    def once(kpred):
        s = {k: jnp.asarray(v) for k, v in st.items()}
        Pp = Z.params({**c, **eco, "kpred": kpred})
        out = Z.step(s, Pp, KA, Ko, KX, do_mut=False, do_hom=False)
        loss = float(np.asarray(out["D"][1]).sum())
        gain = float(np.asarray(out["D"][2]).sum())
        return float(np.asarray(out["A"]).sum()), float(np.asarray(out["AP"]).sum()), loss, gain

    mA0, mP0 = float(A.sum()), float(AP.sum())
    a0, p0, loss0, gain0 = once(0.0)
    a1, p1, loss1, gain1 = once(0.05)
    print(f"smoke pred  off dA={a0-mA0:+.4f} dP={p0-mP0:+.4f} loss={loss0:.4f} gain={gain0:.4f}")
    print(f"smoke pred  on  dA={a1-mA0:+.4f} dP={p1-mP0:+.4f} loss={loss1:.4f} gain={gain1:.4f} ratio={gain1/max(loss1,1e-12):.3f}")
    if loss0 > 1e-8 or gain0 > 1e-8:
        raise SystemExit("kpred=0 still transferred mass")
    if loss1 <= 1e-4 or gain1 <= 1e-4:
        raise SystemExit("kpred=0.05 transferred almost no mass")
    ratio = gain1 / loss1
    if not (0.25 <= ratio <= 0.6):
        raise SystemExit(f"gain/loss={ratio:.3f} not near eff={EFF}")


def smoke():
    write()
    payloads = build()
    assert len(payloads) == len(SEEDS)
    for item, seed in zip(payloads, SEEDS):
        r = item["runs"][0]
        assert r["model"] == "pred"
        assert r["n_pred"] == N_PRED and r["n_init"] == N_PREY
        assert r["steps"] == STEPS and r["K"] == Z.KB
        assert r["batch"] == 3 and len(r["ecos"]) == 3
        assert r["tag"] == f"pp_s{seed}"
        got = [e["kpred"] for e in r["ecos"]]
        assert got == [KPRED[a] for a in ARMS]
        assert all(e["ka"] == KA and e["kp"] == KP for e in r["ecos"])
    assert Z.BDEF["kpred"] == 0.0 and Z.BDEF["ka"] == 0.0 and Z.BDEF["kp"] == 0.0
    smoke_step()
    print("pilot:", MODAL.format(spec="archive/specs/b19_pilot.json", prefix="b19p"))
    print("smoke ok")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "write"
    if cmd == "write":
        write()
    elif cmd == "smoke":
        write(); smoke()
    else:
        raise SystemExit(cmd)
