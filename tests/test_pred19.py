"""Part 19: predator–prey defaults, mass transfer, avoidance taxis, and specs."""
import json
from pathlib import Path

import numpy as np
import pytest
import jax.numpy as jnp

import fl_pred as Z
import fl_glider as FG
import fl_eco3 as E3
import metrics19 as M
import run19 as R

ROOT = Path(__file__).resolve().parents[1]


def _blob(L, cx, cy, sig=3.0, amp=0.8):
    yy, xx = np.mgrid[0:L, 0:L]
    dx = (xx - cx + L / 2) % L - L / 2
    dy = (yy - cy + L / 2) % L - L / 2
    return (amp * np.exp(-(dx ** 2 + dy ** 2) / (2 * sig ** 2))).astype(np.float32)


def _tiny(L=24, kpred=0.0, ka=0.0, kp=0.0, krep_x=0.0, A=None, AP=None, plant_av=0.0):
    A = _blob(L, L // 2, L // 2) if A is None else A
    AP = np.zeros_like(A) if AP is None else AP
    c, eco, _ = R.base_world()
    eco = dict(eco, keat=0.0, dec=0.0, dec_p=0.0, rF=0.0, kb=0.0, kr=0.0,
               ka=ka, kp=kp, krep=0.0, krep_x=krep_x, chi=0.0, mutn=0.0, mutw=0.0,
               hom=0.0, kpred=kpred, eff=0.4, ke=0.0, ce=0.0, U1=0.1, ws=2.0)
    base = dict(A=A, x=0.1 * np.ones_like(A), y=0.1 * np.ones_like(A),
                z=np.zeros_like(A), c=np.full_like(A, 2.6))
    st = Z.make_state(base, {**c, **eco}, K=Z.KB, seed=0)
    st["AP"] = AP.astype(np.float32)
    if plant_av:
        # Set b2[g_Av] so tanh(ws*b) is near sign(plant_av); ws=2, tanh(2*0.8)≈0.96.
        w = np.zeros(Z.NW, np.float32)
        w[Z.b2_flat(Z.OUT_AV)] = 0.8 if plant_av > 0 else -0.8
        Q = np.zeros_like(st["Q"])
        Q[Z.K0:Z.K0 + Z.NW] = w.reshape((-1, 1, 1)) * A
        st["Q"] = Q
    P = Z.params({**c, **eco})
    KA = jnp.asarray(FG.ring_kernel_np(L, 2, float(c["R"]), tuple(c["peaks"])))
    Ko = jnp.asarray(FG.ring_kernel_np(L, 2, 3.0))
    KX = E3.kernels(L, 2)
    return st, P, KA, Ko, KX, c, eco


def _step(st, P, KA, Ko, KX):
    s = {k: jnp.asarray(v) for k, v in st.items()}
    return Z.step(s, P, KA, Ko, KX, do_mut=False, do_hom=False)


def test_defaults_are_off():
    assert Z.BDEF["kpred"] == 0.0
    assert Z.BDEF["ka"] == 0.0
    assert Z.BDEF["kp"] == 0.0
    assert Z.BDEF["krep_x"] == 0.0
    assert Z.BDEF["dec_p"] == 0.0
    P = Z.params({})
    assert float(P["kpred"]) == 0.0
    assert float(P["ka"]) == 0.0
    assert Z.KB == Z.K0 + Z.NW
    assert Z.NIN == 4 and Z.NOUT == 4


def test_kpred_zero_transfers_no_mass():
    L = 24
    A = _blob(L, 12, 12)
    AP = _blob(L, 12, 12, amp=0.4)
    st, P, KA, Ko, KX, _, _ = _tiny(L, kpred=0.0, A=A, AP=AP)
    out = _step(st, P, KA, Ko, KX)
    loss = float(np.asarray(out["D"][1]).sum())
    gain = float(np.asarray(out["D"][2]).sum())
    assert loss == pytest.approx(0.0, abs=1e-8)
    assert gain == pytest.approx(0.0, abs=1e-8)
    # Mass almost conserved (soliton motion may smear a little, not eat).
    assert float(np.asarray(out["A"]).sum()) == pytest.approx(float(A.sum()), rel=1e-3, abs=1e-3)
    assert float(np.asarray(out["AP"]).sum()) == pytest.approx(float(AP.sum()), rel=1e-3, abs=1e-3)


def test_predation_transfers_mass_near_efficiency():
    L = 24
    A = _blob(L, 12, 12)
    AP = _blob(L, 12, 12, amp=0.4)
    st, P, KA, Ko, KX, _, eco = _tiny(L, kpred=0.05, A=A, AP=AP)
    out = _step(st, P, KA, Ko, KX)
    loss = float(np.asarray(out["D"][1]).sum())
    gain = float(np.asarray(out["D"][2]).sum())
    assert loss > 1e-3
    assert gain > 1e-3
    ratio = gain / loss
    assert 0.25 <= ratio <= 0.6
    assert float(np.asarray(out["A"]).sum()) < float(A.sum())
    assert float(np.asarray(out["AP"]).sum()) > float(AP.sum())


def test_avoidance_taxis_moves_prey_away():
    L = 32
    # Prey at the centre, predator to the right. g_Av>0 and ka>0 should push prey left.
    A = _blob(L, 16, 16)
    AP = _blob(L, 22, 16, sig=3.0, amp=0.8)
    st, P, KA, Ko, KX, _, _ = _tiny(L, kpred=0.0, ka=1.0, A=A, AP=AP, plant_av=1.0)
    out = _step(st, P, KA, Ko, KX)
    A1 = np.asarray(out["A"])
    def com_x(field):
        xs = np.arange(L)
        w = field.sum(axis=0)
        # circular COM along x (axis 1 of yy,xx mgrid — xx varies over columns)
        ang = 2 * np.pi * xs / L
        # field is (y, x)
        mass = field.sum(axis=0)
        c, s = (mass * np.cos(ang)).sum(), (mass * np.sin(ang)).sum()
        return (np.arctan2(s, c) % (2 * np.pi)) / (2 * np.pi) * L
    x0, x1 = com_x(A), com_x(A1)
    # shortest signed displacement on the circle, prey should move -x (away)
    dx = ((x1 - x0 + L / 2) % L) - L / 2
    assert dx < -0.02, f"prey COM moved {dx:+.4f}, expected left"


def test_metrics_synthetic_lotka_volterra_lag():
    t = np.arange(0, 4000, 50, dtype=float)
    # Predator lags prey by ~200 steps (quarter of a 800-step period).
    mA = 2 + 0.5 * np.sin(2 * np.pi * t / 800)
    mP = 1.5 + 0.5 * np.sin(2 * np.pi * t / 800 - np.pi / 2)
    lag, r = M.best_lag(mA, mP, t)
    assert r > 0.9
    assert lag == pytest.approx(200, abs=60)
    period = M.dominant_period(t, mA)
    assert period == pytest.approx(800, rel=0.15)
    assert M.final_status(mA, mP) == "coexist"
    assert M.coexistence_time(mA, mP, t) == t[-1]
    assert M.first_extinction(mA, mP, t)[0] is None


def test_metrics_extinction():
    t = np.array([0, 100, 200, 300, 400], float)
    mA = np.array([2, 1, 0.5, 0.05, 0.0])
    mP = np.array([1, 1.2, 0.8, 0.4, 0.2])
    who, when = M.first_extinction(mA, mP, t)
    assert who == "prey"
    assert when == 300
    assert M.final_status(mA, mP) == "pred_only"


def test_committed_19_specs():
    path = ROOT / "archive" / "specs" / "b19_pilot.json"
    if not path.exists():
        R.write()
    spec = json.loads(path.read_text())
    assert len(spec) == 2
    for item, seed in zip(spec, R.SEEDS):
        r = item["runs"][0]
        assert r["model"] == "pred"
        assert r["n_pred"] == R.N_PRED
        assert r["n_init"] == R.N_PREY
        assert r["steps"] == R.STEPS
        assert r["K"] == Z.KB
        assert r["L"] == 48 and r["N"] == 4
        assert [e["kpred"] for e in r["ecos"]] == [R.KPRED[a] for a in R.ARMS]
        assert r["tag"] == f"pp_s{seed}"
        # Defaults-off knobs are set explicitly in the spec (not left to BDEF).
        for e in r["ecos"]:
            assert e["ka"] == R.KA
            assert e["kp"] == R.KP
            assert e["eff"] == R.EFF
            assert e["dec_p"] == R.DEC_P
