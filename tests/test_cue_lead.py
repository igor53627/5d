"""cue_lead: brain sees season(t+lead); food/poison stay on season(t). Default 0 = legacy."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

import fl_season as Z
import fl_glider as FG
import fl_eco3 as E3


def _state_at_t(L=12, t=0.0, Ts=400.0, cue=1.0, cue_lead=0.0, season_rule=1.0):
    yy, xx = np.mgrid[0:L, 0:L]
    dx = (xx - L // 2 + L / 2) % L - L / 2
    dy = (yy - L // 2 + L / 2) % L - L / 2
    A = (0.8 * np.exp(-(dx ** 2 + dy ** 2) / 5)).astype(np.float32)
    base = dict(A=A, x=0.1 * np.ones_like(A), y=0.1 * np.ones_like(A), z=np.zeros_like(A), c=np.full_like(A, 2.6))
    eco = dict(
        keat=0.0, dec=0.0, rF=0.0, F0=0.08, pmu=0.0, hom=0.0, mutn=0.0, mutw=0.0,
        krep=0.0, ks=2, pk=0.0, kb=1.0, kr=0.0, cp=0.5, rG=0.0, lam=float(L),
        smooth=1.0, Ts=Ts, season_rule=season_rule, mem=0.0, cue=cue, cue_lead=cue_lead, U1=0.1,
    )
    c = dict(mode="flow_iso_muscl", R=4.0, mu=0.2, sigma=0.2, chi=0.01, beta=0.5, theta=2.5,
             gam=0.5, tau_c=3.0, peaks=[1.0, 0.4, 1.0], dtA=0.3, eps=0.15, c_hi=3.5, asym=0.0, density=0.6,
             c_lo=2.5)
    st = Z.make_state(base, {**c, **eco}, K=Z.KB, seed=0)
    # Wire cue column of W1 strongly into hidden, and hidden into g_A, so cue sign moves o[0].
    w = np.zeros(Z.NW, dtype=np.float32)
    W1 = w[: Z.H * Z.NIN].reshape(Z.H, Z.NIN)
    W1[:, Z.IN_CUE] = 2.0
    b2 = w[Z.H * Z.NIN + Z.H + Z.NOUT * Z.H :]
    # leave b2 at 0; W2 row for g_A from hidden 0
    W2 = w[Z.H * Z.NIN + Z.H : Z.H * Z.NIN + Z.H + Z.NOUT * Z.H].reshape(Z.NOUT, Z.H)
    W2[Z.OUT_GA, 0] = 2.0
    Q = np.zeros_like(st["Q"])
    Q[Z.K0:Z.iM] = w.reshape((-1, 1, 1)) * A
    st["Q"] = Q
    st["t"] = np.float32(t)
    st["F"] = np.full_like(A, 0.08)
    st["F2"] = np.zeros_like(A)
    return st, c, eco


def _step_out(st, c, eco):
    L = st["A"].shape[0]
    P = Z.params({**c, **eco})
    KA = jnp.asarray(FG.ring_kernel_np(L, 2, float(c["R"]), tuple(c["peaks"])))
    Ko = jnp.asarray(FG.ring_kernel_np(L, 2, 3.0))
    KX = E3.kernels(L, 2)
    s = {k: jnp.asarray(v) for k, v in st.items()}
    s["key"] = jax.random.PRNGKey(3)
    return Z.step(s, P, KA, Ko, KX, do_mut=False, do_hom=False, m=1)


def test_cue_lead_default_is_legacy():
    assert Z.BDEF["cue_lead"] == 0.0
    assert float(Z.params({})["cue_lead"]) == 0.0


def test_cue_lead_zero_matches_current_season_sign():
    """At t just before a half-period flip, lead=0 keeps cue = current food season."""
    Ts = 400.0
    t = 199.0  # still A under season_half (u<=0.5); flip at 200
    st, c, eco = _state_at_t(t=t, Ts=Ts, cue=1.0, cue_lead=0.0)
    out = _step_out(st, c, eco)
    # D3 = A*g_A; with strong cue->g_A wiring, mean D3/mean A tracks cue sign
    gA = float(np.asarray(out["D"][3]).sum() / (np.asarray(out["A"]).sum() + 1e-12))
    assert gA > 0.1  # cue +1 → positive g_A


def test_cue_lead_advances_brain_season_before_food_flip():
    """Same t; cue_lead past the flip → brain sees B while food still A."""
    Ts = 400.0
    t = 199.0
    st0, c, eco0 = _state_at_t(t=t, Ts=Ts, cue=1.0, cue_lead=0.0)
    st1, _, eco1 = _state_at_t(t=t, Ts=Ts, cue=1.0, cue_lead=5.0)  # t+5 = 204 → B
    out0 = _step_out(st0, c, eco0)
    out1 = _step_out(st1, c, eco1)
    gA0 = float(np.asarray(out0["D"][3]).sum() / (np.asarray(out0["A"]).sum() + 1e-12))
    gA1 = float(np.asarray(out1["D"][3]).sum() / (np.asarray(out1["A"]).sum() + 1e-12))
    assert gA0 > 0.1
    assert gA1 < -0.1
    # Food season unchanged: both should have eaten A (D0) and not treated B as food path differently
    # gain uses sg(t); with keat=0 no eat, so just check season helpers agree
    assert float(Z.season_half(t, Ts, np)) == 1.0
    assert float(Z.season_half(t + 5.0, Ts, np)) == -1.0
