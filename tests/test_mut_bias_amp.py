"""Greptile P2: mut_bias overrides only g_A/g_B mutation amplitude; <0 keeps mutw."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

import fl_season as Z
import fl_glider as FG
import fl_eco3 as E3


def _tiny_state(L=12):
    yy, xx = np.mgrid[0:L, 0:L]
    dx = (xx - L // 2 + L / 2) % L - L / 2
    dy = (yy - L // 2 + L / 2) % L - L / 2
    A = (0.8 * np.exp(-(dx ** 2 + dy ** 2) / 5)).astype(np.float32)
    base = dict(A=A, x=0.1 * np.ones_like(A), y=0.1 * np.ones_like(A), z=np.zeros_like(A), c=np.full_like(A, 2.6))
    eco = dict(
        keat=0.0, dec=0.0, rF=0.0, F0=0.08, pmu=0.0, hom=0.0, mutn=0.0, mutw=0.0,
        krep=0.0, ks=2, pk=0.0, kb=0.0, kr=0.0, cp=0.5, rG=0.0, lam=float(L),
        smooth=1.0, Ts=400.0, season_rule=1.0, mem=0.0, cue=0.0, U1=0.1,
    )
    c = dict(mode="flow_iso_muscl", R=4.0, mu=0.2, sigma=0.2, chi=0.01, beta=0.5, theta=2.5,
             gam=0.5, tau_c=3.0, peaks=[1.0, 0.4, 1.0], dtA=0.3, eps=0.15, c_hi=3.5, asym=0.0, density=0.6,
             c_lo=2.5)
    st = Z.make_state(base, {**c, **eco}, K=Z.KB, seed=0)
    w = np.linspace(-0.4, 0.4, Z.NW, dtype=np.float32)
    Q = np.zeros_like(st["Q"])
    Q[Z.K0:Z.iM] = w.reshape((-1, 1, 1)) * A
    Q[0] = 0.2 * A
    Q[1] = -0.2 * A
    st["Q"] = Q
    st["F"] = np.full_like(A, 0.08)
    st["F2"] = np.zeros_like(A)
    return st, c, eco


def _p_vec(Q, A):
    """Mass-weighted genome: sum(Q)/sum(A) per channel."""
    mass = float(A.sum()) + 1e-12
    return np.array([float(Q[k].sum()) / mass for k in range(Z.KB)])


def _one_step(st, c, eco):
    L = st["A"].shape[0]
    P = Z.params({**c, **eco})
    KA = jnp.asarray(FG.ring_kernel_np(L, 2, float(c["R"]), tuple(c["peaks"])))
    Ko = jnp.asarray(FG.ring_kernel_np(L, 2, 3.0))
    KX = E3.kernels(L, 2)
    s = {k: jnp.asarray(v) for k, v in st.items()}
    s["key"] = jax.random.PRNGKey(7)
    out = Z.step(s, P, KA, Ko, KX, do_mut=True, do_hom=False, m=1)
    return np.asarray(out["Q"]), np.asarray(out["A"])


@pytest.fixture(scope="module")
def tiny():
    return _tiny_state()


def test_mut_bias_nonnegative_mutates_only_bias_channels(tiny):
    st, c, eco = tiny
    eco = dict(eco, mutn=0.0, mutw=0.0, mut_bias=0.25, hom=0.0)
    p0 = _p_vec(np.asarray(st["Q"]), np.asarray(st["A"]))
    Q1, A1 = _one_step(st, c, eco)
    assert np.isfinite(A1).all() and float(A1.sum()) > 0.1
    p1 = _p_vec(Q1, A1)
    delta = p1 - p0
    iga = Z.K0 + Z.b2_flat(Z.OUT_GA)
    igb = Z.K0 + Z.b2_flat(Z.OUT_GB)
    assert np.isfinite(delta).all()
    assert abs(delta[iga]) + abs(delta[igb]) > 1e-4
    other = [i for i in range(Z.K0, Z.iM) if i not in (iga, igb)]
    assert np.max(np.abs(delta[other])) < 1e-5
    assert abs(delta[0]) < 1e-5 and abs(delta[1]) < 1e-5
    assert abs(delta[Z.iM]) < 1e-5


def test_mut_bias_negative_uses_mutw_for_biases(tiny):
    st, c, eco = tiny
    iga = Z.K0 + Z.b2_flat(Z.OUT_GA)
    igb = Z.K0 + Z.b2_flat(Z.OUT_GB)
    p0 = _p_vec(np.asarray(st["Q"]), np.asarray(st["A"]))

    eco0 = dict(eco, mutn=0.0, mutw=0.0, mut_bias=-1.0, hom=0.0)
    p1 = _p_vec(*_one_step(st, c, eco0))
    assert abs(p1[iga] - p0[iga]) < 1e-5 and abs(p1[igb] - p0[igb]) < 1e-5

    eco1 = dict(eco, mutn=0.0, mutw=0.2, mut_bias=-1.0, hom=0.0)
    p2 = _p_vec(*_one_step(st, c, eco1))
    assert abs(p2[iga] - p0[iga]) + abs(p2[igb] - p0[igb]) > 1e-4
    other = Z.K0 + Z.w1_flat(0, Z.IN_A)
    assert abs(p2[other] - p0[other]) > 1e-4


def test_mut_bias_default_is_legacy_off():
    assert Z.BDEF["mut_bias"] < 0.0
    assert float(Z.params({})["mut_bias"]) < 0.0
