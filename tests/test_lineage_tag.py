"""Optional p[iL] lineage tag: unused by dynamics, not mutated; K=KB is legacy."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

import fl_season as Z
import fl_glider as FG
import fl_eco3 as E3


def _tiny(L=12, K=None, tag=None, mutn=0.0, mutw=0.0, mut_bias=-1.0, t=0.0):
    K = int(K if K is not None else Z.KB)
    yy, xx = np.mgrid[0:L, 0:L]
    dx = (xx - L // 2 + L / 2) % L - L / 2
    dy = (yy - L // 2 + L / 2) % L - L / 2
    A = (0.8 * np.exp(-(dx ** 2 + dy ** 2) / 5)).astype(np.float32)
    base = dict(A=A, x=0.1 * np.ones_like(A), y=0.1 * np.ones_like(A), z=np.zeros_like(A), c=np.full_like(A, 2.6))
    eco = dict(
        keat=0.0, dec=0.0, rF=0.0, F0=0.08, pmu=0.0, hom=0.0, mutn=mutn, mutw=mutw, mut_bias=mut_bias,
        krep=0.0, ks=2, pk=0.0, kb=1.0, kr=0.0, cp=0.5, rG=0.0, lam=float(L),
        smooth=1.0, Ts=400.0, season_rule=1.0, mem=0.0, cue=1.0, U1=0.1,
    )
    c = dict(mode="flow_iso_muscl", R=4.0, mu=0.2, sigma=0.2, chi=0.01, beta=0.5, theta=2.5,
             gam=0.5, tau_c=3.0, peaks=[1.0, 0.4, 1.0], dtA=0.3, eps=0.15, c_hi=3.5, asym=0.0, density=0.6,
             c_lo=2.5)
    st = Z.make_state(base, {**c, **eco}, K=K, seed=0)
    w = np.linspace(-0.3, 0.3, Z.NW, dtype=np.float32)
    Q = np.zeros((K,) + A.shape, np.float32)
    Q[Z.K0:Z.iM] = w.reshape((-1, 1, 1)) * A
    Q[0] = 0.1 * A
    Q[1] = -0.1 * A
    if K > Z.KB and tag is not None:
        Q[Z.iL] = float(tag) * A
    st["Q"] = Q
    st["t"] = np.float32(t)
    st["F"] = np.full_like(A, 0.08)
    st["F2"] = np.zeros_like(A)
    return st, c, eco


def _step(st, c, eco, key=7):
    L = st["A"].shape[0]
    P = Z.params({**c, **eco})
    KA = jnp.asarray(FG.ring_kernel_np(L, 2, float(c["R"]), tuple(c["peaks"])))
    Ko = jnp.asarray(FG.ring_kernel_np(L, 2, 3.0))
    KX = E3.kernels(L, 2)
    s = {k: jnp.asarray(v) for k, v in st.items()}
    s["key"] = jax.random.PRNGKey(key)
    return Z.step(s, P, KA, Ko, KX, do_mut=True, do_hom=False, m=1)


def _p(Q, A):
    mass = float(np.asarray(A).sum()) + 1e-12
    return np.array([float(np.asarray(Q)[k].sum()) / mass for k in range(Q.shape[0])])


def test_il_index_and_default_k():
    assert Z.iL == Z.iM + 1
    assert Z.KB == Z.iM + 1
    assert Z.iL == Z.KB  # first extra channel index


def test_legacy_k_step_runs():
    st, c, eco = _tiny(K=Z.KB, mutn=0.0, mutw=0.2)
    out = _step(st, c, eco)
    assert np.asarray(out["A"]).sum() > 0.1
    assert np.asarray(out["Q"]).shape[0] == Z.KB


def test_lineage_tag_is_not_mutated():
    st, c, eco = _tiny(K=Z.KB + 1, tag=0.7, mutn=0.2, mutw=0.2, mut_bias=0.2)
    p0 = _p(st["Q"], st["A"])
    out = _step(st, c, eco)
    p1 = _p(out["Q"], out["A"])
    assert abs(p1[Z.iL] - p0[Z.iL]) < 1e-5
    # other channels do move under mutation
    assert abs(p1[0] - p0[0]) + abs(p1[Z.K0] - p0[Z.K0]) > 1e-4


def test_lineage_tag_does_not_affect_dynamics():
    """Same genome, tags +1 vs -1: mass, brain outputs, and non-tag p match."""
    st_p, c, eco = _tiny(K=Z.KB + 1, tag=1.0, mutn=0.0, mutw=0.0)
    st_m, _, _ = _tiny(K=Z.KB + 1, tag=-1.0, mutn=0.0, mutw=0.0)
    out_p = _step(st_p, c, eco)
    out_m = _step(st_m, c, eco)
    Ap, Am = np.asarray(out_p["A"]), np.asarray(out_m["A"])
    assert np.allclose(Ap, Am, atol=1e-6)
    Qp, Qm = np.asarray(out_p["Q"]), np.asarray(out_m["Q"])
    assert np.allclose(Qp[:Z.KB], Qm[:Z.KB], atol=1e-6)
    # tags stay apart
    mass = float(Ap.sum()) + 1e-12
    tp = float(Qp[Z.iL].sum()) / mass
    tm = float(Qm[Z.iL].sum()) / mass
    assert tp > 0.5 and tm < -0.5
    # D (feeding / g_A) matches
    assert np.allclose(np.asarray(out_p["D"]), np.asarray(out_m["D"]), atol=1e-6)
