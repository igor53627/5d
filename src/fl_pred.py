"""Part 19: predator–prey. Second mass field AP consumes nearby prey A.

Prey keep the Part 15 micro-brain plus one input (predator scent) and one
output (avoidance taxis). Predators have a matching brain (first input is
prey scent) and an optional fixed pursuit term. Defaults kpred=ka=kp=krep_x=0
and AP=0 leave a prey-only world: no predation and no extra forces.
"""
import numpy as np, jax, jax.numpy as jnp
from functools import partial
import fl_glider as FG
import fl_eco3 as E3
from fl_eco3 import kernels, pool2, up2, _noise, U1_of

NIN, H, NOUT = 4, 3, 4
NW = H * NIN + H + NOUT * H + NOUT          # 31
K0 = 2; KB = K0 + NW                        # 33
ND = 4
IN_F, IN_UB, IN_UFAR, IN_PRED = range(NIN)
OUT_GF, OUT_GR, OUT_GE, OUT_AV = range(NOUT)
BKEYS = ["kb", "kr", "ke", "ce", "mutw", "ws", "rclamp",
         "kpred", "eff", "dec_p", "ka", "kp", "krep_x",
         "pred_pursuit_evolve"]
# kpred/ka/kp/krep_x = 0 → legacy prey-only forces and no mass transfer.
BDEF = dict(kb=0.0, kr=0.0, ke=0.0, ce=0.0, mutw=None, ws=2.0, rclamp=0.0,
            kpred=0.0, eff=0.4, dec_p=0.0, ka=0.0, kp=0.0, krep_x=0.0,
            pred_pursuit_evolve=1.0)


def w1_flat(h, col):
    return h * NIN + col


def b2_flat(out):
    return H * NIN + H + NOUT * H + out


def w2_flat(out, h):
    return H * NIN + H + out * H + h


def params(c):
    P = E3.params(c)
    for k in BKEYS:
        v = c.get(k, BDEF[k]); v = c.get("mutn", 0.002) if (k == "mutw" and v is None) else v; P[k] = jnp.float32(v)
    return P


def make_state(base, c, K=KB, p0=None, seed=0):
    st = E3.make_state(base, c, K=K, p0=p0, seed=seed)
    z = np.zeros_like(st["A"])
    st["D"] = np.zeros((ND,) + st["A"].shape, np.float32)
    st["AP"] = z.astype(np.float32)
    st["QP"] = np.zeros((K,) + st["A"].shape, np.float32)
    st["xp"] = np.array(st["x"], np.float32, copy=True)
    st["yp"] = np.array(st["y"], np.float32, copy=True)
    st["zp"] = np.array(st["z"], np.float32, copy=True)
    st["cp"] = np.array(st["c"], np.float32, copy=True)
    st["DP"] = np.zeros((ND,) + st["A"].shape, np.float32)
    return st


def unpack(w, ws):
    i = 0; W1 = w[i:i + H * NIN].reshape((H, NIN) + w.shape[1:]) * ws; i += H * NIN
    b1 = w[i:i + H] * ws; i += H; W2 = w[i:i + NOUT * H].reshape((NOUT, H) + w.shape[1:]) * ws; i += NOUT * H
    b2 = w[i:i + NOUT] * ws
    return W1, b1, W2, b2


def brain(w, s, ws):
    xp = jnp if isinstance(w, jax.Array) else np
    W1, b1, W2, b2 = unpack(w, ws)
    h = xp.tanh((W1 * s[None]).sum(1) + b1)
    return xp.tanh((W2 * h[None]).sum(1) + b2)


def _norm_kernel(Kk):
    """Frequency kernel whose spatial kernel sums to 1 (divide by DC)."""
    dc = Kk.reshape(-1)[0]
    return Kk / (jnp.real(dc) + jnp.float32(1e-12))


def _move(A, x, y, z, c, Q, s_in, v_extra, P, KA, Ko, KX, wconv, conv, transport, do_hom, m, N):
    """One species: soliton chemistry, brain taxis, advection, homogenization.

    v_extra is a list of N extra velocity fields (already scaled).
    Returns An, xn, yn, zn, pn, o, v, vb.
    """
    shape = A.shape; K = Q.shape[0]
    p = jnp.where(A[None] > 1e-6, jnp.clip(Q / jnp.maximum(A[None], 1e-6), -1.0, 1.0), 0.0)
    U = conv(A, KA)
    G = 2.0 * jnp.exp(-((U - (P["mu"] + P["pmu"] * p[1])) ** 2) / (2 * P["sigma"] ** 2)) - 1.0
    dt = P["dt"]
    ex, ey = conv(x, Ko) - x, conv(y, Ko) - y
    fx, fy = P["eps"] * ex, P["eps"] * ey
    f = lambda x, y, z: (-y - z + fx, x + P["a"] * y + fy, P["b"] + z * (x - c))
    k1 = f(x, y, z); k2 = f(x + .5*dt*k1[0], y + .5*dt*k1[1], z + .5*dt*k1[2])
    k3 = f(x + .5*dt*k2[0], y + .5*dt*k2[1], z + .5*dt*k2[2]); k4 = f(x + dt*k3[0], y + dt*k3[1], z + dt*k3[2])
    xn = x + dt/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]); yn = y + dt/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    zn = jnp.clip(z + dt/6*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2]), -50.0, 80.0)
    S = conv(jnp.exp(-(ex*ex + ey*ey) / P["sigA"] ** 2), Ko)
    fb = P["beta"] * (S - P["s0"]) - P["gam"] * (c - P["c_lo"]) / (P["c_hi"] - P["c_lo"])
    r = jnp.sqrt(x*x + y*y) + 1e-6; cs, sn = x / r, y / r
    UG = G + fb
    Ufar = wconv(A, KX["far"])
    o = brain(p[K0:], s_in, P["ws"])
    o = o.at[OUT_GR].set(jnp.where(P["rclamp"] > 0, jnp.maximum(o[OUT_GR], 0.0), o[OUT_GR]))
    gU = [FG._grad(Ufar, k) for k in range(N)]
    al = jnp.clip((A / P["theta"]) ** 2, 0.0, 1.0)
    v = [((1 - al) * FG._grad(UG, k) - al * FG._grad(A, k) - P["krep"] * gU[k]
          + P["chi"] * (cs * FG._grad(sn, k) - sn * FG._grad(cs, k))) * P["dtA"] * 10.0
         + v_extra[k] for k in range(N)]
    mag = jnp.sqrt(sum(vi * vi for vi in v)) + 1e-12; fl = jnp.minimum(1.0, (0.45 / np.sqrt(N)) / mag)
    v = [vi * fl for vi in v]
    An = jnp.maximum(FG._advect_muscl(A, v, 1.0), 0.0)
    if transport:
        Qn = jnp.stack([FG._advect_muscl(Q[k], v, 1.0) for k in range(K)])
        pn = jnp.where(An[None] > 1e-6, jnp.clip(Qn / jnp.maximum(An[None], 1e-6), -1.0, 1.0), 0.0)
    else:
        pn = p
    if do_hom:
        hom = P["hom"] if m == 1 else 1.0 - (1.0 - P["hom"]) ** m
        Ab = wconv(An, KX["blur"]); Qb = jnp.stack([wconv(pn[k] * An, KX["blur"]) for k in range(K)])
        ps = Qb / jnp.maximum(Ab[None], 1e-6)
        pn = jnp.where(Ab[None] > 1e-4, (1 - hom) * pn + hom * ps, pn)
    return An, xn, yn, zn, pn, o, v, fl


def _mutate(pn, An2, P, key, nshape, KX, fnoise, coarse, N, m, do_mut):
    K = pn.shape[0]
    if do_mut:
        nz = jnp.stack([_noise(jax.random.fold_in(key, k), nshape, KX["blur"], fnoise) for k in range(K)])
        nz = nz / (jnp.std(nz, axis=tuple(range(1, N + 1)), keepdims=True) + 1e-12)
        if coarse: nz = jnp.stack([up2(nz[k]) for k in range(K)])
        amp = jnp.concatenate([jnp.full((K0,), P["mutn"]), jnp.full((K - K0,), P["mutw"])]).reshape((K,) + (1,) * N)
        pn = jnp.clip(pn + amp * np.sqrt(m) * nz * (An2[None] > 0.05), -1.0, 1.0)
    pn = jnp.where(An2[None] > 1e-6, pn, 0.0)
    return pn


def step(st, P, KA, Ko, KX, transport=True, coarse=False, fnoise=False, do_mut=True, do_hom=True, m=1):
    A, x, y, z, c, F, Q = st["A"], st["x"], st["y"], st["z"], st["c"], st["F"], st["Q"]
    AP, xp, yp, zp, cp, QP = st["AP"], st["xp"], st["yp"], st["zp"], st["cp"], st["QP"]
    N = A.ndim; shape = A.shape; K = Q.shape[0]
    conv = lambda G, Kk: jnp.fft.irfftn(jnp.fft.rfftn(G) * Kk, s=shape)
    if coarse:
        cs_ = tuple(s // 2 for s in shape); wconv = lambda G, Kk: up2(jnp.fft.irfftn(jnp.fft.rfftn(pool2(G)) * Kk, s=cs_)); nshape = cs_
    else:
        wconv = conv; nshape = shape
    Ke = _norm_kernel(KX["blur"])
    # --- senses (pre-advection) ---
    Ft = wconv(F, KX["blur"]) / P["F0"]
    Ub0 = wconv(A, KX["big"])
    Ufar_a = wconv(A, KX["far"])
    Ufar_p = wconv(AP, KX["far"])
    Pt = wconv(AP, KX["blur"]) / P["U1"]
    At = wconv(A, KX["blur"]) / P["U1"]
    pa = jnp.where(A[None] > 1e-6, jnp.clip(Q / jnp.maximum(A[None], 1e-6), -1.0, 1.0), 0.0)
    pp = jnp.where(AP[None] > 1e-6, jnp.clip(QP / jnp.maximum(AP[None], 1e-6), -1.0, 1.0), 0.0)
    s_prey = jnp.stack([Ft - 1.0, Ub0 / P["U1"] - 1.0, Ufar_a / P["U1"], Pt])
    UbP0 = wconv(AP, KX["big"])
    s_pred = jnp.stack([At, UbP0 / P["U1"] - 1.0, Ufar_p / P["U1"], jnp.zeros_like(A)])
    o_pre_p = brain(pa[K0:], s_prey, P["ws"])
    o_pre_p = o_pre_p.at[OUT_GR].set(jnp.where(P["rclamp"] > 0, jnp.maximum(o_pre_p[OUT_GR], 0.0), o_pre_p[OUT_GR]))
    o_pre_h = brain(pp[K0:], s_pred, P["ws"])
    o_pre_h = o_pre_h.at[OUT_GR].set(jnp.where(P["rclamp"] > 0, jnp.maximum(o_pre_h[OUT_GR], 0.0), o_pre_h[OUT_GR]))
    gF = [FG._grad(Ft, k) for k in range(N)]
    gU_a = [FG._grad(Ufar_a, k) for k in range(N)]
    gP = [FG._grad(Pt, k) for k in range(N)]
    gA = [FG._grad(At, k) for k in range(N)]
    gU_p = [FG._grad(Ufar_p, k) for k in range(N)]
    scale = P["dtA"] * 10.0
    v_prey = [(P["kb"] * o_pre_p[OUT_GF] * gF[k] - P["kr"] * o_pre_p[OUT_GR] * gU_a[k]
               - P["ka"] * o_pre_p[OUT_AV] * gP[k] - P["krep_x"] * gU_p[k]) * scale for k in range(N)]
    # pred_pursuit_evolve defaults to 1, matching Parts 19/19b (shared kb=1).
    # Set it to 0 to keep only the fixed kp term, without changing prey kb.
    v_pred = [(P["kb"] * P["pred_pursuit_evolve"] * o_pre_h[OUT_GF] * gA[k]
               - P["kr"] * o_pre_h[OUT_GR] * gU_p[k]
               + P["kp"] * gA[k] - P["krep_x"] * gU_a[k]) * scale for k in range(N)]
    An, xn, yn, zn, pn, oA, vA, flA = _move(
        A, x, y, z, c, Q, s_prey, v_prey, P, KA, Ko, KX, wconv, conv, transport, do_hom, m, N)
    Pn, xpn, ypn, zpn, ppn, oP, vP, flP = _move(
        AP, xp, yp, zp, cp, QP, s_pred, v_pred, P, KA, Ko, KX, wconv, conv, transport, do_hom, m, N)
    # --- food (prey only) and predation ---
    Ub = wconv(An, KX["big"])
    gE = oA[OUT_GE]
    decl = P["dec"] * jnp.exp(P["ks"] * jnp.maximum(Ub / P["U1"] - 1.0, 0.0)) * (1.0 + P["ce"] * jnp.maximum(gE, 0.0))
    t = st.get("t", jnp.float32(0.0))
    x0 = jnp.arange(shape[0], dtype=jnp.float32).reshape((-1,) + (1,) * (N - 1))
    popt = P["na"] * jnp.sin(2 * np.pi * x0 / P["nl"]) + P["sa"] * jnp.sin(2 * np.pi * t / P["nt"])
    rate = P["keat"] * jnp.exp(P["pk"] * pn[0] - P["kn"] * (pn[0] - popt) ** 2 + P["ke"] * gE)
    F2 = st["F2"]; w1 = ((1 + pn[0]) / 2) ** P["g2"]; w2 = ((1 - pn[0]) / 2) ** P["g2"]
    e1 = jnp.minimum(jnp.where(P["res2"] > 0, rate * w1, rate) * An * F, F)
    e2 = jnp.where(P["res2"] > 0, jnp.minimum(rate * w2 * An * F2, F2), 0.0)
    eat = e1 + e2
    F2n = F2 + P["rF"] * (P["F0"] - F2) - e2
    Fn = F + P["rF"] * (P["F0"] - F) - e1
    # Neighborhood predation. Ke is DC-normalized; on the fine grid, sum(A*P_s)
    # equals sum(AP*A_s) when the cap is slack. coarse=True (pool2 then up2)
    # can break that identity slightly, and the min cap can bind, so scale the
    # predator-side field to make total gain = eff * actual prey loss.
    P_s = wconv(Pn, Ke); A_s = wconv(An, Ke)
    loss = jnp.minimum(P["kpred"] * An * P_s, An)
    raw_gain = P["kpred"] * Pn * A_s
    gain = raw_gain * (P["eff"] * loss.sum() / jnp.maximum(raw_gain.sum(), jnp.float32(1e-30)))
    UbP = wconv(Pn, KX["big"])
    decl_p = P["dec_p"] * jnp.exp(P["ks"] * jnp.maximum(UbP / P["U1"] - 1.0, 0.0))
    An2 = jnp.maximum(An + eat - decl * An - loss, 0.0)
    Pn2 = jnp.maximum(Pn + gain - decl_p * Pn, 0.0)
    key, sk1 = jax.random.split(st["key"]); sk1, sk2 = jax.random.split(sk1)
    pn = _mutate(pn, An2, P, sk1, nshape, KX, fnoise, coarse, N, m, do_mut)
    ppn = _mutate(ppn, Pn2, P, sk2, nshape, KX, fnoise, coarse, N, m, do_mut)
    ct = P["c_lo"] + (P["c_hi"] - P["c_lo"]) * jnp.clip(An2, 0.0, 1.0)
    cn = c + (P["dt"] / P["tau_c"]) * (ct - c)
    ctp = P["c_lo"] + (P["c_hi"] - P["c_lo"]) * jnp.clip(Pn2, 0.0, 1.0)
    cpn = cp + (P["dt"] / P["tau_c"]) * (ctp - cp)
    # D: food eaten, prey loss, predator gain, A·g_Av. DP: gain, loss-scent, AP·g_F, AP.
    D = jnp.stack([eat, loss, gain, A * o_pre_p[OUT_AV]])
    DP = jnp.stack([gain, loss, AP * o_pre_h[OUT_GF], AP])
    return dict(A=An2, x=xn, y=yn, z=zn, c=cn, F=Fn, Q=pn * An2[None], key=key,
                F2=F2n, t=t + 1.0, D=D, AP=Pn2, QP=ppn * Pn2[None],
                xp=xpn, yp=ypn, zp=zpn, cp=cpn, DP=DP)


@partial(jax.jit, static_argnames=("n", "transport", "coarse", "fnoise", "m"))
def run_chunk(st, P, KA, Ko, KX, n, transport=True, coarse=False, fnoise=False, m=1):
    mk = lambda full: jax.vmap(partial(step, transport=transport, coarse=coarse, fnoise=fnoise, do_mut=full, do_hom=full, m=m), in_axes=(0, 0, 0, None, None))
    full = mk(True)
    if m == 1:
        return jax.lax.fori_loop(0, n, lambda i, s: full(s, P, KA, Ko, KX), st)
    cheap = mk(False); assert n % m == 0
    def block(i, s):
        s = jax.lax.fori_loop(0, m - 1, lambda j, s_: cheap(s_, P, KA, Ko, KX), s)
        return full(s, P, KA, Ko, KX)
    return jax.lax.fori_loop(0, n // m, block, st)
