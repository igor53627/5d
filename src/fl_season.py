"""Часть 18: «сезоны» в 4D. На базе fl_poison: два корма A (поле F) и B (поле F2), оба восстанавливаются (rG) к взаимодополняющим
движущимся пятнам: A* = F0·relu(c), B* = F0·relu(−c), c = cos(2πx0/λ+ωt)·cos(2πx1/λ−ωt).
Сезон σ(t) = sign(sin(2πt/Ts)): при σ>0 A хороший, B — яд (стоит cp массы за единицу), при σ<0 наоборот.
Мозг: входы s = [Ã, B̃, R, h, cue·σ, Ub/U1−1, Ufar/U1] (Ã = blur(A)/F0, B̃ = blur(B)/F0,
  R — «награда»: размытый по телу чистый прирост от еды за прошлый шаг на единицу массы, /(keat·F0);
  h — наследуемая переменная памяти (канал Q, переносится MUSCL и гомогенизируется по телу, как веса; не мутирует)).
  H=3 (tanh), выходы o = [g_A, g_B, g_R, o_h]:  v += dtA·10·(kb·(g_A∇Ã + g_B∇B̃) − kr·g_R∇Ufar);  h ← h + α(o_h − h) каждый шаг (если mem=1).
mem=0 — без памяти (вход h=0, h не обновляется; вход R остаётся — чистая абляция памяти). cue=1 — сезон дан напрямую.
D: D0 = съедено A, D1 = съедено B, D2 = R, D3 = A·g_A, D4 = A·g_B (реализованные усиления)."""
import numpy as np, jax, jax.numpy as jnp
from functools import partial
import fl_glider as FG
import fl_eco3 as E3
from fl_eco3 import kernels, pool2, up2, _noise, U1_of
NIN, H, NOUT = 7, 3, 4
NW = H * NIN + H + NOUT * H + NOUT          # 40
K0 = 2; iM = K0 + NW; KB = iM + 1           # 43 канала (последний — память h)
ND = 5
BKEYS = ["kb", "kr", "mutw", "ws", "mem", "cue", "alpha", "rG", "cp", "lam", "om", "Ts", "smooth"]
BDEF = dict(kb=0.0, kr=0.0, mutw=None, ws=2.0, mem=1.0, cue=0.0, alpha=0.01, rG=0.001, cp=2.0, lam=28.0, om=2 * np.pi / 6000, Ts=6000.0, smooth=0.0)

def params(c):
    P = E3.params(c)
    for k in BKEYS:
        v = c.get(k, BDEF[k]); v = c.get("mutn", 0.002) if (k == "mutw" and v is None) else v; P[k] = jnp.float32(v)
    return P

def pattern(shape, P, t, xp=jnp):
    N = len(shape)
    x0 = xp.arange(shape[0], dtype=xp.float32).reshape((-1,) + (1,) * (N - 1))
    x1 = xp.arange(shape[1], dtype=xp.float32).reshape((1, -1) + (1,) * (N - 2))
    return xp.cos(2 * np.pi * x0 / P["lam"] + P["om"] * t) * xp.cos(2 * np.pi * x1 / P["lam"] - P["om"] * t)

def targets(cc, P, xp=jnp):
    """smooth=0: A* = F0·relu(c), B* = F0·relu(−c);  smooth=1: A* = F0(1+c)/2, B* = F0(1−c)/2 (Часть 18, после пилотов: корма больше)."""
    a0, b0 = xp.maximum(cc, 0.0), xp.maximum(-cc, 0.0); a1, b1 = (1 + cc) / 2, (1 - cc) / 2
    sm = P["smooth"] > 0
    return P["F0"] * xp.where(sm, a1, a0), P["F0"] * xp.where(sm, b1, b0)

def season(t, Ts, xp=jnp): return xp.where(xp.sin(2 * np.pi * t / Ts) >= 0, 1.0, -1.0)

def make_state(base, c, K=KB, p0=None, seed=0):
    st = E3.make_state(base, c, K=K, p0=p0, seed=seed)
    Pd = {k: c.get(k, BDEF[k]) for k in BKEYS}; F0 = c.get("F0", 1.0); cc = np.broadcast_to(pattern(st["A"].shape, Pd, 0.0, np), st["A"].shape)
    Pd["F0"] = F0; ta, tb = targets(cc, Pd, np); st["F"] = ta.astype(np.float32); st["F2"] = tb.astype(np.float32)
    st["R"] = np.zeros_like(st["A"]); st["D"] = np.zeros((ND,) + st["A"].shape, np.float32); return st

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

def step(st, P, KA, Ko, KX, transport=True, coarse=False, fnoise=False, do_mut=True, do_hom=True, m=1):
    A, x, y, z, c, F, Q = st["A"], st["x"], st["y"], st["z"], st["c"], st["F"], st["Q"]
    FB = st["F2"]; Rw = st["R"]
    N = A.ndim; shape = A.shape; K = Q.shape[0]
    conv = lambda G, Kk: jnp.fft.irfftn(jnp.fft.rfftn(G) * Kk, s=shape)
    if coarse:
        cs_ = tuple(s // 2 for s in shape); wconv = lambda G, Kk: up2(jnp.fft.irfftn(jnp.fft.rfftn(pool2(G)) * Kk, s=cs_)); nshape = cs_
    else:
        wconv = conv; nshape = shape
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
    t = st.get("t", jnp.float32(0.0)); sg = season(t, P["Ts"])
    # --- мозг ---
    At = wconv(F, KX["blur"]) / P["F0"]; Bt = wconv(FB, KX["blur"]) / P["F0"]
    hm = p[iM] * P["mem"]
    Ub0 = wconv(A, KX["big"])
    s_in = jnp.stack([At, Bt, Rw, hm, P["cue"] * sg * jnp.ones_like(A), Ub0 / P["U1"] - 1.0, Ufar / P["U1"]])
    o = brain(p[K0:iM], s_in, P["ws"])
    gA = [FG._grad(At, k) for k in range(N)]; gB = [FG._grad(Bt, k) for k in range(N)]; gU = [FG._grad(Ufar, k) for k in range(N)]
    vb = [(P["kb"] * (o[0] * gA[k] + o[1] * gB[k]) - P["kr"] * o[2] * gU[k]) * P["dtA"] * 10.0 for k in range(N)]
    al = jnp.clip((A / P["theta"]) ** 2, 0.0, 1.0)
    v = [((1 - al) * FG._grad(UG, k) - al * FG._grad(A, k) - P["krep"] * FG._grad(Ufar, k)
          + P["chi"] * (cs * FG._grad(sn, k) - sn * FG._grad(cs, k))) * P["dtA"] * 10.0 + vb[k] for k in range(N)]
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
    pn = pn.at[iM].set(jnp.where(P["mem"] > 0, pn[iM] + P["alpha"] * (o[3] - pn[iM]), pn[iM]))     # обновление памяти
    Ub = wconv(An, KX["big"])
    decl = P["dec"] * jnp.exp(P["ks"] * jnp.maximum(Ub / P["U1"] - 1.0, 0.0))
    rate = P["keat"] * jnp.exp(P["pk"] * pn[0])
    eA = jnp.minimum(rate * An * F, F); eB = jnp.minimum(rate * An * FB, FB)
    cc = pattern(shape, P, t)
    ta, tb = targets(cc, P)
    Fn = F + P["rG"] * (ta - F) - eA
    FBn = FB + P["rG"] * (tb - FB) - eB
    gain = jnp.where(sg > 0, eA - P["cp"] * eB, eB - P["cp"] * eA)
    An2 = jnp.maximum(An + gain - decl * An, 0.0)
    Rn = wconv(gain, KX["blur"]) / jnp.maximum(wconv(An2, KX["blur"]), 1e-4) / (P["keat"] * P["F0"])
    Rn = jnp.clip(Rn, -3.0, 3.0)
    key, sk = jax.random.split(st["key"])
    if do_mut:
        nz = jnp.stack([_noise(jax.random.fold_in(sk, k), nshape, KX["blur"], fnoise) for k in range(K)])
        nz = nz / (jnp.std(nz, axis=tuple(range(1, N + 1)), keepdims=True) + 1e-12)
        if coarse: nz = jnp.stack([up2(nz[k]) for k in range(K)])
        amp = jnp.concatenate([jnp.full((K0,), P["mutn"]), jnp.full((NW,), P["mutw"]), jnp.zeros((1,))]).reshape((K,) + (1,) * N)
        pn = jnp.clip(pn + amp * np.sqrt(m) * nz * (An2[None] > 0.05), -1.0, 1.0)
    pn = jnp.where(An2[None] > 1e-6, pn, 0.0)
    ct = P["c_lo"] + (P["c_hi"] - P["c_lo"]) * jnp.clip(An2, 0.0, 1.0)
    cn = c + (dt / P["tau_c"]) * (ct - c)
    D = jnp.stack([eA, eB, A * Rw, A * o[0], A * o[1]])
    return dict(A=An2, x=xn, y=yn, z=zn, c=cn, F=Fn, Q=pn * An2[None], key=key, F2=FBn, R=Rn, t=t + 1.0, D=D)

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
