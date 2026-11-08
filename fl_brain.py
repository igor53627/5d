"""Часть 15: «микромозг». Тот же шаг, что fl_eco3 (all5), плюс наследуемые веса крошечной сети как дополнительные каналы Q
(переносятся MUSCL вместе с массой, гомогенизируются по телу, мутируют на масштабе тела). Каналы: 0 — p0, 1 — p1, 2.. — веса.
Сеть (на клетку, эквивариантна к поворотам): входы s = [F̃−1, Ub/U1−1, Ufar/U1] (F̃ — корм, размытый по телу, /F0),
скрытый слой H=3 (tanh), выходы o = [g_F, g_R, g_E] ∈ (−1,1):
  v += dtA·10·(kb·g_F·∇F̃ − kr·g_R·∇Ufar)   — хемотаксис и управляемое отталкивание;
  скорость питания × exp(ke·g_E), смертность × (1 + ce·max(g_E,0))   — (ke=0 — выключено).
Диагностика D (не переносится, пересчитывается каждый шаг): D0 = съедено, D1 = A·(v·ĝ), D2 = A·|v|, D3 = A·(v_brain·ĝ), ĝ = ∇F̃/|∇F̃|."""
import numpy as np, jax, jax.numpy as jnp
from functools import partial
import fl_glider as FG
import fl_eco3 as E3
from fl_eco3 import kernels, pool2, up2, _noise, U1_of
NIN, H, NOUT = 3, 3, 3
NW = H * NIN + H + NOUT * H + NOUT          # 24
K0 = 2; KB = K0 + NW                        # 26 каналов
ND = 4
BKEYS = ["kb", "kr", "ke", "ce", "mutw", "ws", "rclamp"]
BDEF = dict(kb=0.0, kr=0.0, ke=0.0, ce=0.0, mutw=None, ws=2.0, rclamp=0.0)   # Часть 16: rclamp=1 → g_R ≥ 0 (нельзя ослабить отталкивание)

def params(c):
    P = E3.params(c)
    for k in BKEYS:
        v = c.get(k, BDEF[k]); v = c.get("mutn", 0.002) if (k == "mutw" and v is None) else v; P[k] = jnp.float32(v)
    return P

def make_state(base, c, K=KB, p0=None, seed=0):
    st = E3.make_state(base, c, K=K, p0=p0, seed=seed); st["D"] = np.zeros((ND,) + st["A"].shape, np.float32); return st

def unpack(w, ws):
    """w: (NW, ...) → W1 (H,NIN,...), b1 (H,...), W2 (NOUT,H,...), b2 (NOUT,...), масштаб ws."""
    i = 0; W1 = w[i:i + H * NIN].reshape((H, NIN) + w.shape[1:]) * ws; i += H * NIN
    b1 = w[i:i + H] * ws; i += H; W2 = w[i:i + NOUT * H].reshape((NOUT, H) + w.shape[1:]) * ws; i += NOUT * H
    b2 = w[i:i + NOUT] * ws
    return W1, b1, W2, b2

def brain(w, s, ws):
    """w: (NW, ...), s: (NIN, ...) → o (NOUT, ...). Работает и на полях, и на векторах весов особей (numpy/jax)."""
    xp = jnp if isinstance(w, jax.Array) else np
    W1, b1, W2, b2 = unpack(w, ws)
    h = xp.tanh((W1 * s[None]).sum(1) + b1)
    return xp.tanh((W2 * h[None]).sum(1) + b2)

def step(st, P, KA, Ko, KX, transport=True, coarse=False, fnoise=False, do_mut=True, do_hom=True, m=1):
    A, x, y, z, c, F, Q = st["A"], st["x"], st["y"], st["z"], st["c"], st["F"], st["Q"]
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
    # --- мозг ---
    Ft = wconv(F, KX["blur"]) / P["F0"]
    Ub0 = wconv(A, KX["big"])
    s_in = jnp.stack([Ft - 1.0, Ub0 / P["U1"] - 1.0, Ufar / P["U1"]])
    o = brain(p[K0:], s_in, P["ws"])
    o = o.at[1].set(jnp.where(P["rclamp"] > 0, jnp.maximum(o[1], 0.0), o[1]))
    gF = [FG._grad(Ft, k) for k in range(N)]; gU = [FG._grad(Ufar, k) for k in range(N)]
    vb = [(P["kb"] * o[0] * gF[k] - P["kr"] * o[1] * gU[k]) * P["dtA"] * 10.0 for k in range(N)]
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
    Ub = wconv(An, KX["big"])
    gE = o[2]
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
    An2 = jnp.maximum(An + eat - decl * An, 0.0)
    key, sk = jax.random.split(st["key"])
    if do_mut:
        nz = jnp.stack([_noise(jax.random.fold_in(sk, k), nshape, KX["blur"], fnoise) for k in range(K)])
        nz = nz / (jnp.std(nz, axis=tuple(range(1, N + 1)), keepdims=True) + 1e-12)
        if coarse: nz = jnp.stack([up2(nz[k]) for k in range(K)])
        amp = jnp.concatenate([jnp.full((K0,), P["mutn"]), jnp.full((K - K0,), P["mutw"])]).reshape((K,) + (1,) * N)
        pn = jnp.clip(pn + amp * np.sqrt(m) * nz * (An2[None] > 0.05), -1.0, 1.0)
    pn = jnp.where(An2[None] > 1e-6, pn, 0.0)
    ct = P["c_lo"] + (P["c_hi"] - P["c_lo"]) * jnp.clip(An2, 0.0, 1.0)
    cn = c + (dt / P["tau_c"]) * (ct - c)
    gn = jnp.sqrt(sum(g * g for g in gF)) + 1e-9; vm = jnp.sqrt(sum(vi * vi for vi in v))
    D = jnp.stack([eat, A * sum(v[k] * gF[k] for k in range(N)) / gn, A * vm, A * sum(vb[k] * fl * gF[k] for k in range(N)) / gn])
    return dict(A=An2, x=xn, y=yn, z=zn, c=cn, F=Fn, Q=pn * An2[None], key=key, F2=F2n, t=t + 1.0, D=D)

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
