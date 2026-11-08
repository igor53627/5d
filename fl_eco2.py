"""Часть 10: fl_eco + (a) анти-слияние: отталкивание от чужой массы (кольцо 8–16 клеток) и штраф за размер (распад растёт с локальной массой),
(b) параметры на уровне особи: быстрое массо-взвешенное усреднение p внутри тела (ядро радиуса ~R) и мутации, коррелированные на масштабе тела."""
import numpy as np, jax, jax.numpy as jnp
from functools import partial
import fl_glider as FG
from fl_eco import make_state
EKEYS = ["keat", "dec", "rF", "F0", "pmu", "pk", "krep", "ks", "U1", "hom", "mutn"]
EDEF = dict(keat=0.0, dec=0.0, rF=0.0, F0=1.0, pmu=0.02, pk=0.0, krep=0.0, ks=0.0, U1=1.0, hom=0.0, mutn=0.0)

def params(c):
    P = FG.params(c); P.update({k: jnp.float32(c.get(k, EDEF[k])) for k in EKEYS}); return P

def kernels(L, N, R=6.0):
    return dict(far=jnp.asarray(FG.ring_kernel_np(L, N, 16.0, (0.0, 1.0))), big=jnp.asarray(FG.ring_kernel_np(L, N, 12.0, (1.0,))),
                blur=jnp.asarray(FG.ring_kernel_np(L, N, R, (1.0,))))

def step(st, P, KA, Ko, KX, transport=True, fast=False):
    A, x, y, z, c, F, Q = st["A"], st["x"], st["y"], st["z"], st["c"], st["F"], st["Q"]
    N = A.ndim; shape = A.shape; K = Q.shape[0]
    conv = lambda G, Kk: jnp.fft.irfftn(jnp.fft.rfftn(G) * Kk, s=shape)
    iconv = lambda FG_, Kk: jnp.fft.irfftn(FG_ * Kk, s=shape)       # fast: свёртка по уже посчитанному спектру
    FA = jnp.fft.rfftn(A) if fast else None
    p = jnp.where(A[None] > 1e-6, jnp.clip(Q / jnp.maximum(A[None], 1e-6), -1.0, 1.0), 0.0)
    U = iconv(FA, KA) if fast else conv(A, KA)
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
    Ufar = iconv(FA, KX["far"]) if fast else conv(A, KX["far"])                       # «чужая» масса на расстоянии 8–16 клеток
    al = jnp.clip((A / P["theta"]) ** 2, 0.0, 1.0)
    v = [((1 - al) * FG._grad(UG, k) - al * FG._grad(A, k) - P["krep"] * FG._grad(Ufar, k)
          + P["chi"] * (cs * FG._grad(sn, k) - sn * FG._grad(cs, k))) * P["dtA"] * 10.0 for k in range(N)]
    mag = jnp.sqrt(sum(vi * vi for vi in v)) + 1e-12; fl = jnp.minimum(1.0, (0.45 / np.sqrt(N)) / mag)
    v = [vi * fl for vi in v]
    An = jnp.maximum(FG._advect_muscl(A, v, 1.0), 0.0)
    if transport:
        Qn = jnp.stack([FG._advect_muscl(Q[k], v, 1.0) for k in range(K)])
        pn = jnp.where(An[None] > 1e-6, jnp.clip(Qn / jnp.maximum(An[None], 1e-6), -1.0, 1.0), 0.0)
    else:
        pn = p
    # (b) усреднение p внутри тела (массо-взвешенное)
    FAn = jnp.fft.rfftn(An) if fast else None
    Ab = iconv(FAn, KX["blur"]) if fast else conv(An, KX["blur"]); Qb = jnp.stack([conv(pn[k] * An, KX["blur"]) for k in range(K)])
    ps = Qb / jnp.maximum(Ab[None], 1e-6)
    pn = jnp.where(Ab[None] > 1e-4, (1 - P["hom"]) * pn + P["hom"] * ps, pn)
    # пища, рост; (a) штраф за размер: распад растёт с локальной массой (U_big/U1)
    Ub = iconv(FAn, KX["big"]) if fast else conv(An, KX["big"])
    decl = P["dec"] * jnp.exp(P["ks"] * jnp.maximum(Ub / P["U1"] - 1.0, 0.0))
    eat = jnp.minimum(P["keat"] * jnp.exp(P["pk"] * pn[0]) * An * F, F)
    An2 = jnp.maximum(An + eat - decl * An, 0.0)
    Fn = F + P["rF"] * (P["F0"] - F) - eat
    # мутации на масштабе тела: шум, сглаженный ядром blur и нормированный на std=1
    key, sk = jax.random.split(st["key"])
    nz = jnp.stack([conv(jax.random.normal(jax.random.fold_in(sk, k), shape), KX["blur"]) for k in range(K)])
    nz = nz / (jnp.std(nz, axis=tuple(range(1, N + 1)), keepdims=True) + 1e-12)
    pn = jnp.clip(pn + P["mutn"] * nz * (An2[None] > 0.05), -1.0, 1.0)
    pn = jnp.where(An2[None] > 1e-6, pn, 0.0)
    ct = P["c_lo"] + (P["c_hi"] - P["c_lo"]) * jnp.clip(An2, 0.0, 1.0)
    cn = c + (dt / P["tau_c"]) * (ct - c)
    return dict(A=An2, x=xn, y=yn, z=zn, c=cn, F=Fn, Q=pn * An2[None], key=key)

@partial(jax.jit, static_argnames=("n", "transport"))
def run_chunk(st, P, KA, Ko, KX, n, transport=True):
    vs = jax.vmap(partial(step, transport=transport), in_axes=(0, 0, 0, None, None))
    return jax.lax.fori_loop(0, n, lambda i, s: vs(s, P, KA, Ko, KX), st)

def U1_of(A, L, N):
    return float(np.asarray(jnp.fft.irfftn(jnp.fft.rfftn(jnp.asarray(A)) * kernels(L, N)["big"], s=A.shape)).max())

@partial(jax.jit, static_argnames=("n", "transport"))
def run_chunk_fast(st, P, KA, Ko, KX, n, transport=True):
    vs = jax.vmap(partial(step, transport=transport, fast=True), in_axes=(0, 0, 0, None, None))
    return jax.lax.fori_loop(0, n, lambda i, s: vs(s, P, KA, Ko, KX), st)
