"""Часть 9: Flow-Lenia с локальными наследуемыми параметрами (переносятся с массой) и источником массы (пища).
Состояние: A, x, y, z, c (как раньше) + F (пища) + Q[K] = A·p (массо-взвешенные локальные параметры) + key.
p[0] — эффективность питания keat·e^{pk·p0}, p[1] — сдвиг μ (Δμ = pmu·p1).
Пища: F' = rF·(F0−F) − eat;  eat = keat·A·F;  A' += eat − dec·A (распад ограничивает общую массу).
Мутации: пространственно-коррелированный шум (свёртка с ядром Ko) в p с амплитудой mut, только там, где A>0.05.
transport=False — контроль: p НЕ переносится потоком (Q пересчитывается из неподвижного поля p)."""
import numpy as np, jax, jax.numpy as jnp
from functools import partial
import fl_glider as FG

EKEYS = ["keat", "dec", "rF", "F0", "mut", "pmu", "psig", "pk"]
EDEF = dict(keat=0.0, dec=0.0, rF=0.0, F0=1.0, mut=0.0, pmu=0.02, psig=0.15, pk=0.0)

def params(c):
    P = FG.params(c); P.update({k: jnp.float32(c.get(k, EDEF[k])) for k in EKEYS}); return P

def step(st, P, KA, Ko, transport=True):
    A, x, y, z, c, F, Q = st["A"], st["x"], st["y"], st["z"], st["c"], st["F"], st["Q"]
    N = A.ndim; shape = A.shape; K = Q.shape[0]
    conv = lambda G, Kk: jnp.fft.irfftn(jnp.fft.rfftn(G) * Kk, s=shape)
    p = jnp.where(A[None] > 1e-6, jnp.clip(Q / jnp.maximum(A[None], 1e-6), -1.0, 1.0), 0.0)   # без разбавления на краях
    U = conv(A, KA)
    mu_l = P["mu"] + P["pmu"] * p[1]; sg_l = P["sigma"]    # p1 → Δμ (морфология/скорость)
    G = 2.0 * jnp.exp(-((U - mu_l) ** 2) / (2 * sg_l ** 2)) - 1.0
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
    vs = [P["chi"] * (cs * FG._grad(sn, k) - sn * FG._grad(cs, k)) for k in range(N)]
    UG = G + fb
    al = jnp.clip((A / P["theta"]) ** 2, 0.0, 1.0)
    v = [((1 - al) * FG._grad(UG, k) - al * FG._grad(A, k) + vs[k]) * P["dtA"] * 10.0 for k in range(N)]
    mag = jnp.sqrt(sum(vi * vi for vi in v)) + 1e-12; fl = jnp.minimum(1.0, (0.45 / np.sqrt(N)) / mag)
    v = [vi * fl for vi in v]
    An = jnp.maximum(FG._advect_muscl(A, v, 1.0), 0.0)
    if transport:
        Qn = jnp.stack([FG._advect_muscl(Q[k], v, 1.0) for k in range(K)])
        pn = jnp.where(An[None] > 1e-6, jnp.clip(Qn / jnp.maximum(An[None], 1e-6), -1.0, 1.0), 0.0)
    else:
        pn = p      # параметры привязаны к месту, а не к массе
    # пища и рост / распад
    eat = jnp.minimum(P["keat"] * jnp.exp(P["pk"] * pn[0]) * An * F, F)     # pk: p0 задаёт эффективность питания (keat·e^{pk·p0})
    An2 = jnp.maximum(An + eat - P["dec"] * An, 0.0)
    Fn = F + P["rF"] * (P["F0"] - F) - eat
    # мутации: коррелированный шум
    key, sk = jax.random.split(st["key"])
    nz = jax.random.normal(sk, (K,) + shape)
    nz = jnp.stack([conv(nz[k], Ko) for k in range(K)]) * (P["mut"] * 3.0)
    pn = jnp.clip(pn + nz * (An2[None] > 0.05), -1.0, 1.0)
    # фон (где массы нет) — p не определено; держим 0, чтобы не копить мусор
    pn = jnp.where(An2[None] > 1e-6, pn, 0.0)
    Qn = pn * An2[None]
    ct = P["c_lo"] + (P["c_hi"] - P["c_lo"]) * jnp.clip(An2, 0.0, 1.0)
    cn = c + (dt / P["tau_c"]) * (ct - c)
    return dict(A=An2, x=xn, y=yn, z=zn, c=cn, F=Fn, Q=Qn, key=key)

@partial(jax.jit, static_argnames=("n", "transport"))
def run_chunk(st, P, KA, Ko, n, transport=True):
    vs = jax.vmap(partial(step, transport=transport), in_axes=(0, 0, 0, None))
    return jax.lax.fori_loop(0, n, lambda i, s: vs(s, P, KA, Ko), st)

def make_state(base, c, K=2, p0=None, seed=0):
    """base: состояние из исходной модели (A,x,y,z,c). p0: массив (K,) или поле (K,...)"""
    L = base["A"].shape; p0 = np.zeros(K) if p0 is None else np.asarray(p0, np.float32)
    pf = p0.reshape((K,) + (1,) * len(L)) * np.ones((K,) + L, np.float32) if p0.ndim == 1 else p0
    st = {k: np.asarray(v, np.float32) for k, v in base.items()}
    st["F"] = np.full(L, c.get("F0", 1.0), np.float32); st["Q"] = (pf * st["A"][None]).astype(np.float32)
    st["key"] = np.asarray(jax.random.PRNGKey(seed))
    return st
