"""Часть 11 (перф): тот же шаг, что fl_eco2, с опциями ускорения:
coarse — широкие гладкие ядра (far R16, big R12, blur R6) считаются на решётке в 2× грубее (пулинг 2^N → свёртка → линейная интерполяция обратно);
fnoise — шум мутаций генерируется прямо в Фурье-пространстве (без прямого FFT);
m — мутации и гомогенизация раз в m шагов (амплитуда ×√m, hom_eff = 1−(1−hom)^m)."""
import numpy as np, jax, jax.numpy as jnp
from functools import partial
import fl_glider as FG
from fl_eco import make_state as _make_state0
import fl_eco2 as _E2
# Часть 12: ниши (пространственный оптимум p0), сезонность (временной оптимум), два ресурса с компромиссом (частотно-зависимый отбор)
XKEYS = ["kn", "na", "nl", "sa", "nt", "res2", "g2"]
XDEF = dict(kn=0.0, na=0.0, nl=1.0, sa=0.0, nt=1.0, res2=0.0, g2=2.0)
EKEYS = _E2.EKEYS + XKEYS
EDEF = dict(_E2.EDEF, **XDEF)

def params(c):
    P = _E2.params(c); P.update({k: jnp.float32(c.get(k, XDEF[k])) for k in XKEYS}); return P

def make_state(base, c, K=2, p0=None, seed=0):
    st = _make_state0(base, c, K=K, p0=p0, seed=seed)
    st["F2"] = np.full_like(st["F"], c.get("F0", 1.0)); st["t"] = np.float32(0.0)
    return st

def kernels(L, N, R=6.0, coarse=False, **_):
    if coarse:
        Lc = L // 2
        return dict(far=jnp.asarray(FG.ring_kernel_np(Lc, N, 8.0, (0.0, 1.0))), big=jnp.asarray(FG.ring_kernel_np(Lc, N, 6.0, (1.0,))),
                    blur=jnp.asarray(FG.ring_kernel_np(Lc, N, R / 2, (1.0,))))
    return dict(far=jnp.asarray(FG.ring_kernel_np(L, N, 16.0, (0.0, 1.0))), big=jnp.asarray(FG.ring_kernel_np(L, N, 12.0, (1.0,))),
                blur=jnp.asarray(FG.ring_kernel_np(L, N, R, (1.0,))))

def pool2(G):
    N = G.ndim; L = G.shape[0]
    return G.reshape(sum(((L // 2, 2) for _ in range(N)), ())).mean(axis=tuple(range(1, 2 * N, 2)))

def up2(c):
    """Линейная интерполяция ×2 по каждой оси (клетки-центры: тонкие 2i, 2i+1 ↔ грубые i∓0.25)."""
    for k in range(c.ndim):
        a = 0.75 * c + 0.25 * jnp.roll(c, 1, k); b = 0.75 * c + 0.25 * jnp.roll(c, -1, k)
        c = jnp.stack([a, b], axis=k + 1); sh = list(a.shape); sh[k] *= 2; c = c.reshape(sh)
    return c

def _noise(key, shape, Kk, fnoise):
    if fnoise:     # комплексный гаусс в формате rfftn → одна обратная FFT
        ks = Kk.shape; k1, k2 = jax.random.split(key)
        Z = (jax.random.normal(k1, ks) + 1j * jax.random.normal(k2, ks)).astype(jnp.complex64)
        return jnp.fft.irfftn(Z * Kk, s=shape)
    return jnp.fft.irfftn(jnp.fft.rfftn(jax.random.normal(key, shape)) * Kk, s=shape)

def step(st, P, KA, Ko, KX, transport=True, coarse=False, fnoise=False, do_mut=True, do_hom=True, m=1):
    A, x, y, z, c, F, Q = st["A"], st["x"], st["y"], st["z"], st["c"], st["F"], st["Q"]
    N = A.ndim; shape = A.shape; K = Q.shape[0]
    conv = lambda G, Kk: jnp.fft.irfftn(jnp.fft.rfftn(G) * Kk, s=shape)
    if coarse:
        cs_ = tuple(s // 2 for s in shape)
        wconv = lambda G, Kk: up2(jnp.fft.irfftn(jnp.fft.rfftn(pool2(G)) * Kk, s=cs_))
        nshape = cs_
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
    if do_hom:
        hom = P["hom"] if m == 1 else 1.0 - (1.0 - P["hom"]) ** m
        Ab = wconv(An, KX["blur"]); Qb = jnp.stack([wconv(pn[k] * An, KX["blur"]) for k in range(K)])
        ps = Qb / jnp.maximum(Ab[None], 1e-6)
        pn = jnp.where(Ab[None] > 1e-4, (1 - hom) * pn + hom * ps, pn)
    Ub = wconv(An, KX["big"])
    decl = P["dec"] * jnp.exp(P["ks"] * jnp.maximum(Ub / P["U1"] - 1.0, 0.0))
    t = st.get("t", jnp.float32(0.0))
    if "F2" in st:
        x0 = jnp.arange(shape[0], dtype=jnp.float32).reshape((-1,) + (1,) * (N - 1))
        popt = P["na"] * jnp.sin(2 * np.pi * x0 / P["nl"]) + P["sa"] * jnp.sin(2 * np.pi * t / P["nt"])
        rate = P["keat"] * jnp.exp(P["pk"] * pn[0] - P["kn"] * (pn[0] - popt) ** 2)
        F2 = st["F2"]; w1 = ((1 + pn[0]) / 2) ** P["g2"]; w2 = ((1 - pn[0]) / 2) ** P["g2"]
        e1 = jnp.minimum(jnp.where(P["res2"] > 0, P["keat"] * w1, rate) * An * F, F)
        e2 = jnp.where(P["res2"] > 0, jnp.minimum(P["keat"] * w2 * An * F2, F2), 0.0)
        eat = e1 + e2
        F2n = F2 + P["rF"] * (P["F0"] - F2) - e2
        Fn = F + P["rF"] * (P["F0"] - F) - e1
    else:
        eat = jnp.minimum(P["keat"] * jnp.exp(P["pk"] * pn[0]) * An * F, F)
        Fn = F + P["rF"] * (P["F0"] - F) - eat
    An2 = jnp.maximum(An + eat - decl * An, 0.0)
    key, sk = jax.random.split(st["key"])
    if do_mut:
        nz = jnp.stack([_noise(jax.random.fold_in(sk, k), nshape, KX["blur"], fnoise) for k in range(K)])
        nz = nz / (jnp.std(nz, axis=tuple(range(1, N + 1)), keepdims=True) + 1e-12)
        if coarse: nz = jnp.stack([up2(nz[k]) for k in range(K)])
        pn = jnp.clip(pn + P["mutn"] * np.sqrt(m) * nz * (An2[None] > 0.05), -1.0, 1.0)
    pn = jnp.where(An2[None] > 1e-6, pn, 0.0)
    ct = P["c_lo"] + (P["c_hi"] - P["c_lo"]) * jnp.clip(An2, 0.0, 1.0)
    cn = c + (dt / P["tau_c"]) * (ct - c)
    out = dict(A=An2, x=xn, y=yn, z=zn, c=cn, F=Fn, Q=pn * An2[None], key=key)
    if "F2" in st: out["F2"] = F2n; out["t"] = t + 1.0
    return out

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

def U1_of(A, L, N, coarse=False, **_):
    A = jnp.asarray(A); KX = kernels(L, N, coarse=coarse)
    if coarse:
        return float(np.asarray(up2(jnp.fft.irfftn(jnp.fft.rfftn(pool2(A)) * KX["big"], s=tuple(s // 2 for s in A.shape)))).max())
    return float(np.asarray(jnp.fft.irfftn(jnp.fft.rfftn(A) * KX["big"], s=A.shape)).max())
