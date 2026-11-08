#!/usr/bin/env python3
"""
Feigenbaum-Lenia на JAX: GPU/CPU движок (jit + lax.scan + jnp.fft, vmap по батчу решёток).

Одна решётка = pytree State с полями формы (L,)*N. Батч решёток (сиды, режимы, гиперпараметры)
получается через jax.vmap по ведущей оси State и Params — так что весь grid-search идёт одним ядром.

Правила эволюции генома c (статический аргумент rule):
  "none" : c заморожен
  "own"  : как в numpy-версии — собственный касательный показатель Ляпунова клетки (без связи и входа)
  "cond" : УСЛОВНЫЙ показатель Ляпунова. Двойник решётки (twin) гонится тем же входом u(t);
           локальная (усреднённая ядром) скорость расхождения двойника и основной решётки,
           с перенормировкой Бенеттина к δ0 на каждом шаге, = локальный условный λc_i.
           dc/dt = η·(−s·tanh((λc − λ*)/λs) + κ·G(S)) + Dc·(K_c*c − c),   λ* < 0 — "край устойчивости при входе".
           s = gsign = +1: измерено (sweep_lamc.py), что λc монотонно растёт с c и для Рёсслера a=0.2,
           и для затухающей клетки a<0, так что "λc выше цели -> уменьшить c" работает в обоих вариантах.
"""
from functools import partial
import numpy as np
import jax, jax.numpy as jnp

# ----------------------------------------------------------------------------- kernels (numpy, один раз)
def lenia_kernel_np(L, N, R, peaks=(1.0,)):
    ax = np.fft.fftfreq(L) * L
    grids = np.meshgrid(*([ax] * N), indexing="ij", sparse=True)
    d = np.sqrt(sum(g.astype(np.float64) ** 2 for g in grids)) / R
    K = np.zeros_like(d); B = len(peaks)
    for i, h in enumerate(peaks):
        r = d * B - i; m = (r > 0) & (r < 1) & (d < 1)
        K[m] += h * np.exp(4.0 - 1.0 / (r[m] * (1.0 - r[m])))
    K /= K.sum()
    return np.fft.rfftn(K.astype(np.float32)).astype(np.complex64)

DEFAULT = dict(eps=0.1, a=0.2, b=0.2, gamma=0.0, dt=0.05, eta=0.1, lam_target=0.02, lam_scale=0.05,
               tau_lam=30.0, kappa=0.7, mu=0.5, sigma=0.2, sigA=2.0, Dc=0.02, cmin=2.0, cmax=7.0,
               warmup=40.0, delta0=1e-3, gsign=1.0)

def make_params(**kw):
    P = dict(DEFAULT); P.update(kw)
    return {k: jnp.asarray(v, jnp.float32) for k, v in P.items()}

def stack_params(plist):
    return jax.tree.map(lambda *a: jnp.stack(a), *plist)

# ----------------------------------------------------------------------------- state
def init_state(key, shape, c_init=5.0, c_spread=1.2, K_smooth=None, delta0=1e-3, in_frac=1.0, win_key=None,
               twin=True):
    """Начальное состояние одной решётки. win_key задаёт маску входа отдельно от начальных условий
    (для теста echo-state: одна маска, разные начальные условия)."""
    k = jax.random.split(key, 8)
    th = jax.random.uniform(k[0], shape, minval=0, maxval=2 * np.pi)
    rad = jax.random.uniform(k[1], shape, minval=2, maxval=6)
    x, y = rad * jnp.cos(th), rad * jnp.sin(th)
    z = jax.random.uniform(k[2], shape, minval=0, maxval=0.5)
    if c_spread > 0 and K_smooth is not None:
        n = jax.random.normal(k[3], shape)
        conv = lambda A: jnp.fft.irfftn(jnp.fft.rfftn(A) * K_smooth, s=shape)
        n = conv(conv(n)); n = (n - n.mean()) / (n.std() + 1e-9)
        c = jnp.clip(c_init + c_spread * n, 2.0, 7.0)
    else:
        c = jnp.full(shape, c_init, jnp.float32)
    v = jax.random.normal(k[4], (3,) + shape); v = v / jnp.sqrt((v ** 2).sum(0, keepdims=True))
    wk = k[5] if win_key is None else win_key
    w1, w2 = jax.random.split(wk)
    win = (jax.random.uniform(w1, shape) < in_frac) * jax.random.normal(w2, shape)
    p = jax.random.normal(k[6], (3,) + shape); p = p / jnp.sqrt((p ** 2).sum(0, keepdims=True)) * delta0
    st = dict(x=x, y=y, z=z, c=c, dx=v[0], dy=v[1], dz=v[2], lam=jnp.zeros(shape), win=win.astype(jnp.float32),
              t=jnp.zeros((), jnp.float32))
    st["x2"], st["y2"], st["z2"] = x + p[0], y + p[1], z + p[2]
    st["lamc"] = jnp.zeros(shape)
    return jax.tree.map(lambda a: a.astype(jnp.float32), st)

# ----------------------------------------------------------------------------- one step (single lattice)
def _rk4(x, y, z, c, a, b, fx, fy, dt):
    f = lambda x, y, z: (-y - z + fx, x + a * y + fy, b + z * (x - c))
    k1 = f(x, y, z); k2 = f(x + .5*dt*k1[0], y + .5*dt*k1[1], z + .5*dt*k1[2])
    k3 = f(x + .5*dt*k2[0], y + .5*dt*k2[1], z + .5*dt*k2[2]); k4 = f(x + dt*k3[0], y + dt*k3[1], z + dt*k3[2])
    xn = x + dt/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]); yn = y + dt/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    zn = jnp.clip(z + dt/6*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2]), -50.0, 80.0)
    return xn, yn, zn, k1

def step(st, u, adapt, P, K, Kc, rule="own", sync=True):
    """Один шаг одной решётки. u — скаляр входа, adapt ∈ {0,1} — включена ли эволюция генома."""
    shape = st["x"].shape
    conv = lambda A, Kh: jnp.fft.irfftn(jnp.fft.rfftn(A) * Kh, s=shape)
    dt, a, b = P["dt"], P["a"], P["b"]
    x, y, z, c = st["x"], st["y"], st["z"], st["c"]
    ex, ey = conv(x, K) - x, conv(y, K) - y
    fx = P["eps"] * ex + P["gamma"] * st["win"] * u; fy = P["eps"] * ey
    xn, yn, zn, k1 = _rk4(x, y, z, c, a, b, fx, fy, dt)
    # собственный касательный FTLE клетки (RK2), как в numpy-версии
    dx, dy, dz = st["dx"], st["dy"], st["dz"]
    jv = lambda dx, dy, dz, x, z: (-dy - dz, dx + a * dy, z * dx + (x - c) * dz)
    j1 = jv(dx, dy, dz, x, z)
    j2 = jv(dx + .5*dt*j1[0], dy + .5*dt*j1[1], dz + .5*dt*j1[2], x + .5*dt*k1[0], z + .5*dt*k1[2])
    dx, dy, dz = dx + dt*j2[0], dy + dt*j2[1], dz + dt*j2[2]
    g = jnp.sqrt(dx*dx + dy*dy + dz*dz) + 1e-30
    lam = st["lam"] + (jnp.log(g) / dt - st["lam"]) * (dt / P["tau_lam"])
    new = dict(st, x=xn, y=yn, z=zn, dx=dx / g, dy=dy / g, dz=dz / g, lam=lam)
    if rule == "cond":
        # двойник: та же динамика, тот же вход, тот же геном
        x2, y2, z2 = st["x2"], st["y2"], st["z2"]
        fx2 = P["eps"] * (conv(x2, K) - x2) + P["gamma"] * st["win"] * u; fy2 = P["eps"] * (conv(y2, K) - y2)
        x2n, y2n, z2n, _ = _rk4(x2, y2, z2, c, a, b, fx2, fy2, dt)
        Dx, Dy, Dz = x2n - xn, y2n - yn, z2n - zn
        D2 = 0.5 * (Dx*Dx + Dy*Dy + Dz*Dz) + 0.5 * conv(Dx*Dx + Dy*Dy + Dz*Dz, K)  # локальная окрестность
        D = jnp.sqrt(jnp.maximum(D2, 0.0) + 1e-30)
        d0 = P["delta0"]
        inst = jnp.log(D / d0) / dt
        lamc = st["lamc"] + (inst - st["lamc"]) * (dt / P["tau_lam"])
        r = d0 / D                                              # локальная перенормировка Бенеттина
        new.update(x2=xn + Dx * r, y2=yn + Dy * r, z2=zn + Dz * r, lamc=lamc)
        signal = lamc
    else:
        signal = lam
    if rule != "none":
        gate = adapt * (st["t"] >= P["warmup"]).astype(jnp.float32)
        drive = -P["gsign"] * jnp.tanh((signal - P["lam_target"]) / P["lam_scale"])
        if sync:
            S = conv(jnp.exp(-(ex*ex + ey*ey) / P["sigA"] ** 2), K)
            drive = drive + P["kappa"] * (2.0 * jnp.exp(-((S - P["mu"]) ** 2) / (2 * P["sigma"] ** 2)) - 1.0)
        dc = P["eta"] * drive + P["Dc"] * (conv(c, Kc) - c)
        cn = c + dt * dc
        cn = jnp.where(cn < P["cmin"], 2 * P["cmin"] - cn, cn); cn = jnp.where(cn > P["cmax"], 2 * P["cmax"] - cn, cn)
        cn = jnp.clip(cn, P["cmin"], P["cmax"])
        new["c"] = jnp.where(gate > 0, cn, c)
    new["t"] = st["t"] + dt
    return new

# ----------------------------------------------------------------------------- batched runners
def _vstep(rule, sync):
    return jax.vmap(partial(step, rule=rule, sync=sync), in_axes=(0, None, 0, 0, None, None))

@partial(jax.jit, static_argnames=("n", "rule", "sync"))
def run_free(st, P, K, Kc, n, rule="own", sync=True, adapt=None):
    """n шагов без входа для батча решёток (ведущая ось). adapt[B] — маска эволюции генома (по умолчанию все)."""
    B = st["x"].shape[0]; vs = _vstep(rule, sync)
    ad = jnp.ones((B,), jnp.float32) if adapt is None else adapt
    return jax.lax.fori_loop(0, n, lambda i, s: vs(s, 0.0, ad, P, K, Kc), st)

def make_reservoir_runner(rule, sync, spp, avg=True):
    """Возвращает jit-функцию: (st, P, K, Kc, u[T], sel[B,nread], aw[B], aa[B], washout) ->
    (st, feats[B,T,3*nread], diag[T,B,4])."""
    vs = _vstep(rule, sync)
    @jax.jit
    def run(st, P, K, Kc, u, sel, aw, aa, washout):
        B = st["x"].shape[0]
        take = jax.vmap(lambda A, idx: A.reshape(-1)[idx])
        def feat(s):
            return jnp.concatenate([take(s["x"], sel), take(s["y"], sel), take(s["z"], sel)], axis=1)
        def body(s, xs):
            uk, k = xs
            adapt = jnp.where(k < washout, aw, aa)
            def inner(i, carry):
                s, acc = carry
                s = vs(s, uk, adapt, P, K, Kc)
                return s, (acc + feat(s) if avg else feat(s))
            s, acc = jax.lax.fori_loop(0, spp, inner, (s, jnp.zeros((B, 3 * sel.shape[1]), jnp.float32)))
            f = acc / spp if avg else acc
            red = lambda A: A.reshape(B, -1).mean(1)
            d = jnp.stack([red(s["lam"]), red(s["lamc"]), red(s["c"]), red((s["lam"] > 0.01).astype(jnp.float32))], 1)
            return s, (f, d)
        T = u.shape[0]
        s, (F, D) = jax.lax.scan(body, st, (u, jnp.arange(T)))
        return s, jnp.swapaxes(F, 0, 1), D
    return run
