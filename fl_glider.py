#!/usr/bin/env python3
"""
Feigenbaum-Lenia с "массой": поиск движущихся структур (глайдеров/солитонов) в N-D.

К каждой клетке Рёсслера добавлено скалярное поле A ("масса"/активность, как в Lenia).
  U = K_A * A,  G(U) = 2·exp(-(U-μ)²/2σ²) − 1                         — Lenia-рост
  геном c_i релаксирует к c(A) = c_lo + (c_hi − c_lo)·clip(A,0,1) с инерцией τ_c   (геном может отставать)
  Рёсслер (x,y,z) с параметром c_i, связь ε·(K_o*x − x)
  обратная связь на рост: fb = β·(S − s0) − γ_c·(c − c_lo)/(c_hi − c_lo);  S — локальная амплитудная синхронизация,
  второй член — геном как медленный ингибитор (классический путь к дрейф-бифуркации пятна)
Механизмы движения (комбинируются):
  asym  — ядро K_A сдвинуто на asym·R по оси 0 (направление задано извне, "дрейф")
  chi   — адвекция A вдоль градиента фазы осцилляторов χ·∇φ (фазовые волны толкают массу; спонтанное направление)
  mode="flow" — Flow-Lenia: масса сохраняется, A_t = −∇·(A v), v = (1−α)∇(G+fb) − α∇A + χ∇φ, α = clip((A/θ)²,0,1)
Шаг для одной решётки; батч по кандидатам — jax.vmap (у каждого своё ядро и параметры).
"""
from functools import partial
import numpy as np
import jax, jax.numpy as jnp

def ring_kernel_np(L, N, R, peaks=(1.0,), shift=0.0, axis=0):
    ax = np.fft.fftfreq(L) * L
    g = [a.astype(np.float64) for a in np.meshgrid(*([ax] * N), indexing="ij", sparse=True)]
    if shift:
        g[axis] = ((g[axis] - shift * R + L / 2) % L) - L / 2
    d = np.sqrt(sum(x ** 2 for x in g)) / R
    K = np.zeros(np.broadcast(*g).shape); B = len(peaks)
    for i, h in enumerate(peaks):
        r = d * B - i; m = (r > 0) & (r < 1) & (d < 1)
        K[m] += h * np.exp(4.0 - 1.0 / (r[m] * (1.0 - r[m])))
    K /= K.sum()
    return np.fft.rfftn(K.astype(np.float32)).astype(np.complex64)

PKEYS = ["lag", "gam", "mu", "sigma", "dtA", "beta", "s0", "chi", "eps", "c_lo", "c_hi", "tau_c", "theta", "vmax", "dt", "a", "b", "sigA"]
DEF = dict(lag=0.0, gam=0.0, mu=0.2, sigma=0.03, dtA=0.1, beta=0.0, s0=0.5, chi=0.0, eps=0.1, c_lo=2.5, c_hi=5.0, tau_c=5.0,
           theta=1.5, vmax=0.1, dt=0.05, a=0.2, b=0.2, sigA=2.0)

def params(d):
    P = dict(DEF); P.update({k: v for k, v in d.items() if k in DEF})
    return {k: jnp.float32(P[k]) for k in PKEYS}

def _grad(F, k):
    return 0.5 * (jnp.roll(F, -1, k) - jnp.roll(F, 1, k))

def _advect(A, vs, dt):
    """Консервативная upwind-адвекция (unsplit), масса сохраняется точно."""
    out = A
    for k, v in enumerate(vs):
        vf = 0.5 * (v + jnp.roll(v, -1, k))
        up = jnp.where(vf > 0, A, jnp.roll(A, -1, k))
        Fl = vf * up * dt
        out = out - (Fl - jnp.roll(Fl, 1, k))
    return out

def _minmod(a, b):
    return jnp.where(a * b > 0, jnp.sign(a) * jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.0)

def _mc(a, b):
    """Monotonized-central лимитер (менее диффузный, чем minmod)."""
    return _minmod(0.5 * (a + b), 2.0 * _minmod(a, b))

def _advect_muscl(A, vs, dt, lim=None):
    """Unsplit консервативная схема 2-го порядка по пространству (MUSCL, minmod): меньше численной диффузии → меньше анизотропии."""
    out = A
    for k, v in enumerate(vs):
        vf = 0.5 * (v + jnp.roll(v, -1, k))
        Ap1 = jnp.roll(A, -1, k); Am1 = jnp.roll(A, 1, k); Ap2 = jnp.roll(A, -2, k)
        lf = lim or _minmod
        aL = A + 0.5 * lf(Ap1 - A, A - Am1)            # грань i+1/2 слева
        aR = Ap1 - 0.5 * lf(Ap2 - Ap1, Ap1 - A)        # грань i+1/2 справа
        Fl = vf * jnp.where(vf > 0, aL, aR) * dt
        out = out - (Fl - jnp.roll(Fl, 1, k))
    return out

def _sgrad(F):
    """Спектральный градиент (i·k·F̂), изотропный в отличие от центральных разностей."""
    N = F.ndim; L = F.shape[0]
    Fh = jnp.fft.rfftn(F); out = []
    for k in range(N):
        kk = np.fft.rfftfreq(L) * 2 * np.pi if k == N - 1 else np.fft.fftfreq(L) * 2 * np.pi
        if L % 2 == 0: kk = kk.copy(); kk[L // 2 if k < N - 1 else -1] = 0.0     # нулевая частота Найквиста
        sh = [1] * N; sh[k] = len(kk)
        out.append(jnp.fft.irfftn(Fh * (1j * kk.reshape(sh)).astype(np.complex64), s=F.shape))
    return out

def step(st, P, KA, Ko, mode="lenia", use_chi=False):
    A, x, y, z, c = st["A"], st["x"], st["y"], st["z"], st["c"]
    N = A.ndim; shape = A.shape
    conv = lambda F, K: jnp.fft.irfftn(jnp.fft.rfftn(F) * K, s=shape)
    U = conv(A, KA)
    G = 2.0 * jnp.exp(-((U - P["mu"]) ** 2) / (2 * P["sigma"] ** 2)) - 1.0
    # осцилляторы
    dt = P["dt"]
    ex, ey = conv(x, Ko) - x, conv(y, Ko) - y
    fx, fy = P["eps"] * ex, P["eps"] * ey
    f = lambda x, y, z: (-y - z + fx, x + P["a"] * y + fy, P["b"] + z * (x - c))
    k1 = f(x, y, z); k2 = f(x + .5*dt*k1[0], y + .5*dt*k1[1], z + .5*dt*k1[2])
    k3 = f(x + .5*dt*k2[0], y + .5*dt*k2[1], z + .5*dt*k2[2]); k4 = f(x + dt*k3[0], y + dt*k3[1], z + dt*k3[2])
    xn = x + dt/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]); yn = y + dt/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    zn = jnp.clip(z + dt/6*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2]), -50.0, 80.0)
    S = conv(jnp.exp(-(ex*ex + ey*ey) / P["sigA"] ** 2), Ko)
    fb = P["beta"] * (S - P["s0"]) - P["gam"] * (c - P["c_lo"]) / (P["c_hi"] - P["c_lo"]) \
         + P["lag"] * (jnp.clip(A, 0.0, 1.0) - (c - P["c_lo"]) / (P["c_hi"] - P["c_lo"]))
    # gam: медленный геном тормозит рост; lag: рост поощряется там, где геном ОТСТАЁТ от массы (фронт), и
    # подавляется там, где геном ещё высок, а массы уже нет (хвост). В стационаре lag-член = 0.
    vs = None
    if use_chi:
        r = jnp.sqrt(x*x + y*y) + 1e-6; cs, sn = x / r, y / r
        vs = [P["chi"] * (cs * _grad(sn, k) - sn * _grad(cs, k)) for k in range(N)]
    if mode == "lenia":
        An = jnp.clip(A + P["dtA"] * (G + fb), 0.0, 1.0)
        if use_chi:
            vs = [jnp.clip(v, -P["vmax"], P["vmax"]) for v in vs]
            An = jnp.clip(_advect(An, vs, 1.0), 0.0, 1.0)
    else:   # Flow-Lenia. mode: "flow" (исходная: центр. разности, покомпонентный клип, upwind 1-го порядка)
        #   "flow_iso*": клип по модулю вектора; "*_spec*": спектральные (изотропные) градиенты; "*muscl*": MUSCL-поток с minmod
        UG = G + fb
        al = jnp.clip((A / P["theta"]) ** 2, 0.0, 1.0)
        if "spec" in mode:
            gUG = _sgrad(UG); gA = _sgrad(A)
        else:
            gUG = [_grad(UG, k) for k in range(N)]; gA = [_grad(A, k) for k in range(N)]
        v = [(1 - al) * gUG[k] - al * gA[k] for k in range(N)]
        if use_chi: v = [vi + wi for vi, wi in zip(v, vs)]
        if mode == "flow":
            lim = 0.45 / N
            v = [jnp.clip(vi * P["dtA"] * 10.0, -lim, lim) for vi in v]     # скорость в клетках/шаг
        else:
            v = [vi * P["dtA"] * 10.0 for vi in v]
            mag = jnp.sqrt(sum(vi * vi for vi in v)) + 1e-12; f = jnp.minimum(1.0, (0.45 / np.sqrt(N)) / mag)
            v = [vi * f for vi in v]
        nsub = 2 if "sub2" in mode else 1      # "*sub2": два полушага адвекции (меньше ошибка расщепления по осям)
        An = A
        for _ in range(nsub):
            if "mc" in mode: An = _advect_muscl(An, v, 1.0 / nsub, lim=_mc)
            elif "muscl" in mode: An = _advect_muscl(An, v, 1.0 / nsub)
            else: An = _advect(An, v, 1.0 / nsub)
        An = jnp.maximum(An, 0.0)
    ct = P["c_lo"] + (P["c_hi"] - P["c_lo"]) * jnp.clip(An, 0.0, 1.0)
    cn = c + (dt / P["tau_c"]) * (ct - c)
    return dict(A=An, x=xn, y=yn, z=zn, c=cn)

def init_state(key, L, N, R, n_blobs=6, blob_r=1.0, density=0.5, mode="lenia"):
    """Случайный «суп»: n_blobs шаров радиуса blob_r·R со случайным шумом внутри (как в Lenia)."""
    shape = (L,) * N
    k = jax.random.split(key, 6)
    grids = jnp.meshgrid(*([jnp.arange(L, dtype=jnp.float32)] * N), indexing="ij")
    cen = jax.random.uniform(k[0], (n_blobs, N), maxval=L)
    A = jnp.zeros(shape)
    for i in range(n_blobs):
        d2 = sum(((g - cen[i, j] + L / 2) % L - L / 2) ** 2 for j, g in enumerate(grids))
        A = jnp.maximum(A, (d2 < (blob_r * R) ** 2).astype(jnp.float32))
    A = A * jax.random.uniform(k[1], shape) * (2 * density)
    A = jnp.clip(A, 0, 1)
    th = jax.random.uniform(k[2], shape, maxval=2 * np.pi); rad = jax.random.uniform(k[3], shape, minval=2, maxval=6)
    x, y = rad * jnp.cos(th), rad * jnp.sin(th)
    z = jax.random.uniform(k[4], shape, maxval=0.5)
    c = 2.5 + 2.5 * A
    return dict(A=A, x=x, y=y, z=z, c=c)

@partial(jax.jit, static_argnames=("n", "mode", "use_chi"))
def run_chunk(st, P, KA, Ko, n, mode="lenia", use_chi=False):
    vs = jax.vmap(partial(step, mode=mode, use_chi=use_chi), in_axes=(0, 0, 0, None))
    return jax.lax.fori_loop(0, n, lambda i, s: vs(s, P, KA, Ko), st)

# ----------------------------------------------------------------------------- детекторы (CPU, scipy)
def components(A, thr, min_mass=1.0, C=None, c_lo=2.5, c_hi=5.0):
    """Связные компоненты A>thr с периодическими границами. -> список dict(mass, cen[N], size)."""
    from scipy import ndimage
    A = np.asarray(A, np.float32); N = A.ndim; L = A.shape[0]
    lab, n = ndimage.label(A > thr)
    if n == 0: return [], lab
    parent = np.arange(n + 1)
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for k in range(N):
        a = np.take(lab, 0, axis=k).ravel(); b = np.take(lab, L - 1, axis=k).ravel()
        m = (a > 0) & (b > 0)
        for u, v in set(zip(a[m].tolist(), b[m].tolist())):
            ru, rv = find(u), find(v)
            if ru != rv: parent[ru] = rv
    roots = np.array([find(i) for i in range(n + 1)])
    lab = roots[lab]; lab[A <= thr] = 0
    ids = np.unique(lab); ids = ids[ids > 0]
    if len(ids) == 0: return [], lab
    mass = ndimage.sum(A, lab, ids); size = ndimage.sum(np.ones_like(A), lab, ids)
    cen = np.zeros((len(ids), N))
    for k in range(N):
        sh = [1] * N; sh[k] = L
        th = (2 * np.pi * np.arange(L) / L).reshape(sh)
        cs = ndimage.sum(A * np.cos(th), lab, ids); sn = ndimage.sum(A * np.sin(th), lab, ids)
        cen[:, k] = (np.arctan2(sn, cs) % (2 * np.pi)) * L / (2 * np.pi)
    cenC = np.full_like(cen, np.nan)
    if C is not None:
        W = np.clip((np.asarray(C, np.float32) - c_lo) / (c_hi - c_lo), 0, None)
        for k in range(N):
            sh = [1] * N; sh[k] = L
            th = (2 * np.pi * np.arange(L) / L).reshape(sh)
            cs = ndimage.sum(W * np.cos(th), lab, ids); sn = ndimage.sum(W * np.sin(th), lab, ids)
            cenC[:, k] = (np.arctan2(sn, cs) % (2 * np.pi)) * L / (2 * np.pi)
    out = [dict(id=int(i), mass=float(m), size=float(s), cen=c, cenC=cc) for i, m, s, c, cc in zip(ids, mass, size, cen, cenC) if m >= min_mass]
    return out, lab

def pdist(a, b, L):
    d = np.abs(a - b); d = np.minimum(d, L - d); return np.sqrt((d ** 2).sum(-1))

def pdelta(a, b, L):
    d = b - a; return (d + L / 2) % L - L / 2

def patch(A, cen, h):
    """Куб (2h+1)^N вокруг центра, с периодическим оборачиванием."""
    A = np.asarray(A); L = A.shape[0]; out = A
    for k, c in enumerate(cen):
        idx = (np.arange(-h, h + 1) + int(round(c))) % L
        out = np.take(out, idx, axis=k)
    return out

def track(frames, L, R, every, max_jump=None):
    """frames: список списков компонент (с 'patch'). Жадное сопоставление ближайших центров.
    Возвращает треки: dict(t0, cens[T,N], mass[T], corr[T-1])."""
    max_jump = max_jump or 0.75 * R
    tracks, active = [], []
    for t, comps in enumerate(frames):
        used = set(); new_active = []
        for tr in active:
            last = tr["cens"][-1]
            best, bd = None, max_jump
            for j, c in enumerate(comps):
                if j in used: continue
                d = pdist(last, c["cen"], L)
                if d < bd and 0.5 < c["mass"] / tr["mass"][-1] < 2.0: best, bd = j, d
            if best is not None:
                c = comps[best]; used.add(best)
                p0, p1 = tr["_patch"], c["patch"]
                cc = float(np.corrcoef(p0.ravel(), p1.ravel())[0, 1]) if p0.std() > 0 and p1.std() > 0 else 0.0
                tr["cens"].append(c["cen"]); tr["mass"].append(c["mass"]); tr["size"].append(c["size"]); tr["cenC"].append(c.get("cenC")); tr["corr"].append(cc); tr["_patch"] = p1
                new_active.append(tr)
            else:
                tr["t1"] = t - 1; tracks.append(tr)
        for j, c in enumerate(comps):
            if j not in used:
                new_active.append(dict(t0=t, cens=[c["cen"]], mass=[c["mass"]], size=[c["size"]], cenC=[c.get("cenC")], corr=[], _patch=c["patch"]))
        active = new_active
    for tr in active:
        tr["t1"] = len(frames) - 1; tracks.append(tr)
    out = []
    for tr in tracks:
        C = np.array(tr["cens"]); T = len(C)
        steps = np.array([pdelta(C[i], C[i + 1], L) for i in range(T - 1)]) if T > 1 else np.zeros((0, C.shape[1]))
        path = np.linalg.norm(steps, axis=1).sum() if T > 1 else 0.0
        net = np.linalg.norm(steps.sum(0)) if T > 1 else 0.0
        m = np.array(tr["mass"])
        # отставание генома: смещение c-центроида относительно центра массы вдоль скорости (отрицательное = «след»)
        lagp = np.nan
        if T > 1 and tr["cenC"][0] is not None and np.linalg.norm(steps.sum(0)) > 0:
            u = steps.sum(0) / np.linalg.norm(steps.sum(0))
            offs = np.array([pdelta(C[i], np.asarray(tr["cenC"][i]), L) for i in range(T)])
            lagp = float(np.nanmean(offs @ u))
        out.append(dict(t0=tr["t0"], t1=tr["t1"], life=(T - 1) * every, net=float(net), path=float(path),
                        straight=float(net / path) if path > 0 else 0.0, speed=float(net / max((T - 1) * every, 1)),
                        mass_mean=float(m.mean()), size_max=float(np.max(tr["size"])), mass_cv=float(m.std() / (m.mean() + 1e-9)),
                        corr=float(np.mean(tr["corr"])) if tr["corr"] else 0.0, genome_offset=lagp, vel=(steps.sum(0) / max((T - 1) * every, 1)).tolist(),
                        cens=C.tolist(), masses=m.tolist()))
    return out

def score_tracks(tr, R, total_steps, vol_total=None):
    """Глайдер: локализован (объём < 10% решётки), живёт ≥ 40% прогона, ушёл дальше 1.5R,
    прямолинейность > 0.6, масса стабильна (CV<0.3), форма держится (corr>0.6). Статичный солитон — то же, но net < 0.5R."""
    loc = lambda t: vol_total is None or t["size_max"] < 0.1 * vol_total
    g = [t for t in tr if loc(t) and t["life"] >= 0.4 * total_steps and t["net"] >= 1.5 * R and t["straight"] > 0.6
         and t["mass_cv"] < 0.3 and t["corr"] > 0.6]
    stat = [t for t in tr if loc(t) and t["life"] >= 0.4 * total_steps and t["net"] < 0.5 * R and t["mass_cv"] < 0.3]
    return g, stat
