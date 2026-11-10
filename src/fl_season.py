"""Часть 18: «сезоны» в 4D. На базе fl_poison: два корма A (поле F) и B (поле F2), оба восстанавливаются (rG) к взаимодополняющим
движущимся пятнам: A* = F0·relu(c), B* = F0·relu(−c), c = cos(2πx0/λ+ωt)·cos(2πx1/λ−ωt).
Сезон σ: при σ>0 A хороший, B — яд (стоит cp массы за единицу), при σ<0 наоборот.
По умолчанию σ = sign(sin(2πt/Ts)) — так считались Части 18 и 18b. season_rule=1 (только 18c) берёт
половину периода, t/Ts mod 1 <= 1/2, включая концы: float32 sin(kπ) ~ −1e-7 и на этих кадрах
float32 и float64 расходились на один шаг.
Мозг: входы s = [Ã, B̃, R, h, cue·σ, Ub/U1−1, Ufar/U1] (Ã = blur(A)/F0, B̃ = blur(B)/F0,
  R — «награда»: размытый по телу чистый прирост от еды за прошлый шаг на единицу массы, /(keat·F0);
  h — наследуемая переменная памяти (канал Q, переносится MUSCL и гомогенизируется по телу, как веса; не мутирует)).
  H=3 (tanh), выходы o = [g_A, g_B, g_R, o_h]:  v += dtA·10·(kb·(g_A∇Ã + g_B∇B̃) − kr·g_R∇Ufar);  h ← h + α(o_h − h) каждый шаг (если mem=1).
mem=0 — без памяти (вход h=0, h не обновляется; вход R остаётся — чистая абляция памяти). cue=1 — сезон дан напрямую. cue_lead>0 — мозг видит σ(t+cue_lead); еда/яд по-прежнему от σ(t) (default 0 = legacy).
D: D0 = съедено A, D1 = съедено B, D2 = R, D3 = A·g_A, D4 = A·g_B (реализованные усиления)."""
import numpy as np, jax, jax.numpy as jnp
from functools import partial
import fl_glider as FG
import fl_eco3 as E3
from fl_eco3 import kernels, pool2, up2, _noise, U1_of
NIN, H, NOUT = 7, 3, 4
NW = H * NIN + H + NOUT * H + NOUT          # 40
K0 = 2; iM = K0 + NW; KB = iM + 1           # 43 канала (последний — память h)
# Optional lineage tag: p[iL] when K>KB (K=KB+1). Not used by the brain or feeding.
# Mutation pads extra channels with 0, so K=KB stays the legacy amp. Default runs keep K=KB.
iL = iM + 1
ND = 5
# Sensory columns of W1. R, h and the oracle cue are the channels Part 18 added; A, B, Ub, Ufar come from Part 17.
IN_A, IN_B, IN_R, IN_H, IN_CUE, IN_UB, IN_UFAR = range(NIN)
OUT_GA, OUT_GB, OUT_GR, OUT_H = range(NOUT)
BKEYS = ["kb", "kr", "mutw", "mut_bias", "ws", "mem", "cue", "cue_lead", "alpha", "rG", "cp", "lam", "om", "Ts", "smooth", "season_rule"]
# mut_bias < 0 → use mutw for g_A/g_B (legacy; bit-identical amp). >=0 overrides only those two biases.
BDEF = dict(kb=0.0, kr=0.0, mutw=None, mut_bias=-1.0, ws=2.0, mem=1.0, cue=0.0, cue_lead=0.0, alpha=0.01, rG=0.001, cp=2.0, lam=28.0, om=2 * np.pi / 6000, Ts=6000.0, smooth=0.0, season_rule=0.0)

def w1_flat(h, col):
    """Index of W1[h, col] inside the flat NW weight vector (before the K0 offset)."""
    return h * NIN + col

def b2_flat(out):
    return H * NIN + H + NOUT * H + out

def w2_flat(out, h):
    return H * NIN + H + out * H + h

def new_channel_flat():
    """Flat indices of the Part-18 channels: W1 columns R, h, cue, the o_h row, and b2[o_h]."""
    idx = [w1_flat(h, col) for h in range(H) for col in (IN_R, IN_H, IN_CUE)]
    idx += [w2_flat(OUT_H, h) for h in range(H)]
    idx.append(b2_flat(OUT_H))
    return tuple(idx)

def food_gain(eA, eB, sigma, cp, xp=np):
    """Mass added by one feeding event. sigma > 0 means A is food and B is poison.

    With cost cp the sign flips only when the poison share of intake exceeds 1/(1+cp)
    (2/3 at the Part 18b value cp=0.5). step() calls this with xp=jax.numpy.
    """
    sigma = xp.asarray(sigma)
    return xp.where(sigma > 0, eA - cp * eB, eB - cp * eA)

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

def season(t, Ts, xp=jnp):
    """Historical Part 18/18b rule: +1 when sin(2πt/Ts) >= 0. Kept so old logs re-analyze the same way."""
    return xp.where(xp.sin(2 * np.pi * t / Ts) >= 0, 1.0, -1.0)

def season_half(t, Ts, xp=jnp):
    """+1 on the closed first half of each period, including t = n·Ts/2.

    float32 sin(kπ) is about −1e-7 and float64 sin(2kπ) is about −1e-16, so `season`
    labels those endpoints differently in the simulator and in numpy. This rule does not.
    """
    tt = xp.asarray(t)
    if getattr(tt.dtype, "kind", "f") not in "fc":
        tt = tt.astype(xp.result_type(tt, xp.float32(0)))
    Ts = xp.asarray(Ts, dtype=tt.dtype)
    u = xp.mod(tt / Ts, xp.asarray(1.0, dtype=tt.dtype))
    return xp.where(u <= 0.5, xp.asarray(1.0, dtype=tt.dtype), xp.asarray(-1.0, dtype=tt.dtype))

def snapshot_season(t, Ts, xp=np):
    """18c label of the feeding step recorded at log time t. That step read the clock t−1."""
    t = float(t)
    return float(season_half(0.0 if t <= 0.0 else t - 1.0, Ts, xp))

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
    t = st.get("t", jnp.float32(0.0))
    sg = jnp.where(P["season_rule"] > 0, season_half(t, P["Ts"]), season(t, P["Ts"]))
    # cue_lead: brain sees a future season; food/poison keep using sg (legacy when cue_lead=0).
    t_cue = t + P["cue_lead"]
    sg_cue = jnp.where(P["season_rule"] > 0, season_half(t_cue, P["Ts"]), season(t_cue, P["Ts"]))
    # --- мозг ---
    At = wconv(F, KX["blur"]) / P["F0"]; Bt = wconv(FB, KX["blur"]) / P["F0"]
    hm = p[iM] * P["mem"]
    Ub0 = wconv(A, KX["big"])
    s_in = jnp.stack([At, Bt, Rw, hm, P["cue"] * sg_cue * jnp.ones_like(A), Ub0 / P["U1"] - 1.0, Ufar / P["U1"]])
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
    # Memory write uses the pre-advection output on the post-advection field (one-step spatial lag).
    # Bodies move far less than the blur radius per step, and Part 18c keeps this order so a re-run of 18/18b matches.
    pn = pn.at[iM].set(jnp.where(P["mem"] > 0, pn[iM] + P["alpha"] * (o[3] - pn[iM]), pn[iM]))
    Ub = wconv(An, KX["big"])
    decl = P["dec"] * jnp.exp(P["ks"] * jnp.maximum(Ub / P["U1"] - 1.0, 0.0))
    rate = P["keat"] * jnp.exp(P["pk"] * pn[0])
    eA = jnp.minimum(rate * An * F, F); eB = jnp.minimum(rate * An * FB, FB)
    cc = pattern(shape, P, t)
    ta, tb = targets(cc, P)
    Fn = F + P["rG"] * (ta - F) - eA
    FBn = FB + P["rG"] * (tb - FB) - eB
    gain = food_gain(eA, eB, sg, P["cp"], jnp)
    An2 = jnp.maximum(An + gain - decl * An, 0.0)
    Rn = wconv(gain, KX["blur"]) / jnp.maximum(wconv(An2, KX["blur"]), 1e-4) / (P["keat"] * P["F0"])
    Rn = jnp.clip(Rn, -3.0, 3.0)
    key, sk = jax.random.split(st["key"])
    if do_mut:
        nz = jnp.stack([_noise(jax.random.fold_in(sk, k), nshape, KX["blur"], fnoise) for k in range(K)])
        nz = nz / (jnp.std(nz, axis=tuple(range(1, N + 1)), keepdims=True) + 1e-12)
        if coarse: nz = jnp.stack([up2(nz[k]) for k in range(K)])
        w_amp = jnp.full((NW,), P["mutw"])
        mb = P["mut_bias"]
        # When mut_bias < 0 the where keeps mutw, so the selected values match the legacy full(NW, mutw).
        w_amp = jnp.where(
            (jnp.arange(NW) == b2_flat(OUT_GA)) | (jnp.arange(NW) == b2_flat(OUT_GB)),
            jnp.where(mb >= 0.0, mb, P["mutw"]),
            w_amp,
        )
        # K=KB → zeros(1) on memory (legacy). K>KB (lineage) → extra channels do not mutate.
        amp = jnp.concatenate([jnp.full((K0,), P["mutn"]), w_amp, jnp.zeros((K - K0 - NW,))]).reshape((K,) + (1,) * N)
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
