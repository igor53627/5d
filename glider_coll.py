"""Систематические столкновения двух солитонов (точные копии через знако-перестановки решётки).
Конфигурация: угол между скоростями θ, прицельный параметр b, сдвиг фазы τ (второй объект прогоняется ещё τ шагов)."""
import itertools, time, numpy as np

def _sim(c, st, n, N, L, every=50, track=True):
    import jax, jax.numpy as jnp, fl_glider as FG
    from glider_es5 import _metric_fns
    KA = jnp.asarray(FG.ring_kernel_np(L, N, c["R"], tuple(c["peaks"])))[None]; Ko = jnp.asarray(FG.ring_kernel_np(L, N, 3.0))
    P = jax.tree.map(lambda a: a[None], FG.params(c)); s = jax.tree.map(lambda a: jnp.asarray(a)[None], st)
    vone, _ = _metric_fns(N, L); cens = []
    for _ in range(n // every):
        s = FG.run_chunk(s, P, KA, Ko, every, mode=c.get("mode", "flow"), use_chi=True)
        if track: cens.append(np.asarray(vone(s["A"])[1][0]))
    return {k: np.asarray(v[0]) for k, v in s.items()}, cens

def soliton(c, N, L, seed, pre=1500, every=50):
    from glider_es5 import _single_state
    st, cens = _sim(c, _single_state(c, N, L, seed), pre, N, L, every)
    C = np.array(cens[len(cens) // 2:]); d = (np.diff(C, axis=0) + L / 2) % L - L / 2; v = d.sum(0) / (len(d) * every)
    sh = tuple(int(L // 2 - round(x)) for x in cens[-1]); st = {k: np.roll(x, sh, axis=tuple(range(N))) for k, x in st.items()}
    return st, v

def signed_perms(N):
    for p in itertools.permutations(range(N)):
        for s in itertools.product([1, -1], repeat=N): yield p, s

def transform(st, p, s):
    out = {}
    for k, x in st.items():
        y = np.transpose(x, p)
        for ax, sg in enumerate(s):
            if sg < 0: y = np.flip(y, ax)
        out[k] = np.roll(y, tuple(1 if sg < 0 and x.shape[0] % 2 == 0 else 0 for sg in s), axis=tuple(range(len(s)))).copy()  # flip сдвигает центр на 1 при чётном L
    return out

def tvel(u, p, s): return np.array([s[k] * u[p[k]] for k in range(len(u))])

def best_T(u, theta):
    best = None
    for p, s in signed_perms(len(u)):
        u2 = tvel(u, p, s); ang = np.degrees(np.arccos(np.clip(u @ u2 / (u @ u), -1, 1)))
        if best is None or abs(ang - theta) < abs(best[2] - theta): best = (p, s, ang)
    return best

def classify(log, M1, v_in, L, every):
    """log: снимки (t, comps[(mass, cen)]) после столкновения."""
    nend = [len(x[1]) for x in log[-4:]]; n = int(np.median(nend)); nmax = max(len(x[1]) for x in log)
    res = dict(n_end=n, n_max=nmax)
    if n == 0: res["outcome"] = "аннигиляция/рассеяние массы"; return res
    if n == 1: res["outcome"] = "слияние"; res["mass_ratio"] = log[-1][1][0][0] / M1; return res
    if n > 2: res["outcome"] = "рождение/фрагментация"; return res
    # два объекта: скорости по последним снимкам (сопоставление по ближайшему центру)
    a = log[-10][1]; b = log[-1][1]
    if len(a) != 2: a = log[-5][1]
    vs = []
    for m, cen in b:
        j = int(np.argmin([np.linalg.norm((cen - c0 + L / 2) % L - L / 2) for _, c0 in a]))
        dt = (len(log) - 1 - (len(log) - 10 if len(log[-10][1]) == 2 else len(log) - 5)) * every
        vs.append(((cen - a[j][1] + L / 2) % L - L / 2) / dt)
    cosm = [max(float(np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w) + 1e-12)) for w in v_in) for v in vs]
    res.update(v_out=[v.tolist() for v in vs], cos_to_incoming=cosm, masses=[float(m) / M1 for m, _ in b])
    res["outcome"] = "прохождение" if min(cosm) > 0.9 else ("отклонение (рассеяние)" if min(cosm) > 0.0 else "отскок")
    return res

def run_coll(c, N=5, L=32, seed=7, configs=(), sep=14.0, after=1500, every=50, batch=6, pool=2, save_cb=None, diam=18.0, frac=0.3, pre=1500, extra=0, full=False):
    import jax, jax.numpy as jnp, fl_glider as FG
    from scipy import ndimage
    t0 = time.time()
    st, v = soliton(c, N, L, seed, pre=pre); sp = float(np.linalg.norm(v)); u = v / sp; M1 = float(st["A"].sum())
    mask1 = st["A"] > 0.02
    out = dict(speed=sp, u=u.tolist(), M1=M1, runs=[], partial=True)
    jobs = []
    phase_cache = {}
    for cfg in configs:
        th, b, tau = cfg["theta"], cfg.get("b", 0.0), cfg.get("tau", 0)
        p, s, ang = best_T(u, th); u2 = tvel(u, p, s)
        if tau not in phase_cache: phase_cache[tau] = st if tau == 0 else _sim(c, st, tau, N, L, track=False)[0]
        S2 = transform(phase_cache[tau], p, s)
        rel = u - u2; nr = np.linalg.norm(rel)
        if nr < 1e-6: continue
        e = rel / nr                                   # p1 - p2 направлено против относительной скорости
        perp = np.random.default_rng(int(th) * 7 + int(b)).standard_normal(N); perp -= perp @ e * e; perp -= perp @ u * u if abs(u @ e) < 0.99 else 0; perp /= np.linalg.norm(perp) + 1e-12
        d1 = -0.5 * sep * e - 0.5 * b * perp; d2 = +0.5 * sep * e + 0.5 * b * perp   # стартовые смещения от центра
        r1 = {k: np.roll(x, tuple(int(round(z)) for z in d1), axis=tuple(range(N))) for k, x in st.items()}
        r2 = {k: np.roll(x, tuple(int(round(z)) for z in d2), axis=tuple(range(N))) for k, x in S2.items()}
        m2 = r2["A"] > 0.02; m1 = r1["A"] > 0.02
        two = {k: np.where(m2 & ~m1, r2[k], r1[k]) for k in st}; two["A"] = r1["A"] + r2["A"]
        t_hit = sep / (sp * nr)
        t_cls = t_hit + min(after, (diam + frac * L) / (sp * nr))   # классифицируем до того, как продукты обернутся по тору и встретятся снова
        jobs.append(dict(cfg=dict(cfg, theta_real=float(ang)), init=two, t_hit=float(t_hit), t_cls=float(t_cls), v_in=[u * sp, u2 * sp], overlap=int((m1 & m2).sum())))
    for i0 in range(0, len(jobs), batch):
        J = jobs[i0:i0 + batch]; B = len(J)
        n_steps = int(min(max(j["t_cls"] for j in J) + extra, 6000)); n_steps -= n_steps % every
        KA = jnp.stack([jnp.asarray(FG.ring_kernel_np(L, N, c["R"], tuple(c["peaks"])))] * B); Ko = jnp.asarray(FG.ring_kernel_np(L, N, 3.0))
        P = jax.tree.map(lambda *a: jnp.stack(a), *[FG.params(c)] * B)
        s = jax.tree.map(lambda *a: jnp.stack([jnp.asarray(x) for x in a]), *[j["init"] for j in J])
        logs = [[] for _ in range(B)]; slices = [[] for _ in range(B)]
        for k in range(n_steps // every + 1):
            if k: s = FG.run_chunk(s, P, KA, Ko, every, mode=c.get("mode", "flow"), use_chi=True)
            A = np.asarray(s["A"])
            for bb in range(B):
                Ad = A[bb].reshape(sum(((L // pool, pool) for _ in range(N)), ())).mean(axis=tuple(range(1, 2 * N, 2)))
                comps, _ = FG.components(Ad, 0.05, min_mass=0.1 * M1 / pool ** N)
                logs[bb].append((k * every, [(cc["mass"] * pool ** N, cc["cen"] * pool) for cc in sorted(comps, key=lambda q: -q["mass"])[:6]]))
                if k % 2 == 0:
                    # 2D-проекция (среднее по осям 2..N) для анимации
                    slices[bb].append(A[bb].mean(axis=tuple(range(2, N))).astype(np.float16))
        for bb, j in enumerate(J):
            kc = min(len(logs[bb]), int(j["t_cls"] // every) + 1)
            res = classify2(logs[bb][:kc], M1, j["v_in"], L, every, j["t_hit"]) if full else classify(logs[bb][:kc], M1, j["v_in"], L, every)
            res["n_final"] = len(logs[bb][-1][1]); res["mass_final"] = [float(m) / M1 for m, _ in logs[bb][-1][1]]
            out["runs"].append(dict(cfg=j["cfg"], t_hit=j["t_hit"], t_cls=j["t_cls"], overlap0=j["overlap"], n_steps=n_steps, ncomp=[len(x[1]) for x in logs[bb]],
                                    log=[(t, [(float(m), list(map(float, cc))) for m, cc in comps]) for t, comps in logs[bb]],
                                    proj=slices[bb], **res))
        if save_cb: save_cb(out)
    out["partial"] = False; out["t_total"] = time.time() - t0
    out["summary"] = [(r["cfg"]["theta"], r["cfg"].get("b", 0), r["cfg"].get("tau", 0), r["outcome"], r["n_end"]) for r in out["runs"]]
    return out

def _track_vel(log, k0, k1, L, every):
    """Скорости двух объектов между снимками k0 и k1 (оба с 2 компонентами) — сопоставление по ближайшему центру с шагом по снимкам."""
    cur = [np.array(c) for _, c in log[k0][1]]; disp = [np.zeros_like(cur[0]) for _ in cur]
    for k in range(k0 + 1, k1 + 1):
        nxt = [np.array(c) for _, c in log[k][1]]
        if len(nxt) != len(cur): return None
        for i in range(len(cur)):
            d = [((n - cur[i] + L / 2) % L - L / 2) for n in nxt]; j = int(np.argmin([np.linalg.norm(x) for x in d]))
            disp[i] = disp[i] + d[j]; cur[i] = nxt[j]
    return [x / ((k1 - k0) * every) for x in disp]

def classify2(log, M1, v_in, L, every, t_hit, bound_min=150):
    """Классификация по всей траектории: слияние (до конца), захват→переизлучение, прохождение/отклонение/отскок, обмен массой, фрагментация."""
    n = np.array([len(c) for _, c in log]); kh = int(t_hit // every); T = len(n)
    tail = n[-6:]; n_end = int(np.median(tail)); res = dict(n_end=n_end, n_max=int(n.max()))
    one = (n[kh // 2:] == 1); bound = int(one.sum()) * every          # сколько шагов после сближения был один объект
    res["bound_steps"] = bound
    first_split = None
    if one.any():
        k1 = kh // 2 + int(np.argmax(one)); after = np.where(n[k1:] >= 2)[0]
        first_split = k1 + int(after[0]) if len(after) else None
    res["t_split"] = None if first_split is None else first_split * every
    if n_end == 0: res["outcome"] = "аннигиляция"; return res
    if n_end == 1:
        res["outcome"] = "слияние"; res["mass_ratio"] = sum(m for m, _ in log[-1][1]) / M1; return res
    if n_end > 2: res["outcome"] = "рождение/фрагментация"; return res
    # два объекта в конце: последний непрерывный отрезок с 2 компонентами
    k = T - 1
    while k > 0 and n[k - 1] == 2: k -= 1
    k0 = max(k + 3, T - 12); vs = _track_vel(log, k0, T - 1, L, every) if T - 1 - k0 >= 3 else None
    ms = sorted([float(m) / M1 for m, _ in log[-1][1]], reverse=True); res["masses"] = ms
    if vs is not None:
        cos = [max(float(np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w) + 1e-12)) for w in v_in) for v in vs]
        res.update(v_out=[v.tolist() for v in vs], speeds_out=[float(np.linalg.norm(v)) for v in vs], cos_to_incoming=cos,
                   angle_out=float(np.degrees(np.arccos(np.clip(np.dot(vs[0], vs[1]) / (np.linalg.norm(vs[0]) * np.linalg.norm(vs[1]) + 1e-12), -1, 1)))))
    if bound >= bound_min: base = "захват→переизлучение"
    elif vs is None: base = "два объекта"
    else: base = "прохождение" if min(cos) > 0.9 else ("отклонение" if min(cos) > 0.0 else "отскок")
    if ms[-1] < 0.6: base += " + обмен массой"
    res["outcome"] = base; return res
