import sys; from pathlib import Path; _root = next(p for p in [Path(__file__).resolve().parent] + list(Path(__file__).resolve().parents) if (p / "pyproject.toml").exists()); sys.path.insert(0, str(_root / "src"))
"""Часть 8: CMA-ES на «репликацию» — столкновения двух солитонов, дающие ≥3 выживших жёстких движущихся солитона.
Режим flow_iso_muscl. Особь оценивается в 4D (L=48 по умолчанию): одиночный солитон (скорость, жёсткость) + набор столкновений в одном vmap-батче."""
import os, time, pickle, numpy as np
from glider_es5 import CMA, _single_state
import glider_coll as GC

GENES = [("mu", 0.10, 0.40), ("sigma", 0.05, 0.35), ("chi", 0.0, 1.0), ("beta", -0.5, 1.5), ("theta", 0.8, 4.0),
         ("gam", 0.0, 1.0), ("log_tau", 0.0, 1.5), ("h1", 0.05, 1.0), ("h2", 0.05, 1.0), ("h3", 0.05, 1.0),
         ("dtA", 0.1, 0.45), ("eps", 0.03, 0.4), ("c_hi", 3.0, 6.0)]
def decode(z, R=6.0):
    z = np.clip(z, 0, 1); d = {g: lo + (hi - lo) * float(z[i]) for i, (g, lo, hi) in enumerate(GENES)}
    hs = [d["h1"], d["h2"], d["h3"]]; m = max(hs)
    return dict(mode="flow_iso_muscl", R=R, mu=d["mu"], sigma=d["sigma"], chi=d["chi"], beta=d["beta"], theta=d["theta"], gam=d["gam"],
                tau_c=10 ** d["log_tau"], peaks=[h / m for h in hs], dtA=d["dtA"], eps=d["eps"], c_hi=d["c_hi"], asym=0.0, density=0.6)
def encode(c):
    v = dict(c); v["log_tau"] = np.log10(v.get("tau_c", 2.0)); pk = (list(c["peaks"]) + [c["peaks"][-1]] * 3)[:3]
    for i in range(3): v[f"h{i+1}"] = pk[i]
    return np.array([np.clip((v[g] - lo) / (hi - lo), 0, 1) for g, lo, hi in GENES])

CONFIGS = [dict(theta=180), dict(theta=180, tau=50), dict(theta=180, tau=100), dict(theta=90, tau=50), dict(theta=90, tau=100), dict(theta=150, b=6)]

def _pool(A, N, L, pool=2):
    import jax.numpy as jnp
    B = A.shape[0]; sh = (B,) + sum(((L // pool, pool) for _ in range(N)), ())
    return A.reshape(sh).mean(axis=tuple(range(2, 2 * N + 1, 2)))

def _centered(A, cen):
    return np.roll(A, tuple(int(A.shape[0] // 2 - round(x)) for x in cen), axis=tuple(range(A.ndim)))

def _corr(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))

def evaluate_rep(c, N=4, L=48, seed=7, configs=CONFIGS, sep=22, horizon=1400, pre=1000, every=50, pool=2, keep=False, fv=1):
    """-> (fitness, info). Батч: [одиночный] + столкновения."""
    import jax, jax.numpy as jnp, fl_glider as FG
    t0 = time.time(); info = dict(c=c)
    try:
        st, v = GC.soliton(c, N, L, seed, pre=pre)
    except Exception as e:
        return 0.0, dict(info, err=str(e))
    sp = float(np.linalg.norm(v)); M1 = float(st["A"].sum()); info.update(speed_pre=sp, M1=M1)
    if not np.isfinite(sp) or sp < 0.01 or M1 < 1: return 0.0, dict(info, dead=True)
    u = v / sp; inits = [st]; jobs = []
    for cfg in configs:
        th, b, tau = cfg["theta"], cfg.get("b", 0.0), cfg.get("tau", 0)
        p, s, ang = GC.best_T(u, th); u2 = GC.tvel(u, p, s)
        S2 = GC.transform(st if tau == 0 else GC._sim(c, st, tau, N, L, track=False)[0], p, s)
        rel = u - u2; nr = np.linalg.norm(rel); e = rel / nr
        perp = np.random.default_rng(int(th) * 7 + int(b)).standard_normal(N); perp -= perp @ e * e; perp /= np.linalg.norm(perp) + 1e-12
        d1 = -0.5 * sep * e - 0.5 * b * perp; d2 = 0.5 * sep * e + 0.5 * b * perp
        r1 = {k: np.roll(x, tuple(int(round(z)) for z in d1), axis=tuple(range(N))) for k, x in st.items()}
        r2 = {k: np.roll(x, tuple(int(round(z)) for z in d2), axis=tuple(range(N))) for k, x in S2.items()}
        m1 = r1["A"] > 0.02; m2 = r2["A"] > 0.02
        two = {k: np.where(m2 & ~m1, r2[k], r1[k]) for k in st}; two["A"] = r1["A"] + r2["A"]
        inits.append(two); jobs.append(dict(cfg=cfg, t_hit=float(sep / (sp * nr))))
    B = len(inits); n_steps = int(max(j["t_hit"] for j in jobs) + horizon); n_steps -= n_steps % every
    KA = jnp.stack([jnp.asarray(FG.ring_kernel_np(L, N, c["R"], tuple(c["peaks"])))] * B); Ko = jnp.asarray(FG.ring_kernel_np(L, N, 3.0))
    P = jax.tree.map(lambda *a: jnp.stack(a), *[FG.params(c)] * B)
    s = jax.tree.map(lambda *a: jnp.stack([jnp.asarray(x) for x in a]), *inits)
    logs = [[] for _ in range(B)]; snaps = []; projs = [[] for _ in range(B)]
    for k in range(n_steps // every + 1):
        if k: s = FG.run_chunk(s, P, KA, Ko, every, mode=c["mode"], use_chi=True)
        Ap = np.asarray(_pool(s["A"], N, L, pool))
        if not np.all(np.isfinite(Ap)): return 0.0, dict(info, nan=True)
        for bb in range(B):
            comps, _ = FG.components(Ap[bb], 0.05, min_mass=0.1 * M1 / pool ** N)
            logs[bb].append([(cc["mass"] * pool ** N / M1, cc["cen"] * pool) for cc in sorted(comps, key=lambda q: -q["mass"])[:8]])
        if k % 5 == 0 and len(logs[0][-1]): snaps.append(_centered(Ap[0], logs[0][-1][0][1] / pool))
        if keep and k % 2 == 0:
            for bb in range(B): projs[bb].append(np.asarray(s["A"][bb]).mean(axis=tuple(range(2, N))).astype(np.float16))
    # одиночный: скорость и жёсткость
    C0 = [lg[0][1] for lg in logs[0] if len(lg)]; n0 = [len(lg) for lg in logs[0]]
    if len(C0) < 10 or max(n0[-5:]) != 1 or min(n0[-5:]) != 1: single_ok = 0.0; v1 = 0.0; r250 = 0.0
    else:
        Cc = np.array(C0[len(C0) // 3:]); d = (np.diff(Cc, axis=0) + L / 2) % L - L / 2; v1 = float(np.linalg.norm(d.sum(0)) / (len(d) * every))
        r250 = float(np.mean([_corr(snaps[i], snaps[i + 1]) for i in range(len(snaps) // 3, len(snaps) - 1)])) if len(snaps) > 3 else 0.0
        single_ok = float(np.clip(v1 / 0.03, 0, 1) * np.clip((r250 - 0.75) / 0.1, 0, 1))
    info.update(v_single=v1, r250_single=r250, single_ok=single_ok)
    # столкновения: «хорошие» продукты в конце — масса ≥0.6 M1, стабильная масса, движется
    lag = 300 // every; sc = []; runs = []
    for j, bb in zip(jobs, range(1, B)):
        lg = logs[bb]; end = lg[-1]; prev = lg[-1 - lag]
        good = 0; masses = [m for m, _ in end]
        for m, cen in end:
            if m < 0.6: continue
            # ближайший объект lag снимков назад
            dd = [np.linalg.norm((cen - c0 + L / 2) % L - L / 2) for _, c0 in prev] if prev else [0]
            j0 = int(np.argmin(dd)); mv = dd[j0] / (lag * every)
            m_prev = prev[j0][0] if prev else 0
            if mv > 0.4 * max(v1, 0.02) and abs(m - m_prev) / m < 0.1: good += 1
        n_end = len(end)
        m3 = sorted(masses)[-3] if n_end >= 3 else 0.0
        if fv == 2:   # гладкая версия: награда за массу третьего объекта, бонус за ≥3 «хороших»
            f = (0.3 + 0.7 * min(1.0, m3 / 0.6) if n_end >= 3 else (0.1 if n_end == 2 else 0.0)) + (0.5 if good >= 3 else 0.0)
        elif good >= 3: f = 1.0 + 0.25 * (good - 3)
        elif good == 2: f = 0.25 + (0.25 * min(1.0, sorted(masses)[-3] / 0.6) if n_end >= 3 else 0.0)
        elif n_end >= 3: f = 0.15
        else: f = 0.0          # слияние / одиночный объект
        sc.append(f); runs.append(dict(cfg=j["cfg"], t_hit=j["t_hit"], good=good, n_end=n_end, masses=masses, ncomp=[len(x) for x in lg], score=f,
                                       proj=projs[bb] if keep else None, log=lg if keep else None))
    if keep == "daughters":
        for r, bb in zip(runs, range(1, B)):
            if r["n_end"] >= 3:
                try: r["daughters"] = daughter_test(c, {k: np.asarray(x[bb]) for k, x in s.items()}, M1, N, L)
                except Exception as e: r["daughters_err"] = str(e)
    fit = float(np.mean(sc) * (0.3 + 0.7 * single_ok)) if single_ok > 0 else 0.0
    info.update(fit=fit, scores=sc, runs=runs, n_rep=int(sum(r["good"] >= 3 for r in runs)), t=time.time() - t0, n_steps=n_steps,
                single_log=logs[0] if keep else None, single_proj=projs[0] if keep else None)
    return fit, info

def run_rep_cma(seed_cand, name="S1", gens=40, lam=8, sigma0=0.08, max_wall=5400, seed=0, N=4, L=48, save_cb=None, extra_seeds=(), fv=1, configs=None):
    t_start = time.time(); ckpt = f"/vol/rep_ckpt_{name}.pkl"
    if os.path.exists(ckpt):
        S = pickle.load(open(ckpt, "rb")); es, hist, best = S["es"], S["hist"], S["best"]
    else:
        es = CMA(encode(seed_cand), sigma0, lam, seed=seed); hist = []; best = []
    while es.gen < gens and time.time() - t_start < max_wall:
        tg = time.time(); X = es.ask()
        if es.gen == 0:
            X[0] = encode(seed_cand)
            for j, c in enumerate(extra_seeds[: lam - 1]): X[j + 1] = encode(c)
            es._Y = (X - es.m) / es.sigma
        f = np.zeros(lam); infos = []
        for j in range(lam):
            fj, inf = evaluate_rep(decode(X[j]), N=N, L=L, seed=7 + es.gen % 3, fv=fv, configs=configs or CONFIGS)
            f[j] = fj; inf.pop("c", None); infos.append(inf)
        es.tell(X, f)
        for j in range(lam):
            hist.append(dict(gen=es.gen - 1, fit=float(f[j]), z=X[j].tolist(), info={k: v for k, v in infos[j].items() if k not in ("single_log", "single_proj")}))
            for r in hist[-1]["info"].get("runs", []): r.pop("proj", None); r.pop("log", None)
        best = sorted(best + [(float(f[j]), X[j].tolist()) for j in range(lam)], key=lambda t: -t[0])[:8]
        gi = int(np.argmax(f)); bi = infos[gi]
        print(f"[{name}] gen {es.gen-1}: best {f[gi]:.3f} mean {np.mean(f):.3f} sig {es.sigma:.3f} | n_rep {bi.get('n_rep')} v1 {bi.get('v_single',0):.3f} r250 {bi.get('r250_single',0):.2f} | {time.time()-tg:.0f}s", flush=True)
        pickle.dump(dict(es=es, hist=hist, best=best), open(ckpt + ".tmp", "wb")); os.replace(ckpt + ".tmp", ckpt)
        if save_cb: save_cb(dict(hist=hist, best=best, gen=es.gen, sigma=es.sigma, mean=es.m.tolist(), partial=True))
    return dict(hist=hist, best=best, gen=es.gen, sigma=es.sigma, mean=es.m.tolist(), partial=es.gen < gens,
                summary=[round(b[0], 3) for b in best[:5]], t_total=time.time() - t_start)

def run_rep_eval(cands, N=4, L=48, seeds=(7,), keep=False, save_cb=None, **kw):
    out = dict(runs={}, partial=True)
    for name, c in cands.items():
        for sd in seeds:
            f, inf = evaluate_rep(c, N=N, L=L, seed=sd, keep=keep, **kw); out["runs"][f"{name}|{sd}"] = dict({k: v for k, v in inf.items() if k not in ("c", "fit")}, fit=f)
            if save_cb: save_cb(out)
    out["partial"] = False; out["summary"] = {k: (round(v["fit"], 3), v.get("n_rep"), v.get("scores")) for k, v in out["runs"].items()}
    return out

def _extract(st, lab, i, L, N):
    """Вырезать объект i (метка на пулинговой сетке ×2) в отдельное поле, остальное — фон."""
    m = np.kron(lab == i, np.ones((2,) * N, bool)) if lab.shape[0] * 2 == L else (lab == i)
    from scipy import ndimage
    m = ndimage.binary_dilation(m, iterations=2)
    out = {k: np.where(m, x, (x if k != "A" else 0 * x)) for k, x in st.items()}
    out["A"] = np.where(m, st["A"], 0.0).astype(np.float32); return out

def daughter_test(c, st_end, M1, N, L, steps=1500, every=50):
    """Каждого продукта столкновения (масса ≥0.3 M1) запускаем в одиночку: скорость, жёсткость, масса, геном c."""
    import fl_glider as FG
    Ap = st_end["A"].reshape(sum(((L // 2, 2) for _ in range(N)), ())).mean(axis=tuple(range(1, 2 * N, 2)))
    comps, lab = FG.components(Ap, 0.05, min_mass=0.3 * M1 / 2 ** N)
    ids = [i for i in np.unique(lab) if i > 0]; res = []
    for i in ids:
        d = _extract(st_end, lab, i, L, N); m = float(d["A"].sum()) / M1
        if m < 0.3: continue
        msk = d["A"] > 0.1; cin = float(d["c"][msk].mean()) if msk.any() else np.nan
        cA = float(np.corrcoef(d["c"][msk], d["A"][msk])[0, 1]) if msk.sum() > 10 else np.nan
        fin, cens = GC._sim(c, d, steps, N, L, every)
        C = np.array(cens[len(cens) // 3:]); dd = (np.diff(C, axis=0) + L / 2) % L - L / 2; v = float(np.linalg.norm(dd.sum(0)) / (len(dd) * every))
        Ap2 = fin["A"].reshape(sum(((L // 2, 2) for _ in range(N)), ())).mean(axis=tuple(range(1, 2 * N, 2)))
        cc2, _ = FG.components(Ap2, 0.05, min_mass=0.1 * M1 / 2 ** N)
        res.append(dict(mass=m, c_mean_in=cin, corr_cA=cA, v_alone=v, n_after=len(cc2), mass_after=float(fin["A"].sum()) / M1))
    return res

def run_rep_verify(cands, N=4, L=64, seeds=(7, 8), configs=None, sep=24, horizon=1800, daughters=True, save_cb=None):
    """Частота репликации по сетке конфигураций + тест дочерних (одиночный прогон каждой) для конфигураций с ≥3 объектами."""
    import jax, jax.numpy as jnp, fl_glider as FG
    configs = configs or [dict(theta=t, tau=tau, b=b) for t in (180, 150, 120, 90) for tau in (0, 50, 100, 150) for b in (0,)] + [dict(theta=180, b=6, tau=tau) for tau in (0, 50, 100)]
    out = dict(runs={}, partial=True)
    for name, c in cands.items():
        for sd in seeds:
            for i0 in range(0, len(configs), 6):
                f, inf = evaluate_rep(c, N=N, L=L, seed=sd, configs=configs[i0:i0 + 6], sep=sep, horizon=horizon, keep="daughters" if daughters else True)
                key = f"{name}|{sd}|{i0}"; inf.pop("c", None); out["runs"][key] = inf
                if save_cb: save_cb(out)
    out["partial"] = False
    out["summary"] = {k: (v.get("n_rep"), [r["good"] for r in v.get("runs", [])]) for k, v in out["runs"].items()}
    return out

def run_cascade(cands, N=4, L=96, n0=4, steps=8000, seed=3, every=100, save_cb=None):
    """Популяционный тест: n0 копий солитона в случайных местах/ориентациях в большом поле; число объектов и массы во времени."""
    import jax, jax.numpy as jnp, fl_glider as FG
    out = dict(runs={}, partial=True)
    for name, c in cands.items():
        st, v = GC.soliton(c, N, 64, seed, pre=1000); M1 = float(st["A"].sum()); rng = np.random.default_rng(seed)
        big = {k: np.zeros((L,) * N, np.float32) for k in st}; pad = (L - 64) // 2
        big["c"][:] = c.get("c_lo", 2.5)
        perms = list(GC.signed_perms(N))
        for i in range(n0):
            p, sg = perms[rng.integers(len(perms))]; S = GC.transform(st, p, sg)
            emb = {k: np.zeros((L,) * N, np.float32) + (c.get("c_lo", 2.5) if k == "c" else 0) for k in st}
            sl = tuple(slice(pad, pad + 64) for _ in range(N))
            for k in st: emb[k][sl] = S[k]
            sh = tuple(int(x) for x in rng.integers(0, L, N))
            emb = {k: np.roll(x, sh, axis=tuple(range(N))) for k, x in emb.items()}
            m = emb["A"] > 0.02
            for k in st: big[k] = np.where(m, emb[k], big[k]) if k != "A" else big[k] + emb[k]
        KA = jnp.asarray(FG.ring_kernel_np(L, N, c["R"], tuple(c["peaks"])))[None]; Ko = jnp.asarray(FG.ring_kernel_np(L, N, 3.0))
        P = jax.tree.map(lambda a: a[None], FG.params(c)); s = jax.tree.map(lambda a: jnp.asarray(a)[None], big)
        log = []; proj = []
        for k in range(steps // every + 1):
            if k: s = FG.run_chunk(s, P, KA, Ko, every, mode=c["mode"], use_chi=True)
            Ap = np.asarray(_pool(s["A"], N, L, 2))[0]
            comps, _ = FG.components(Ap, 0.05, min_mass=0.1 * M1 / 16)
            log.append((k * every, sorted([cc["mass"] * 16 / M1 for cc in comps], reverse=True)))
            if k % 2 == 0: proj.append(np.asarray(s["A"][0]).mean(axis=tuple(range(2, N))).astype(np.float16))
            if k % 10 == 0:
                out["runs"][name] = dict(log=log, proj=proj, M1=M1, n0=n0, L=L, partial=True)
                if save_cb: save_cb(out)
        out["runs"][name] = dict(log=log, proj=proj, M1=M1, n0=n0, L=L)
    out["partial"] = False; out["summary"] = {k: [len(x[1]) for x in v["log"]][::10] for k, v in out["runs"].items()}
    return out
