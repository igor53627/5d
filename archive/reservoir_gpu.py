#!/usr/bin/env python3
import _path  # noqa: F401
"""
Reservoir-бенчмарк Feigenbaum-Lenia на JAX (GPU/CPU): гомеостаз по условному показателю Ляпунова.

Один вызов run_batch() = одна задача (mg / lorenz / mc), один размер решётки, произвольный список "элементов"
(режим × гиперпараметры × сид × реплика). Всё считается одним vmap-батчем.
  реплика 0 — для ридаута; реплика 1 — та же маска входа и те же клетки ридаута, но другие начальные условия
  (и другой стартовый геном) -> тест echo-state: корреляция состояний реплик под общим входом.
"""
import json, os, time, itertools
import numpy as np
import jax, jax.numpy as jnp
import fl_jax as F
from reservoir_experiment import make_input, targets, split, ridge_eval

SPLIT = dict(washout=400, ntr=1400, nva=400, nte=400)

def element_list(spec):
    """spec: список dict(method, a, gamma, eps, lam_target, kappa, c0, seed). Возвращает элементы с репликами."""
    out = []
    for e in spec:
        for rep in range(e.get("replicas", 1)):
            out.append(dict(e, rep=rep))
    return out

def run_batch(task, N, L, elems, nread=400, spp=10, dt=0.05, prewarm=60.0, R=3.0, Rc=3.0, chunk=None, verbose=True):
    sp = SPLIT; n_total = sp["washout"] + sp["ntr"] + sp["nva"] + sp["nte"] + 60
    u = make_input(task, n_total, seed=0)
    tr, va, te = split(n_total, sp["washout"], sp["ntr"], sp["nva"], sp["nte"])
    shape = (L,) * N
    K = jnp.asarray(F.lenia_kernel_np(L, N, R)); Kc = jnp.asarray(F.lenia_kernel_np(L, N, Rc))
    Ks = jnp.asarray(F.lenia_kernel_np(L, N, max(2.0, R)))
    ncell = L ** N
    t0 = time.time()
    B = len(elems)
    # параметры
    Ps, sts, sels, aw, aa = [], [], [], [], []
    for e in elems:
        a = e["a"]; gsign = 1.0   # эмпирически λc растёт с c для обоих вариантов клетки (см. sweep_lamc)
        Ps.append(F.make_params(a=a, gamma=e["gamma"], eps=e["eps"], lam_target=e.get("lam_target", -0.02),
                                kappa=e.get("kappa", 0.0), gsign=gsign, dt=dt, eta=e.get("eta", 0.1),
                                lam_scale=e.get("lam_scale", 0.01), tau_lam=e.get("tau_lam", 10.0)))
        adaptive = e["method"] in ("cond-adapt", "cond-freeze", "cond-freeze-shuf", "own-adapt")
        init_key = jax.random.PRNGKey(1000 * e["seed"] + 17 * e["rep"] + 1)
        win_key = jax.random.PRNGKey(5000 + e["seed"])
        hetero = adaptive or e["method"] == "fixed-hetero"   # fixed-hetero: случайное гладкое поле c, без адаптации
        st = F.init_state(init_key, shape, c_init=e.get("c0", 5.0), c_spread=1.2 if hetero else 0.0,
                          K_smooth=Ks, in_frac=1.0, win_key=win_key)
        sts.append(st)
        sels.append(np.random.default_rng(100 + e["seed"]).choice(ncell, nread, replace=False))
        aw.append(1.0 if adaptive else 0.0); aa.append(1.0 if e["method"] in ("cond-adapt", "own-adapt") else 0.0)
    rules = {("own" if e["method"] == "own-adapt" else "cond") for e in elems}
    assert len(rules) == 1; rule = rules.pop()
    P = F.stack_params(Ps); st = F.stack_params(sts)
    sel = jnp.asarray(np.stack(sels)); aw = jnp.asarray(aw, jnp.float32); aa = jnp.asarray(aa, jnp.float32)
    st = F.run_free(st, P, K, Kc, int(prewarm / dt), rule=rule, sync=False, adapt=aw)   # прогрев без входа
    st["t"] = jnp.zeros_like(st["t"]) + 1e6                                     # warmup уже пройден
    runner = F.make_reservoir_runner(rule, False, spp, avg=True)
    W = sp["washout"]
    st, f1, d1 = runner(st, P, K, Kc, jnp.asarray(u[:W]), sel, aw, aw, W)
    # абляция "cond-freeze-shuf": геном, выращенный гомеостазом, случайно переставлен по решётке
    # (то же распределение c, но без пространственной привязки) — проверка, что важна именно раскладка.
    shuf = np.array([e["method"] == "cond-freeze-shuf" for e in elems])
    if shuf.any():
        c = np.asarray(st["c"]).reshape(B, -1).copy()
        for i in np.where(shuf)[0]:
            c[i] = c[i][np.random.default_rng(777 + i).permutation(c.shape[1])]
        st["c"] = jnp.asarray(c.reshape(st["c"].shape))
    st, f2, d2 = runner(st, P, K, Kc, jnp.asarray(u[W:]), sel, aa, aa, 0)
    feats = np.concatenate([np.asarray(f1), np.asarray(f2)], 1); diag = np.concatenate([np.asarray(d1), np.asarray(d2)], 0)
    del f1, f2
    t_sim = time.time() - t0
    if verbose: print(f"[{task} N={N} L={L} B={B}] sim {t_sim:.1f}s", flush=True)
    Y, tnames = targets(task, u, np.arange(n_total - 60) if task != "mc" else np.arange(n_total))
    nrows = len(Y)
    out = []
    idx = {}
    for i, e in enumerate(elems):
        idx.setdefault(_key(e), {})[e["rep"]] = i
    for k, reps in idx.items():
        i0 = reps[0]; e = elems[i0]
        X = feats[i0, :nrows].astype(np.float64)
        bad = not np.isfinite(X).all() or np.abs(X).max() > 1e5
        r = ridge_eval(X, Y, tr, va, te, task)
        r.pop("pred_test"); r.pop("true_test")
        r.update({kk: e[kk] for kk in e if kk != "rep"})
        sl = slice(sp["washout"], None)
        r.update(lam=float(diag[sl, i0, 0].mean()), lamc=float(diag[sl, i0, 1].mean()), c_mean=float(diag[sl, i0, 2].mean()),
                 chaos=float(diag[sl, i0, 3].mean()), diverged=bool(bad), targets=tnames)
        if 1 in reps:   # echo-state: корреляция x-признаков двух реплик по клеткам в каждый момент test
            Xa = feats[i0, te, :nread]; Xb = feats[reps[1], te, :nread]
            Xa = Xa - Xa.mean(1, keepdims=True); Xb = Xb - Xb.mean(1, keepdims=True)
            cc = (Xa * Xb).sum(1) / (np.sqrt((Xa ** 2).sum(1) * (Xb ** 2).sum(1)) + 1e-12)
            r["twin_corr"] = float(np.nanmean(cc))
        out.append(r)
    return dict(task=task, N=N, L=L, B=B, t_sim=t_sim, t_total=time.time() - t0, runs=out)

def _key(e):
    return tuple(sorted((k, v) for k, v in e.items() if k != "rep"))

# ----------------------------------------------------------------------------- grids
def tuning_spec(seeds=(0,), replicas=1):
    spec = []
    for a in (0.2, -0.05):
        for g, eps in itertools.product((1.0, 2.0, 3.0), (0.1, 0.3)):
            for s in seeds:
                for lt, kap in itertools.product((-0.05, -0.03, -0.02, -0.01, -0.003), (0.0, 0.5)):
                    for m in ("cond-adapt", "cond-freeze"):
                        spec.append(dict(method=m, a=a, gamma=g, eps=eps, lam_target=lt, kappa=kap, c0=5.0, seed=s, replicas=replicas))
                for c in (2.5, 2.8, 3.5, 4.2, 5.0, 6.0):
                    spec.append(dict(method="fixed", a=a, gamma=g, eps=eps, c0=c, seed=s, replicas=replicas))
    return spec

def family(r):
    """Группа сравнения: метод × вариант клетки."""
    v = "Рёсслер a=0.2" if r["a"] > 0 else "затух. a=-0.05"
    return f"{r['method']} | {v}"

def select_best(tune_res):
    """Для каждой задачи, цели и семейства — лучшие гиперпараметры по val (ровно как у ESN — по цели)."""
    best = {}
    for res in tune_res:
        task = res["task"]
        for r in res["runs"]:
            if r["diverged"]: continue
            fam = family(r)
            vals = r["val"]
            for j, tn in enumerate(r["targets"] if task != "mc" else ["MC"]):
                v = (-np.clip(1 - np.array(vals) ** 2, 0, 1).sum()) if task == "mc" else vals[j]
                k = (task, tn, fam)
                if k not in best or v < best[k][0]:
                    best[k] = (float(v), {kk: r[kk] for kk in ("method", "a", "gamma", "eps", "lam_target", "kappa", "c0") if kk in r})
    return best

def tuning_spec_ext(seeds=(0,)):
    """Расширение сетки (после первого прохода лучшие точки легли на край: γ=3, ε=0.3, λ*=−0.05)."""
    spec = []
    for a in (0.2, -0.05):
        for g, eps in itertools.product((3.0, 4.0), (0.3, 0.5)):
            for s in seeds:
                for lt in (-0.1, -0.07, -0.05, -0.03, -0.01):
                    for m in ("cond-adapt", "cond-freeze"):
                        spec.append(dict(method=m, a=a, gamma=g, eps=eps, lam_target=lt, kappa=0.0, c0=5.0, seed=s, replicas=1))
                for c in (2.0, 2.2, 2.5, 2.8, 3.5, 5.0):
                    spec.append(dict(method="fixed", a=a, gamma=g, eps=eps, c0=c, seed=s, replicas=1))
    return spec

def tuning_spec_all(seeds=(0,)):
    spec = tuning_spec(seeds) + tuning_spec_ext(seeds)
    for a in (0.2, -0.05):
        for g, eps in itertools.product((1.0, 2.0, 3.0, 4.0), (0.1, 0.3, 0.5)):
            for c in (2.5, 3.5, 5.0):
                spec.append(dict(method="fixed-hetero", a=a, gamma=g, eps=eps, c0=c, seed=seeds[0], replicas=1))
    out = []
    for e in spec:
        if e not in out: out.append(e)
    return out

def final_spec(best, seeds=(0, 1, 2, 3, 4), replicas=2):
    """Для каждой задачи: объединение лучших по val конфигураций (по целям) + абляция shuf для каждой лучшей freeze."""
    per_task = {}
    for (task, tn, fam), (v, cfg) in best.items():
        lst = per_task.setdefault(task, [])
        cfgs = [cfg]
        if cfg["method"] == "cond-freeze":
            cfgs.append(dict(cfg, method="cond-freeze-shuf"))
        for c in cfgs:
            if c not in lst: lst.append(c)
    out = {}
    for task, cfgs in per_task.items():
        out[task] = [dict(c, seed=s, replicas=replicas) for c in cfgs for s in seeds]
    return out
