#!/usr/bin/env python3
"""
Reservoir computing на Feigenbaum-Lenia vs ESN.
Задачи: Mackey-Glass (tau=17) прогноз на h шагов, Lorenz-x прогноз, линейная ёмкость памяти Джегера (MC).
Режимы FL: adapt (геном эволюционирует всё время), frozen c=2.5/2.8/5.0, adapt_freeze (адаптация на washout, потом заморозка).
Все режимы x сиды одной задачи считаются одним батчем (батч-ось + N-мерный FFT по осям решётки).

  python reservoir_experiment.py run   --task mg --tag main      # симуляция + ридж-ридаут, пишет results/<tag>_<task>.json
  python reservoir_experiment.py esn   --task mg --tag main      # ESN и линейный бейзлайн
  python reservoir_experiment.py report --tags main fix1 ...      # таблица + рисунок
"""
import argparse, json, os, sys, time
import numpy as np
from feigenbaum_lenia import lenia_kernel, growth

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

# ============================================================================ data
def mackey_glass(n, tau=17.0, dt=0.1, sample=1.0, seed=0, burn=1000):
    rng = np.random.default_rng(seed)
    d = int(round(tau / dt)); k = int(round(sample / dt))
    total = (n + burn) * k + d + 1
    x = np.empty(total); x[: d + 1] = 1.2 + 0.05 * rng.standard_normal(d + 1)
    f = lambda xt, xd: 0.2 * xd / (1 + xd ** 10) - 0.1 * xt
    for i in range(d, total - 1):                     # RK2 с запаздыванием (xd на полушаге — среднее)
        xd0, xd1 = x[i - d], x[i - d + 1]
        k1 = f(x[i], xd0); k2 = f(x[i] + dt * k1, xd1)
        x[i + 1] = x[i] + 0.5 * dt * (k1 + k2)
    return x[d + 1 + burn * k::k][:n]

def lorenz_x(n, dt=0.01, sample=0.05, seed=0, burn=2000):
    rng = np.random.default_rng(seed)
    s = np.array([1.0, 1.0, 1.0]) + 0.1 * rng.standard_normal(3)
    f = lambda s: np.array([10 * (s[1] - s[0]), s[0] * (28 - s[2]) - s[1], s[0] * s[1] - 8 / 3 * s[2]])
    k = int(round(sample / dt)); out = []
    for i in range((n + burn) * k):
        k1 = f(s); k2 = f(s + .5 * dt * k1); k3 = f(s + .5 * dt * k2); k4 = f(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if i % k == 0 and i // k >= burn: out.append(s[0])
    return np.array(out[:n])

TASKS = {
    "mg":     dict(horizons=[1, 10, 50]),
    "lorenz": dict(horizons=[1, 10]),
    "mc":     dict(delays=list(range(1, 81))),
}

def make_input(task, n, seed=0):
    if task == "mg": u = mackey_glass(n, seed=seed)
    elif task == "lorenz": u = lorenz_x(n, seed=seed)
    else: u = np.random.default_rng(1000 + seed).uniform(-1, 1, n)
    return ((u - u.mean()) / u.std()).astype(np.float32)

def targets(task, u, idx):
    """Матрица целей Y (len(idx) x n_targets) и имена."""
    if task == "mc":
        ks = TASKS["mc"]["delays"]; return np.stack([u[idx - k] for k in ks], 1), [f"d{k}" for k in ks]
    hs = TASKS[task]["horizons"]; return np.stack([u[idx + h] for h in hs], 1), [f"h{h}" for h in hs]

# ============================================================================ split
def split(T_total, washout, ntr, nva, nte):
    tr = np.arange(washout, washout + ntr)
    va = np.arange(tr[-1] + 1, tr[-1] + 1 + nva)
    te = np.arange(va[-1] + 1, va[-1] + 1 + nte)
    return tr, va, te

# ============================================================================ ridge readout
ALPHAS = np.logspace(-6, 3, 19)

def ridge_eval(X, Y, tr, va, te, task):
    """Стандартизация по train, SVD один раз, перебор alpha по val (для каждой цели отдельно), отчёт по test."""
    X = np.clip(np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)   # защита от разлёта
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Z = (X - mu) / sd
    Z = np.concatenate([Z, np.ones((len(Z), 1), Z.dtype)], 1)
    ym = Y[tr].mean(0)
    try:
        U, s, Vt = np.linalg.svd(Z[tr].astype(np.float64), full_matrices=False)
    except np.linalg.LinAlgError:
        U, s, Vt = np.linalg.svd(Z[tr].astype(np.float64) + 1e-9 * np.random.default_rng(0).standard_normal(Z[tr].shape), full_matrices=False)
    UtY = U.T @ (Y[tr] - ym)
    best = None
    for a in ALPHAS:
        W = Vt.T @ ((s / (s ** 2 + a))[:, None] * UtY)
        pv = Z[va] @ W + ym
        e = np.sqrt(((pv - Y[va]) ** 2).mean(0)) / Y[va].std(0)
        if best is None: best = (e.copy(), np.full(Y.shape[1], a), W.copy())
        else:
            m = e < best[0]; best[0][m] = e[m]; best[1][m] = a; best[2][:, m] = W[:, m]
    pt = Z[te] @ best[2] + ym
    nrmse = np.sqrt(((pt - Y[te]) ** 2).mean(0)) / Y[te].std(0)
    out = {"nrmse": nrmse.tolist(), "alpha": best[1].tolist(), "val": best[0].tolist()}
    if task == "mc":
        r2 = np.array([np.corrcoef(pt[:, j], Y[te][:, j])[0, 1] ** 2 for j in range(Y.shape[1])])
        out["mc_curve"] = np.nan_to_num(r2).tolist(); out["MC"] = float(np.nan_to_num(r2).sum())
    out["pred_test"] = pt[:300].tolist(); out["true_test"] = Y[te][:300].tolist()
    return out

# ============================================================================ batched Feigenbaum-Lenia reservoir
class BatchFL:
    def __init__(self, B, N, L, R, eps, gamma, c0, adapt, in_frac, seed, P):
        self.P = P; self.shape = (L,) * N; self.axes = tuple(range(1, N + 1))
        self.K = lenia_kernel(L, N, R)[None]; self.Kc = lenia_kernel(L, N, P["Rc"])[None]
        rng = np.random.default_rng(seed)
        S = (B,) + self.shape; f32 = np.float32
        th = rng.uniform(0, 2 * np.pi, S); rad = rng.uniform(2, 6, S)
        self.x = (rad * np.cos(th)).astype(f32); self.y = (rad * np.sin(th)).astype(f32)
        self.z = rng.uniform(0, .5, S).astype(f32)
        self.c = np.broadcast_to(np.asarray(c0, f32).reshape((B,) + (1,) * N), S).copy()
        self.adapt = np.asarray(adapt, bool).reshape((B,) + (1,) * N)
        v = rng.standard_normal((3,) + S).astype(f32); v /= np.sqrt((v ** 2).sum(0, keepdims=True))
        self.dx, self.dy, self.dz = v
        self.lam = np.zeros(S, f32)
        mask = (rng.random(S) < in_frac) * rng.standard_normal(S)
        self.win = (gamma * mask).astype(f32)
        self.eps = np.broadcast_to(np.asarray(eps, f32), (B,)).reshape((B,) + (1,) * N).copy()
        self.t = 0.0

    def conv(self, A, K):
        return np.fft.irfftn(np.fft.rfftn(A, axes=self.axes) * K, s=self.shape, axes=self.axes).astype(np.float32)

    def step(self, u, dt):
        P = self.P; a, b = P["a"], P["b"]
        x, y, z, c = self.x, self.y, self.z, self.c
        ex, ey = self.conv(x, self.K) - x, self.conv(y, self.K) - y
        fx = self.eps * ex + self.win * u; fy = self.eps * ey
        rhs = lambda x, y, z: (-y - z + fx, x + a * y + fy, b + z * (x - c))
        k1 = rhs(x, y, z); k2 = rhs(x + .5*dt*k1[0], y + .5*dt*k1[1], z + .5*dt*k1[2])
        k3 = rhs(x + .5*dt*k2[0], y + .5*dt*k2[1], z + .5*dt*k2[2]); k4 = rhs(x + dt*k3[0], y + dt*k3[1], z + dt*k3[2])
        dx, dy, dz = self.dx, self.dy, self.dz
        jv = lambda dx, dy, dz, x, z: (-dy - dz, dx + a * dy, z * dx + (x - c) * dz)
        j1 = jv(dx, dy, dz, x, z)
        j2 = jv(dx + .5*dt*j1[0], dy + .5*dt*j1[1], dz + .5*dt*j1[2], x + .5*dt*k1[0], z + .5*dt*k1[2])
        dx, dy, dz = dx + dt*j2[0], dy + dt*j2[1], dz + dt*j2[2]
        g = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-30
        self.dx, self.dy, self.dz = dx / g, dy / g, dz / g
        self.lam += (np.log(g) / dt - self.lam) * (dt / P["tau_lam"])
        self.x = x + dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]); self.y = y + dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
        self.z = np.clip(z + dt/6*(k1[2]+2*k2[2]+2*k3[2]+k4[2]), -50, 80)
        if self.t >= P["warmup"] and self.adapt.any():
            Sx = self.conv(np.exp(-(ex*ex + ey*ey) / P["sigA"]**2).astype(np.float32), self.K)
            dc = P["eta"] * (-np.tanh((self.lam - P["lam_target"]) / P["lam_scale"]) + P["kappa"] * growth(Sx, P["mu"], P["sigma"])) \
                 + P["Dc"] * (self.conv(c, self.Kc) - c)
            cn = c + dt * dc
            cn = np.where(cn < P["cmin"], 2*P["cmin"] - cn, cn); cn = np.where(cn > P["cmax"], 2*P["cmax"] - cn, cn)
            self.c = np.where(self.adapt, np.clip(cn, P["cmin"], P["cmax"]), c).astype(np.float32)
        self.t += dt

FL_DEFAULT = dict(a=0.2, b=0.2, Rc=3.0, tau_lam=30.0, warmup=40.0, sigA=2.0, eta=0.1, lam_target=0.02,
                  lam_scale=0.05, kappa=0.7, mu=0.5, sigma=0.2, Dc=0.02, cmin=2.0, cmax=7.0)

CONDS = {  # name: (c0 per cell, adapt during washout, adapt after washout)
    "FL-adapt":         ("rand", True, True),
    "FL-adapt→freeze":  ("rand", True, False),
    "FL-c2.5 (порядок)": (2.5, False, False),
    "FL-c2.8 (бифурк.)": (2.8, False, False),
    "FL-c5.0 (хаос)":    (5.0, False, False),
}

def run_fl(args):
    task = args.task
    n_total = args.washout + args.ntr + args.nva + args.nte + 60
    u = make_input(task, n_total, seed=0)
    tr, va, te = split(n_total, args.washout, args.ntr, args.nva, args.nte)
    if task == "mc": pass
    names = [c for c in CONDS if (not args.conds or c.split()[0] in args.conds)]
    B = len(names) * len(args.seeds)
    rng0 = np.random.default_rng(123)
    c0 = []; ad_w = []; ad_a = []
    for nm in names:
        for s in args.seeds:
            cc, aw, aa = CONDS[nm]
            c0.append(5.0 if cc == "rand" else cc); ad_w.append(aw); ad_a.append(aa)
    P = dict(FL_DEFAULT); P["a"] = args.a
    fl = BatchFL(B, args.N, args.L, args.R, args.eps, args.gamma, c0, ad_w, args.in_frac, seed=args.seeds[0] * 7 + 1, P=P)
    # для адаптивных режимов — гладкий случайный стартовый геном как в основной модели
    rr = np.random.default_rng(5)
    noise = rr.standard_normal(fl.c.shape).astype(np.float32)
    sm = fl.conv(fl.conv(noise, fl.K), fl.K); sm = (sm - sm.mean()) / sm.std()
    randc = np.clip(5.0 + 1.2 * sm, 2, 7)
    adaptive = np.array([CONDS[nm][0] == "rand" for nm in names for _ in args.seeds]).reshape((B,) + (1,) * args.N)
    fl.c = np.where(adaptive, randc, fl.c).astype(np.float32)
    # подмножество клеток для ридаута (своё для каждого элемента батча)
    ncell = int(np.prod(fl.shape))
    sel = np.stack([np.random.default_rng(100 + i).choice(ncell, args.nread, replace=False) for i in range(B)])
    bidx = np.arange(B)[:, None]
    feats = np.zeros((B, n_total, 3 * args.nread), np.float32)
    diag = {"lam": np.zeros((B, n_total), np.float32), "c": np.zeros((B, n_total), np.float32),
            "chaos": np.zeros((B, n_total), np.float32)}
    t0 = time.time()
    # предварительный прогрев без входа (переходный процесс синхронизации), одинаковый для всех режимов
    for _ in range(int(args.prewarm / args.dt)): fl.step(0.0, args.dt)
    fl.t = 0.0
    for k in range(n_total):
        if k == args.washout:
            fl.adapt = np.array(ad_a, bool).reshape(fl.adapt.shape)
        acc = 0
        for j in range(args.spp):
            fl.step(u[k], args.dt)
            if args.avg:
                sx = fl.x.reshape(B, -1)[bidx, sel]; sy = fl.y.reshape(B, -1)[bidx, sel]; sz = fl.z.reshape(B, -1)[bidx, sel]
                acc = acc + np.concatenate([sx, sy, sz], 1)
        if args.avg: feats[:, k] = acc / args.spp
        else:
            feats[:, k] = np.concatenate([fl.x.reshape(B, -1)[bidx, sel], fl.y.reshape(B, -1)[bidx, sel],
                                          fl.z.reshape(B, -1)[bidx, sel]], 1)
        diag["lam"][:, k] = fl.lam.reshape(B, -1).mean(1); diag["c"][:, k] = fl.c.reshape(B, -1).mean(1)
        diag["chaos"][:, k] = (fl.lam.reshape(B, -1) > 0.01).mean(1)
        if k % 500 == 0: print(f"[{args.tag}/{task}] sample {k}/{n_total}  {time.time()-t0:.0f}s", flush=True)
        if not np.isfinite(fl.x).all(): print("NaN!"); break
    Y, tnames = targets(task, u, np.arange(n_total - 60) if task != "mc" else np.arange(n_total))
    res = {"task": task, "tag": args.tag, "targets": tnames, "args": vars(args), "runs": []}
    i = 0
    for nm in names:
        for s in args.seeds:
            X = feats[i]
            if task != "mc": Xu, Yu = X[: n_total - 60], Y
            else: Xu, Yu = X, Y
            r = ridge_eval(Xu.astype(np.float64), Yu, tr, va, te, task)
            r.update(cond=nm, seed=s, lam_mean=float(diag["lam"][i, args.washout:].mean()),
                     c_mean=float(diag["c"][i, args.washout:].mean()), chaos=float(diag["chaos"][i, args.washout:].mean()),
                     c_drift=float(diag["c"][i, -1] - diag["c"][i, args.washout]))
            res["runs"].append(r); i += 1
            print(f"  {nm:22s} seed {s}: NRMSE/MC={r.get('MC', r['nrmse'])}  λ={r['lam_mean']:+.3f} c={r['c_mean']:.2f} chaos={r['chaos']:.2f}", flush=True)
    os.makedirs(RES, exist_ok=True)
    json.dump(res, open(os.path.join(RES, f"{args.tag}_{task}.json"), "w"))
    print(f"done {time.time()-t0:.0f}s")

# ============================================================================ ESN + linear baselines
def run_esn(args):
    task = args.task
    n_total = args.washout + args.ntr + args.nva + args.nte + 60
    u = make_input(task, n_total, seed=0)
    tr, va, te = split(n_total, args.washout, args.ntr, args.nva, args.nte)
    Y, tnames = targets(task, u, np.arange(n_total - 60) if task != "mc" else np.arange(n_total))
    nrows = n_total - 60 if task != "mc" else n_total
    Nn = 3 * args.nread
    res = {"task": task, "tag": args.tag, "targets": tnames, "runs": []}
    for s in args.seeds:
        rng = np.random.default_rng(s)
        W0 = rng.standard_normal((Nn, Nn)) * (rng.random((Nn, Nn)) < 10 / Nn)
        W0 /= np.max(np.abs(np.linalg.eigvals(W0)))
        win = rng.uniform(-1, 1, Nn); bias = rng.uniform(-0.2, 0.2, Nn)
        cands = []
        for rho in [0.5, 0.8, 0.95, 1.1, 1.3]:
            for sc in [0.1, 0.5, 1.0]:
                for leak in [0.3, 1.0]:
                    W = rho * W0; h = np.zeros(Nn); X = np.zeros((nrows, Nn))
                    for k in range(nrows):
                        h = (1 - leak) * h + leak * np.tanh(W @ h + sc * win * u[k] + bias); X[k] = h
                    cands.append((ridge_eval(X, Y[:nrows], tr, va, te, task), dict(rho=rho, scale=sc, leak=leak)))
        # гиперпараметры ESN выбираются по val отдельно для каждой цели (горизонта / задержки)
        V = np.array([c[0]["val"] for c in cands]); bi = V.argmin(0)
        if task == "mc":   # для MC — один резервуар на все задержки: max суммарной val-ёмкости
            bi = np.full(V.shape[1], np.argmax(np.clip(1 - V ** 2, 0, 1).sum(1)))
        nr = [cands[b][0]["nrmse"][j] for j, b in enumerate(bi)]
        r = {"nrmse": nr, "hp": [cands[b][1] for b in bi], "cond": "ESN", "seed": s}
        if task == "mc":
            r["mc_curve"] = [cands[b][0]["mc_curve"][j] for j, b in enumerate(bi)]; r["MC"] = float(np.sum(r["mc_curve"]))
        r["pred_test"] = np.stack([np.array(cands[b][0]["pred_test"])[:, j] for j, b in enumerate(bi)], 1).tolist()
        r["true_test"] = cands[0][0]["true_test"]
        res["runs"].append(r)
        print(f"  ESN seed {s}: {r.get('MC', r['nrmse'])}", flush=True)
    # линейный бейзлайн: ридж на окне последних D входов (без резервуара)
    D = 3 * args.nread if task == "mc" else 50
    D = min(D, 200)
    X = np.zeros((nrows, D))
    for d in range(D): X[d:, d] = u[: nrows - d]
    r = ridge_eval(X, Y[:nrows], tr, va, te, task); r.update(cond=f"Linear-delay({D})", seed=0)
    res["runs"].append(r); print(f"  Linear({D}): {r.get('MC', r['nrmse'])}")
    json.dump(res, open(os.path.join(RES, f"{args.tag}_{task}_esn.json"), "w"))

def val_score(X, Y, tr, va, task):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Z = np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], 1); ym = Y[tr].mean(0)
    U, s, Vt = np.linalg.svd(Z[tr], full_matrices=False); UtY = U.T @ (Y[tr] - ym)
    best = np.full(Y.shape[1], np.inf)
    for a in ALPHAS:
        W = Vt.T @ ((s / (s ** 2 + a))[:, None] * UtY)
        e = np.sqrt(((Z[va] @ W + ym - Y[va]) ** 2).mean(0)) / Y[va].std(0)
        best = np.minimum(best, e)
    return float(np.mean(-np.clip(1 - best ** 2, 0, 1)) if task == "mc" else np.mean(np.log(best)))

# ============================================================================ main
def common(ap):
    A = ap.add_argument
    A("--task", choices=list(TASKS), default="mg"); A("--tag", default="main")
    A("--seeds", type=int, nargs="+", default=[0, 1, 2])
    A("--washout", type=int, default=400); A("--ntr", type=int, default=1400)
    A("--nva", type=int, default=400); A("--nte", type=int, default=400)
    A("--nread", type=int, default=400, help="клеток в ридауте (признаков = 3*nread)")

if __name__ == "__main__":
    pass
    cmd = sys.argv[1]; argv = sys.argv[2:]
    ap = argparse.ArgumentParser()
    if cmd in ("run", "esn"):
        common(ap)
        A = ap.add_argument
        A("--N", type=int, default=4); A("--L", type=int, default=8); A("--R", type=float, default=3.0)
        A("--eps", type=float, default=0.1); A("--gamma", type=float, default=1.0, help="масштаб входа")
        A("--in_frac", type=float, default=0.3); A("--dt", type=float, default=0.05)
        A("--spp", type=int, default=10, help="шагов интегрирования на один отсчёт входа")
        A("--avg", action="store_true", help="признаки = среднее по интервалу удержания входа")
        A("--a", type=float, default=0.2, help="параметр Рёсслера a (<0 -> устойчивый фокус, затухающая клетка)")
        A("--prewarm", type=float, default=60.0); A("--conds", nargs="*", default=[])
        args = ap.parse_args(argv)
        (run_fl if cmd == "run" else run_esn)(args)
    elif cmd == "consistency": pass
    elif cmd == "report":
        from reservoir_report import report; report(argv)

# ============================================================================ consistency (echo-state property) test
def consistency(argv):
    """Две копии резервуара с одинаковой маской входа и разными начальными условиями получают один и тот же вход.
    Если состояния сходятся (корреляция → 1), есть echo-state property / generalized synchronization с входом."""
    ap = argparse.ArgumentParser(); A = ap.add_argument
    A("--gamma", type=float, default=2.0); A("--a", type=float, default=0.2); A("--n", type=int, default=800)
    A("--spp", type=int, default=10); A("--out", default="consistency")
    args = ap.parse_args(argv)
    conds = [("c2.5", 2.5, False), ("c2.8", 2.8, False), ("c5.0", 5.0, False), ("adapt", 5.0, True)]
    B = 2 * len(conds); P = dict(FL_DEFAULT); P["a"] = args.a
    c0 = [c for _, c, _ in conds for _ in range(2)]; ad = [a for _, _, a in conds for _ in range(2)]
    fl = BatchFL(B, 4, 8, 3.0, 0.1, args.gamma, c0, ad, 1.0, seed=3, P=P)
    fl.win = np.broadcast_to(fl.win[:1], fl.win.shape).copy()          # одинаковая маска во всех копиях
    u = make_input("mg", args.n)
    corr = np.zeros((len(conds), args.n))
    for k in range(args.n):
        for _ in range(args.spp): fl.step(u[k], 0.05)
        X = fl.x.reshape(B, -1)
        for i in range(len(conds)):
            corr[i, k] = np.corrcoef(X[2 * i], X[2 * i + 1])[0, 1]
    res = {nm: float(np.nanmean(corr[i, -200:])) for i, (nm, _, _) in enumerate(conds)}
    print(args.out, res)
    os.makedirs(RES, exist_ok=True)
    json.dump({"final_corr": res, "curve": corr.tolist(), "args": vars(args)}, open(os.path.join(RES, f"cons_{args.out}.json"), "w"))

if __name__ == "__main__" and sys.argv[1] == "consistency":
    consistency(sys.argv[2:])
