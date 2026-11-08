#!/usr/bin/env python3
"""
Feigenbaum-Lenia: N-мерный непрерывный клеточный автомат, в котором каждая клетка —
осциллятор Рёсслера, связанный с соседями гладким ядром Lenia (FFT-свёртка в N измерениях),
а "геном" клетки — бифуркационный параметр c_i — медленно эволюционирует под действием
локального показателя Ляпунова (гомеостаз к краю хаоса) и локальной фазовой когерентности
(Lenia-подобная функция роста). См. README.md.
"""
import argparse, json, os, time
import numpy as np

# ----------------------------------------------------------------------------- kernel
def lenia_kernel(L, N, R, peaks=(1.0,)):
    """N-мерное радиальное ядро Lenia (кольцевое, экспоненциальное ядро), нормированное на 1."""
    ax = np.fft.fftfreq(L) * L                      # периодические координаты 0..L/2,-L/2..-1
    grids = np.meshgrid(*([ax] * N), indexing="ij", sparse=True)
    d = np.sqrt(sum(g.astype(np.float64) ** 2 for g in grids)) / R
    K = np.zeros_like(d)
    B = len(peaks)
    for i, h in enumerate(peaks):                    # многокольцевой вариант как в Lenia
        r = d * B - i
        m = (r > 0) & (r < 1) & (d < 1)
        K[m] += h * np.exp(4.0 - 1.0 / (r[m] * (1.0 - r[m])))
    K /= K.sum()
    return np.fft.rfftn(K.astype(np.float32))

class Conv:
    def __init__(self, Khat, shape):
        self.Khat, self.shape = Khat, shape
    def __call__(self, A):
        return np.fft.irfftn(np.fft.rfftn(A) * self.Khat, s=self.shape, axes=tuple(range(len(self.shape)))).astype(np.float32)

# ----------------------------------------------------------------------------- model
def rossler_rhs(x, y, z, c, a, b, fx, fy):
    return (-y - z + fx), (x + a * y + fy), (b + z * (x - c))

def growth(u, mu, sigma):                            # функция роста Lenia: [-1, 1]
    return 2.0 * np.exp(-((u - mu) ** 2) / (2 * sigma ** 2)) - 1.0

class FeigenbaumLenia:
    def __init__(self, p):
        self.p = p
        rng = np.random.default_rng(p.seed)
        self.shape = (p.L,) * p.N
        self.conv = Conv(lenia_kernel(p.L, p.N, p.R, tuple(p.peaks)), self.shape)
        self.convc = Conv(lenia_kernel(p.L, p.N, p.Rc), self.shape)
        f32 = np.float32
        # случайные фазы на "кольце" аттрактора Рёсслера
        th = rng.uniform(0, 2 * np.pi, self.shape)
        rad = rng.uniform(2, 6, self.shape)
        self.x = (rad * np.cos(th)).astype(f32)
        self.y = (rad * np.sin(th)).astype(f32)
        self.z = rng.uniform(0, 0.5, self.shape).astype(f32)
        # гладкий случайный "геном" c_i
        noise = rng.standard_normal(self.shape).astype(f32)
        sm = Conv(lenia_kernel(p.L, p.N, max(2.0, p.R)), self.shape)
        n = sm(sm(noise)); n = (n - n.mean()) / (n.std() + 1e-9)
        self.c = np.clip(p.c_init + p.c_init_spread * n, p.cmin, p.cmax).astype(f32)
        # касательные векторы для локального FTLE
        v = rng.standard_normal((3,) + self.shape).astype(f32)
        v /= np.sqrt((v ** 2).sum(0, keepdims=True))
        self.dx, self.dy, self.dz = v
        self.lam = np.zeros(self.shape, f32)         # EMA локального показателя Ляпунова
        self.R = np.zeros(self.shape, f32)           # локальная фазовая когерентность
        self.S = np.zeros(self.shape, f32)           # локальная амплитудная синхронизация
        self.t = 0.0

    def phase(self):
        return np.arctan2(self.y, self.x)

    def step(self):
        p = self.p; dt = p.dt
        x, y, z, c = self.x, self.y, self.z, self.c
        # 1) нелокальная связь через N-мерное ядро (операторное расщепление: поле соседей фиксировано на шаге)
        ex, ey = self.conv(x) - x, self.conv(y) - y
        fx, fy = p.eps * ex, p.eps * ey
        # 2) RK4 для локального аттрактора
        k1 = rossler_rhs(x, y, z, c, p.a, p.b, fx, fy)
        k2 = rossler_rhs(x + .5*dt*k1[0], y + .5*dt*k1[1], z + .5*dt*k1[2], c, p.a, p.b, fx, fy)
        k3 = rossler_rhs(x + .5*dt*k2[0], y + .5*dt*k2[1], z + .5*dt*k2[2], c, p.a, p.b, fx, fy)
        k4 = rossler_rhs(x + dt*k3[0], y + dt*k3[1], z + dt*k3[2], c, p.a, p.b, fx, fy)
        # 3) касательная динамика (якобиан собственного потока клетки, RK2) -> локальный FTLE
        dx, dy, dz = self.dx, self.dy, self.dz
        def jv(dx, dy, dz, x, z):
            return (-dy - dz), (dx + p.a * dy), (z * dx + (x - c) * dz)
        j1 = jv(dx, dy, dz, x, z)
        xm, zm = x + .5*dt*k1[0], z + .5*dt*k1[2]
        j2 = jv(dx + .5*dt*j1[0], dy + .5*dt*j1[1], dz + .5*dt*j1[2], xm, zm)
        dx, dy, dz = dx + dt*j2[0], dy + dt*j2[1], dz + dt*j2[2]
        g = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-30
        inst = np.log(g) / dt
        self.dx, self.dy, self.dz = dx / g, dy / g, dz / g
        self.lam += (inst - self.lam) * (dt / p.tau_lam)
        self.x = x + dt/6 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        self.y = y + dt/6 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        self.z = z + dt/6 * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        np.clip(self.z, -50, 80, out=self.z)
        # 4) локальный параметр порядка Курамото через то же ядро
        ph = np.arctan2(self.y, self.x)
        cr, sr = self.conv(np.cos(ph)), self.conv(np.sin(ph))
        self.R = np.sqrt(cr*cr + sr*sr)
        # локальная амплитудная (полная) синхронизация: 1 = клетка совпадает с полем соседей
        self.S = self.conv(np.exp(-(ex*ex + ey*ey) / p.sigA**2).astype(np.float32))
        # 5) эволюция "генома" c: гомеостаз к краю хаоса + Lenia-рост по когерентности + диффузия
        if self.t >= p.warmup:
            Q = self.R if p.sync == "phase" else self.S
            if p.growth == "bell":   # Lenia-колокол: «правильная» когерентность -> к хаосу, иначе -> к порядку
                gR = growth(Q, p.mu, p.sigma)
            else:                    # анти-синхронизация: слишком синхронно -> возбуждаем хаос
                gR = np.tanh((Q - p.mu) / p.sigma)
            dc = p.eta * (-np.tanh((self.lam - p.lam_target) / p.lam_scale) + p.kappa * gR) \
                 + p.Dc * (self.convc(c) - c)
            cn = c + dt * dc
            cn = np.where(cn < p.cmin, 2 * p.cmin - cn, cn); cn = np.where(cn > p.cmax, 2 * p.cmax - cn, cn)
            self.c = np.clip(cn, p.cmin, p.cmax).astype(np.float32)
        self.t += dt

# ----------------------------------------------------------------------------- diagnostics
def slice_grid(A, n=4):
    """Для N>=4: сетка n x n 2D-срезов (x1,x2) при разных (x3,x4); для N=3: n срезов по x3; N=2: само поле."""
    N = A.ndim; L = A.shape[0]
    if N == 2:
        return A
    idx = [int(round(i * L / n)) for i in range(n)]
    if N == 3:
        return np.concatenate([np.pad(A[:, :, k], 1, constant_values=np.nan) for k in idx], axis=1)
    rest = (0,) * (N - 4)
    rows = []
    for k4 in idx:
        rows.append(np.concatenate([np.pad(A[(slice(None), slice(None), k3, k4) + rest], 1,
                                           constant_values=np.nan) for k3 in idx], axis=1))
    return np.concatenate(rows, axis=0)

def participation_dim(S):
    """PCA participation ratio D = (sum λ)^2 / sum λ^2 по снимкам S (T x D)."""
    S = S - S.mean(0, keepdims=True)
    G = S @ S.T / S.shape[1]
    ev = np.clip(np.linalg.eigvalsh(G), 0, None)
    return float(ev.sum() ** 2 / (np.sum(ev ** 2) + 1e-30)), ev[::-1]

def correlation_dimension(S, n_pc=12):
    """Грассбергер–Прокачча на траектории глобального поля (в пространстве главных компонент)."""
    S = S - S.mean(0, keepdims=True)
    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    Y = U[:, :n_pc] * s[:n_pc]
    D = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1))
    d = D[np.triu_indices(len(Y), k=3)]               # исключаем ближайших во времени соседей
    d = d[d > 0]
    rs = np.logspace(np.log10(np.percentile(d, 1)), np.log10(np.percentile(d, 50)), 20)
    C = np.array([(d < r).mean() for r in rs])
    m = C > 0
    slope = np.polyfit(np.log(rs[m]), np.log(C[m]), 1)[0]
    return float(slope), rs, C

def rossler_bifurcation(a, b, cs, T=600.0, dt=0.01, burn=300.0):
    """Бифуркационная диаграмма одиночного Рёсслера: локальные максимумы x при каждом c."""
    x = np.full_like(cs, 1.0); y = np.zeros_like(cs); z = np.zeros_like(cs)
    pts_c, pts_x = [], []
    prev2 = prev1 = None
    n = int(T / dt); nb = int(burn / dt)
    for i in range(n):
        f = lambda x, y, z: (-y - z, x + a*y, b + z*(x - cs))
        k1 = f(x, y, z); k2 = f(x+.5*dt*k1[0], y+.5*dt*k1[1], z+.5*dt*k1[2])
        k3 = f(x+.5*dt*k2[0], y+.5*dt*k2[1], z+.5*dt*k2[2]); k4 = f(x+dt*k3[0], y+dt*k3[1], z+dt*k3[2])
        x = x + dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]); y = y + dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
        z = z + dt/6*(k1[2]+2*k2[2]+2*k3[2]+k4[2])
        if i > nb and prev2 is not None:
            m = (prev1 > prev2) & (prev1 > x)
            pts_c.append(cs[m]); pts_x.append(prev1[m])
        prev2, prev1 = prev1, x.copy()
    return np.concatenate(pts_c), np.concatenate(pts_x)

def single_lyapunov(a, b, cs, T=400.0, dt=0.01, burn=100.0):
    """Наибольший показатель Ляпунова одиночного Рёсслера как функция c (касательный вектор)."""
    x = np.full_like(cs, 1.0); y = np.zeros_like(cs); z = np.zeros_like(cs)
    v = np.stack([np.ones_like(cs), np.zeros_like(cs), np.zeros_like(cs)])
    acc = np.zeros_like(cs); n = int(T/dt); nb = int(burn/dt)
    for i in range(n):
        dx, dy, dz = v
        v = np.stack([dx + dt*(-dy-dz), dy + dt*(dx + a*dy), dz + dt*(z*dx + (x-cs)*dz)])
        x, y, z = x + dt*(-y-z), y + dt*(x + a*y), z + dt*(b + z*(x-cs))
        g = np.sqrt((v**2).sum(0)); v /= g
        if i >= nb: acc += np.log(g)
    return acc / ((n - nb) * dt)

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Feigenbaum-Lenia: N-мерный КА из осцилляторов Рёсслера на краю хаоса")
    A = ap.add_argument
    A("--N", type=int, default=4, help="размерность решётки"); A("--L", type=int, default=24, help="сторона решётки")
    A("--R", type=float, default=4.0, help="радиус ядра связи"); A("--peaks", type=float, nargs="+", default=[1.0])
    A("--Rc", type=float, default=3.0, help="радиус ядра диффузии генома")
    A("--eps", type=float, default=0.1, help="сила связи"); A("--dt", type=float, default=0.05)
    A("--a", type=float, default=0.2); A("--b", type=float, default=0.2)
    A("--c_init", type=float, default=5.0); A("--c_init_spread", type=float, default=1.2)
    A("--cmin", type=float, default=2.0); A("--cmax", type=float, default=7.0)
    A("--eta", type=float, default=0.1, help="скорость эволюции генома c")
    A("--lam_target", type=float, default=0.02, help="целевой локальный показатель Ляпунова")
    A("--lam_scale", type=float, default=0.05); A("--tau_lam", type=float, default=30.0)
    A("--kappa", type=float, default=0.7, help="вес Lenia-роста по когерентности")
    A("--mu", type=float, default=0.5); A("--sigma", type=float, default=0.2)
    A("--growth", choices=["tanh", "bell"], default="bell"); A("--warmup", type=float, default=40.0)
    A("--sync", choices=["phase", "amp"], default="amp", help="какая синхронизация управляет геномом")
    A("--sigA", type=float, default=2.0, help="масштаб амплитудной синхронизации")
    A("--Dc", type=float, default=0.02, help="диффузия генома")
    A("--steps", type=int, default=7000); A("--every", type=int, default=14, help="шагов между кадрами")
    A("--seed", type=int, default=1); A("--out", default="out")
    A("--no-render", action="store_true")
    p = ap.parse_args()
    os.makedirs(p.out, exist_ok=True)
    json.dump(vars(p), open(os.path.join(p.out, "params.json"), "w"), indent=2)

    sim = FeigenbaumLenia(p)
    frames_phase, frames_c, frames_lam, snaps = [], [], [], []
    diag = {k: [] for k in ["t", "Rglob", "Rloc", "lam_mean", "frac_chaos", "c_mean", "c_std", "Sloc"]}
    chist = []; cbins = np.linspace(p.cmin, p.cmax, 81)
    ds = max(1, p.L // 12)
    t0 = time.time()
    for s in range(p.steps + 1):
        if s % p.every == 0:
            ph = sim.phase()
            frames_phase.append(slice_grid(ph).astype(np.float32))
            frames_c.append(slice_grid(sim.c).astype(np.float32))
            frames_lam.append(slice_grid(sim.lam).astype(np.float32))
            sl = (slice(None, None, ds),) * p.N
            snaps.append(np.concatenate([sim.x[sl].ravel(), sim.y[sl].ravel()]))
            diag["t"].append(sim.t)
            diag["Rglob"].append(float(np.abs(np.exp(1j * ph).mean())))
            diag["Rloc"].append(float(sim.R.mean()))
            diag["lam_mean"].append(float(sim.lam.mean()))
            diag["frac_chaos"].append(float((sim.lam > 0.01).mean()))
            diag["Sloc"].append(float(sim.S.mean()))
            diag["c_mean"].append(float(sim.c.mean())); diag["c_std"].append(float(sim.c.std()))
            chist.append(np.histogram(sim.c, bins=cbins)[0] / sim.c.size)
            if s % (p.every * 25) == 0:
                print(f"step {s:6d} t={sim.t:7.1f}  R_glob={diag['Rglob'][-1]:.3f} R_loc={diag['Rloc'][-1]:.3f} "
                      f"<λ>={diag['lam_mean'][-1]:+.3f} chaos={diag['frac_chaos'][-1]:.2f} S={diag['Sloc'][-1]:.2f} "
                      f"<c>={diag['c_mean'][-1]:.2f}±{diag['c_std'][-1]:.2f}  ({time.time()-t0:.0f}s)", flush=True)
            if not np.isfinite(sim.x).all():
                print("NaN — останов"); break
        sim.step()
    np.savez_compressed(os.path.join(p.out, "run.npz"), phase=np.array(frames_phase), c=np.array(frames_c),
                        lam=np.array(frames_lam), snaps=np.array(snaps, np.float32), chist=np.array(chist),
                        cbins=cbins, final_c=sim.c, final_lam=sim.lam, final_x=sim.x, final_y=sim.y,
                        **{k: np.array(v) for k, v in diag.items()})
    print(f"симуляция: {time.time()-t0:.0f} c")
    if not p.no_render:
        from render import render_all
        render_all(p.out)

if __name__ == "__main__":
    main()
