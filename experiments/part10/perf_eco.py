import sys; from pathlib import Path; _root = next(p for p in [Path(__file__).resolve().parent] + list(Path(__file__).resolve().parents) if (p / "pyproject.toml").exists()); sys.path.insert(0, str(_root / "src"))
"""Часть 10: профилирование шага eco2 (тайминги по компонентам, steps/s без компиляции, загрузка GPU, cost_analysis)."""
import time, subprocess, threading, numpy as np

def _t(f, *a, reps=5):
    import jax
    jax.block_until_ready(f(*a))                      # компиляция + прогрев
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); jax.block_until_ready(f(*a)); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))

def _gpu_sampler(stop, out):
    while not stop.is_set():
        try:
            s = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5).stdout.strip().split(",")
            out.append([float(x) for x in s])
        except Exception: pass
        time.sleep(0.25)

def make(L, N, B, c, eco, seed=0, nblob=0):
    import jax.numpy as jnp, fl_glider as FG, fl_eco2 as E
    rng = np.random.default_rng(seed)
    A = np.zeros((L,) * N, np.float32)
    r = np.arange(-10, 11); pg = np.meshgrid(*([r] * N), indexing="ij"); d2 = sum(q.astype(np.float32) ** 2 for q in pg)
    patch = (np.exp(-d2 / 18.0) * (d2 < 100)).astype(np.float32)
    for _ in range(nblob):                                   # сгустки для теста логирования
        c0 = rng.integers(0, L, N); A[np.ix_(*[(c0[k] + r) % L for k in range(N)])] += patch
    if not nblob:
        A = (rng.random((L,) * N) < 0.05).astype(np.float32) * 0.8
    st = dict(A=A, x=rng.standard_normal(A.shape).astype(np.float32), y=rng.standard_normal(A.shape).astype(np.float32),
              z=np.zeros_like(A), c=np.full_like(A, 2.5), F=np.full_like(A, eco["F0"]))
    st["Q"] = np.stack([A * 0.3, A * -0.1]); 
    import jax
    st["key"] = np.asarray(jax.random.PRNGKey(seed))
    cc = dict(c, **eco)
    P = jax.tree.map(lambda *a: jnp.stack(a), *[E.params(cc)] * B)
    s = jax.tree.map(lambda *a: jnp.stack([jnp.asarray(x) for x in a]), *[st] * B)
    KA = jnp.stack([jnp.asarray(FG.ring_kernel_np(L, N, c["R"], tuple(c["peaks"])))] * B); Ko = jnp.asarray(FG.ring_kernel_np(L, N, 3.0))
    KX = E.kernels(L, N)
    return s, P, KA, Ko, KX

def run_perf(c, eco, L=96, N=4, n=50, batches=(1,), fast=True, log_test=True, nblob=400, check_steps=300):
    import jax, jax.numpy as jnp, fl_glider as FG, fl_eco2 as E, glider_eco as GE
    res = dict(L=L, N=N, gpu=str(jax.devices()[0].device_kind), jax=jax.__version__)
    s, P, KA, Ko, KX = make(L, N, 1, c, eco)
    sh = (L,) * N; x = s["A"][0]; Kk = KX["blur"]
    # --- компоненты (одна симуляция) ---
    fft = jax.jit(lambda G: jnp.fft.irfftn(jnp.fft.rfftn(G) * Kk, s=sh))
    res["t_conv"] = _t(fft, x)
    res["t_rfft_only"] = _t(jax.jit(lambda G: jnp.fft.rfftn(G)), x)
    v = [s["x"][0] * 0.1 for _ in range(N)]
    res["t_muscl"] = _t(jax.jit(lambda A, *v: FG._advect_muscl(A, list(v), 1.0)), x, *v)
    res["t_grad_all"] = _t(jax.jit(lambda A: sum(FG._grad(A, k) for k in range(N))), x)
    res["t_rng_normal"] = _t(jax.jit(lambda k: jax.random.normal(k, sh)), jax.random.PRNGKey(1))
    res["t_elementwise_exp"] = _t(jax.jit(lambda A: jnp.exp(-(A - .2) ** 2) * 2 - 1), x)
    # --- шаг целиком ---
    one = jax.jit(lambda st: E.step(st, jax.tree.map(lambda a: a[0], P), KA[0], Ko, KX, True))
    s0 = jax.tree.map(lambda a: a[0], s)
    try:
        ca = one.lower(s0).compile().cost_analysis(); ca = ca[0] if isinstance(ca, list) else ca
        res["cost"] = {k: float(ca[k]) for k in ("flops", "bytes accessed") if k in ca}
    except Exception as e: res["cost_err"] = repr(e)[:200]
    res["t_step_jit1"] = _t(one, s0)
    for B in batches:
        sB = s if B == 1 else make(L, N, B, c, eco)
        sB, PB, KAB, KoB, KXB = sB if B != 1 else (s, P, KA, Ko, KX)
        try:
            f = lambda st: E.run_chunk(st, PB, KAB, KoB, KXB, n, True)
            t0 = time.perf_counter(); out = jax.block_until_ready(f(sB)); tc = time.perf_counter() - t0
            stop = threading.Event(); smp = []; th = threading.Thread(target=_gpu_sampler, args=(stop, smp)); th.start()
            ts = []
            for _ in range(3):
                t0 = time.perf_counter(); out = jax.block_until_ready(f(out)); ts.append(time.perf_counter() - t0)
            stop.set(); th.join()
            t = float(np.median(ts)); res[f"B{B}"] = dict(t_first=tc, t_chunk=t, steps_s=n / t, sim_steps_s=B * n / t,
                                                        gpu_util=np.mean([q[0] for q in smp]) if smp else None,
                                                        mem_util=np.mean([q[1] for q in smp]) if smp else None,
                                                        mem_MiB=max([q[2] for q in smp]) if smp else None, power_W=np.mean([q[3] for q in smp]) if smp else None)
            if fast and B == 1 and hasattr(E, "run_chunk_fast"):
                g = lambda st: E.run_chunk_fast(st, PB, KAB, KoB, KXB, n, True)
                jax.block_until_ready(g(sB)); ts = []
                for _ in range(3):
                    t0 = time.perf_counter(); jax.block_until_ready(g(sB)); ts.append(time.perf_counter() - t0)
                res["B1_fast"] = dict(t_chunk=float(np.median(ts)), steps_s=n / float(np.median(ts)))
            del sB, out
        except Exception as e:
            res[f"B{B}"] = dict(err=repr(e)[:300])
    # --- проверка fast == исходный ---
    if fast and hasattr(E, "run_chunk_fast"):
        a = E.run_chunk(s, P, KA, Ko, KX, check_steps, True); b = E.run_chunk_fast(s, P, KA, Ko, KX, check_steps, True)
        res["fast_check"] = dict(maxdiff_A=float(jnp.abs(a["A"] - b["A"]).max()), maxdiff_Q=float(jnp.abs(a["Q"] - b["Q"]).max()),
                                 mass_ref=float(a["A"].sum()), mass_fast=float(b["A"].sum()), bitwise=bool(jnp.all(a["A"] == b["A"])))
    # --- логирование: пулинг + перенос на хост + поиск объектов на CPU ---
    if log_test:
        sl, *_ = make(L, N, 1, c, eco, nblob=nblob)
        t0 = time.perf_counter(); Ap = np.asarray(GE._pool(sl["A"], N, L)); Qp = np.asarray(GE._pool(sl["Q"], N, L)); t1 = time.perf_counter()
        Ap = np.asarray(GE._pool(sl["A"], N, L)); Qp = np.asarray(GE._pool(sl["Q"], N, L)); t2 = time.perf_counter()
        objs = GE._objects(Ap[0], Qp[0], 80.0, 2, N); t3 = time.perf_counter()
        pj = np.asarray(sl["A"][0]).mean(axis=tuple(range(2, N))); t4 = time.perf_counter()
        res["log"] = dict(t_pool_transfer=t2 - t1, t_objects_cpu=t3 - t2, n_obj=len(objs), t_proj_full_transfer=t4 - t3)
    return res
