import _path  # noqa: F401
"""Бенчмарк шагов/с: JAX (CPU/GPU) и numpy. Модель = основная Feigenbaum-Lenia (правило own, без входа)."""
import time, json, sys
import numpy as np

CASES = [  # (N, L, batch, rule)
    (4, 24, 1, "own"), (4, 48, 1, "own"), (4, 64, 1, "own"), (5, 16, 1, "own"), (5, 24, 1, "own"),
    (4, 24, 1, "cond"), (4, 64, 1, "cond"),
    (4, 8, 512, "cond"), (4, 16, 64, "cond"), (4, 24, 32, "own"),
]

def bench_jax(cases=CASES, target_sec=3.0, max_cells=None):
    import jax, jax.numpy as jnp, fl_jax as F
    dev = str(jax.devices()[0]); out = []
    for (N, L, B, rule) in cases:
        cells = B * L ** N
        if max_cells and cells > max_cells: continue
        shape = (L,) * N
        K = jnp.asarray(F.lenia_kernel_np(L, N, 4.0)); Kc = jnp.asarray(F.lenia_kernel_np(L, N, 3.0))
        Ks = jnp.asarray(F.lenia_kernel_np(L, N, 4.0))
        keys = jax.random.split(jax.random.PRNGKey(0), B)
        st = jax.vmap(lambda k: F.init_state(k, shape, K_smooth=Ks))(keys)
        P = F.stack_params([F.make_params(warmup=0.0, lam_target=(0.02 if rule == "own" else -0.02))] * B)
        sync = rule == "own"
        try:
            s = F.run_free(st, P, K, Kc, 10, rule=rule, sync=sync); jax.block_until_ready(s)   # compile
            t = time.time(); s = F.run_free(st, P, K, Kc, 10, rule=rule, sync=sync); jax.block_until_ready(s)
            dt10 = time.time() - t
            n = int(max(10, min(5000, 10 * target_sec / max(dt10, 1e-4))))
            t = time.time(); s = F.run_free(st, P, K, Kc, n, rule=rule, sync=sync); jax.block_until_ready(s)
            el = time.time() - t
            r = dict(N=N, L=L, B=B, rule=rule, cells=cells, steps=n, steps_per_s=n / el, cell_steps_per_s=n * cells / el,
                     finite=bool(np.isfinite(np.asarray(s["x"])).all()), device=dev)
        except Exception as ex:
            r = dict(N=N, L=L, B=B, rule=rule, cells=cells, error=str(ex)[:300], device=dev)
        print(json.dumps(r), flush=True); out.append(r)
    return out

def bench_numpy(cases=((4, 24), (4, 48), (4, 64), (5, 16), (5, 24)), nsteps=None):
    import types
    from feigenbaum_lenia import FeigenbaumLenia
    out = []
    for N, L in cases:
        p = types.SimpleNamespace(N=N, L=L, R=4.0, peaks=[1.0], Rc=3.0, eps=0.1, dt=0.05, a=0.2, b=0.2, c_init=5.0,
                                  c_init_spread=1.2, cmin=2.0, cmax=7.0, eta=0.1, lam_target=0.02, lam_scale=0.05,
                                  tau_lam=30.0, kappa=0.7, mu=0.5, sigma=0.2, growth="bell", warmup=0.0, sync="amp",
                                  sigA=2.0, Dc=0.02, seed=1)
        sim = FeigenbaumLenia(p); sim.step()
        n = nsteps or max(2, int(2e7 / L ** N))
        t = time.time()
        for _ in range(n): sim.step()
        el = time.time() - t
        r = dict(N=N, L=L, B=1, rule="own", cells=L ** N, steps=n, steps_per_s=n / el, cell_steps_per_s=n * L ** N / el, device="numpy-cpu")
        print(json.dumps(r), flush=True); out.append(r)
    return out

if __name__ == "__main__":
    which = sys.argv[1]
    res = bench_numpy() if which == "numpy" else bench_jax(max_cells=int(sys.argv[2]) if len(sys.argv) > 2 else None)
    json.dump(res, open(f"results/bench_{which}_local.json", "w"), indent=1)
