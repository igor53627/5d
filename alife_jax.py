"""Режим без входа (ALife-визуалы) с гомеостазом по условному λc: двойник решётки возмущён на δ0,
локальная скорость расхождения (с перенормировкой) управляет геномом. Считается на GPU через fl_jax."""
import time, numpy as np

def run_alife(N=4, L=48, steps=8000, every=40, lam_target=-0.005, kappa=0.0, a=0.2, eps=0.1, seed=1, R=4.0,
              lam_scale=0.01, eta=0.1, c_init=5.0):
    import jax, jax.numpy as jnp, fl_jax as F
    from feigenbaum_lenia import slice_grid
    shape = (L,) * N
    K = jnp.asarray(F.lenia_kernel_np(L, N, R)); Kc = jnp.asarray(F.lenia_kernel_np(L, N, 3.0))
    st = F.init_state(jax.random.PRNGKey(seed), shape, c_init=c_init, K_smooth=K)
    st = jax.tree.map(lambda x: x[None], st)
    P = F.stack_params([F.make_params(gamma=0.0, a=a, eps=eps, lam_target=lam_target, lam_scale=lam_scale, tau_lam=10.0,
                                      kappa=kappa, eta=eta, warmup=40.0)])
    fr = {k: [] for k in ("phase", "c", "lamc")}; ts = {k: [] for k in ("t", "lamc", "lam", "c_mean", "c_std", "chaos", "Rglob")}
    chist = []; bins = np.linspace(2, 7, 81); t0 = time.time()
    for i in range(steps // every + 1):
        x = np.asarray(st["x"][0]); y = np.asarray(st["y"][0]); c = np.asarray(st["c"][0]); lc = np.asarray(st["lamc"][0])
        lam = np.asarray(st["lam"][0]); ph = np.arctan2(y, x)
        fr["phase"].append(slice_grid(ph).astype(np.float16)); fr["c"].append(slice_grid(c).astype(np.float16))
        fr["lamc"].append(slice_grid(lc).astype(np.float16))
        ts["t"].append(float(st["t"][0])); ts["lamc"].append(float(lc.mean())); ts["lam"].append(float(lam.mean()))
        ts["c_mean"].append(float(c.mean())); ts["c_std"].append(float(c.std())); ts["chaos"].append(float((lc > 0).mean()))
        ts["Rglob"].append(float(np.abs(np.exp(1j * ph).mean())))
        chist.append(np.histogram(c, bins=bins)[0] / c.size)
        st = F.run_free(st, P, K, Kc, every, rule="cond", sync=kappa > 0)
    return dict(frames={k: np.stack(v) for k, v in fr.items()}, ts={k: np.array(v) for k, v in ts.items()},
                chist=np.array(chist), bins=bins, final_c=np.asarray(st["c"][0]).astype(np.float16) if L ** N < 3e6 else None,
                params=dict(N=N, L=L, steps=steps, lam_target=lam_target, kappa=kappa, a=a, eps=eps), sec=time.time() - t0)
