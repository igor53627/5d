import sys; _R = __import__("pathlib").Path(__file__).resolve().parents[1]; [sys.path.insert(0, str(p)) for p in [_R, _R / "src"] + list((_R / "experiments").glob("part*"))]
"""Сверка JAX-порта с numpy-версией на маленькой решётке (4D, L=8) из одинакового начального состояния."""
import json, sys, time, types
import numpy as np, jax, jax.numpy as jnp
from feigenbaum_lenia import FeigenbaumLenia
import fl_jax as F

def main(L=8, N=4, steps=(1, 10, 50, 200, 1000, 4000), warmup=2.0):
    p = types.SimpleNamespace(N=N, L=L, R=3.0, peaks=[1.0], Rc=3.0, eps=0.1, dt=0.05, a=0.2, b=0.2, c_init=5.0,
                              c_init_spread=1.2, cmin=2.0, cmax=7.0, eta=0.1, lam_target=0.02, lam_scale=0.05,
                              tau_lam=30.0, kappa=0.7, mu=0.5, sigma=0.2, growth="bell", warmup=warmup, sync="amp",
                              sigA=2.0, Dc=0.02, seed=1)
    sim = FeigenbaumLenia(p)
    shape = (L,) * N
    K = jnp.asarray(F.lenia_kernel_np(L, N, p.R)); Kc = jnp.asarray(F.lenia_kernel_np(L, N, p.Rc))
    st = {k: jnp.asarray(getattr(sim, k))[None] for k in ["x", "y", "z", "c", "dx", "dy", "dz", "lam"]}
    st.update(win=jnp.zeros((1,) + shape), t=jnp.zeros((1,)), x2=st["x"], y2=st["y"], z2=st["z"], lamc=jnp.zeros((1,) + shape))
    P = F.stack_params([F.make_params(eps=p.eps, warmup=warmup, R=0)]) if False else F.stack_params([F.make_params(eps=p.eps, warmup=warmup)])
    out = []; done = 0
    for n in steps:
        st = F.run_free(st, P, K, Kc, n - done, rule="own", sync=True)
        for _ in range(n - done): sim.step()
        done = n
        r = {"steps": n}
        for k in ["x", "z", "c", "lam"]:
            A = np.asarray(st[k][0]); B = getattr(sim, k)
            r[f"maxdiff_{k}"] = float(np.abs(A - B).max()); r[f"mean_{k}_jax"] = float(A.mean()); r[f"mean_{k}_np"] = float(B.mean())
        r["frac_chaos_jax"] = float((np.asarray(st["lam"][0]) > 0.01).mean()); r["frac_chaos_np"] = float((sim.lam > 0.01).mean())
        out.append(r); print(json.dumps(r), flush=True)
    json.dump(out, open("results/validate_jax.json", "w"), indent=1)

if __name__ == "__main__":
    main()
