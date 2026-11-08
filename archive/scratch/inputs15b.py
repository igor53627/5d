import _path  # noqa: F401
"""Part 15b: realistic ranges of brain inputs s1=Ub/U1-1, s2=Ufar/U1 (4D), from a single CPU soliton embedded in 48^4, and with a neighbour at distance d."""
import json, numpy as np, jax, jax.numpy as jnp, fl_eco3 as E3, glider_coll as GC, pickle
c = json.load(open(_path._root / 'archive/specs/eco13m.json'))[0]['runs'][0]['c']; N = 4
st, v = GC.soliton(c, N, 24, 1, pre=300); A0 = np.zeros((48,) * N, np.float32); A0[(slice(12, 36),) * N] = st["A"]
KX = E3.kernels(48, N, coarse=True)
wconv = lambda G, Kk: np.asarray(E3.up2(jnp.fft.irfftn(jnp.fft.rfftn(E3.pool2(jnp.asarray(G))) * Kk, s=(24,) * N)))
U1 = float(wconv(A0, KX["big"]).max()); out = {"U1": U1}
for d in (None, 8, 12, 16, 20, 24):
    A = A0.copy() if d is None else A0 + np.roll(A0, d, 0)
    Ub = wconv(A, KX["big"]); Uf = wconv(A, KX["far"]); m = A0 > 0.05; w = A0[m]
    s1 = Ub[m] / U1 - 1; s2 = Uf[m] / U1
    q = lambda x: [float(np.average(x, weights=w)), *[float(np.percentile(x, p)) for p in (5, 50, 95)]]
    out[str(d)] = dict(s1=q(s1), s2=q(s2)); print(d, "s1 mean/p5/p50/p95", np.round(q(s1), 3), "s2", np.round(q(s2), 3))
pickle.dump(out, open("results/eco15/inputs15b.pkl", "wb"))
