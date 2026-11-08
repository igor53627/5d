import sys; from pathlib import Path; _root = next(p for p in [Path(__file__).resolve().parent] + list(Path(__file__).resolve().parents) if (p / "pyproject.toml").exists()); sys.path.insert(0, str(_root / "src"))
_archive = str(_root / "archive")
import json, numpy as np, jax, jax.numpy as jnp, fl_eco3 as E3, fl_brain as B, fl_glider as FG, glider_coll as GC
c = json.load(open(_root / 'archive/specs/eco13m.json'))[0]['runs'][0]['c']
eco = dict(keat=0.01, dec=0.0003, rF=0.0001, F0=0.08, pmu=0.0, hom=0.05, mutn=0.002, krep=10, ks=2, pk=0.0, kb=1.0, kr=10.0)
N, L = 4, 24; st, v = GC.soliton(c, N, L, 1, pre=200); U1 = E3.U1_of(st["A"], L, N, coarse=True); cc = dict(c, **eco, U1=U1)
s = B.make_state({k: st[k] for k in ("A","x","y","z","c")}, cc, K=B.KB, seed=5)
s["Q"] = (np.random.default_rng(0).uniform(-.9, .9, (B.KB, 1, 1, 1, 1)) * st["A"][None]).astype(np.float32)
KA = jnp.asarray(FG.ring_kernel_np(L, N, c["R"], tuple(c["peaks"])))[None]; Ko = jnp.asarray(FG.ring_kernel_np(L, N, 3.0)); KX = E3.kernels(L, N, coarse=True)
bat = lambda s: jax.tree.map(lambda a: jnp.asarray(a)[None], s); o = dict(coarse=True, fnoise=True, m=5)
r = {cl: B.run_chunk(bat(s), jax.tree.map(lambda a: a[None], B.params(dict(cc, rclamp=cl))), KA, Ko, KX, 20, True, **o) for cl in (0.0, 1.0)}
w = np.asarray(s["Q"][2:, 0, 0, 0, 0]); print("g at s=0:", B.brain(w, np.zeros(3), 2.0), " maxdiff A clamp vs not:", float(jnp.abs(r[0.0]["A"] - r[1.0]["A"]).max()))
