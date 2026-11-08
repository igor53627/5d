import _path  # noqa: F401
_root = _path._root
_archive = str(_root / "archive") if _root is not None else ""
import json, numpy as np, time, jax, jax.numpy as jnp, fl_eco3 as E3, fl_brain as B, fl_glider as FG, glider_coll as GC
c = json.load(open(_root / 'archive/specs/eco13m.json'))[0]['runs'][0]['c']
eco = dict(keat=0.01, dec=0.0003, rF=0.0001, F0=0.08, pmu=0.0, hom=0.05, mutn=0.002, krep=10, ks=2, pk=0.0)
N, L = 4, 24
st, v = GC.soliton(c, N, L, 1, pre=300)
U1 = E3.U1_of(st["A"], L, N, coarse=True); cc = dict(c, **eco, U1=U1)
rng = np.random.default_rng(0)
def mk(Emod, K):
    s = Emod.make_state({k: st[k] for k in ("A","x","y","z","c")}, cc, K=K, seed=5)
    s["Q"] = (rng.uniform(-.8, .8, (K, 1, 1, 1, 1)) * st["A"][None]).astype(np.float32); return s
KA = jnp.asarray(FG.ring_kernel_np(L, N, c["R"], tuple(c["peaks"])))[None]; Ko = jnp.asarray(FG.ring_kernel_np(L, N, 3.0)); KX = E3.kernels(L, N, coarse=True)
o = dict(coarse=True, fnoise=True, m=5)
s3 = mk(E3, B.KB); sb = dict(s3, D=np.zeros((B.ND,) + st["A"].shape, np.float32))
bat = lambda s: jax.tree.map(lambda a: jnp.asarray(a)[None], s)
P3 = jax.tree.map(lambda a: a[None], E3.params(cc)); Pb = jax.tree.map(lambda a: a[None], B.params(dict(cc, kb=0.0, kr=0.0)))
r3 = E3.run_chunk(bat(s3), P3, KA, Ko, KX, 50, True, **o); rb = B.run_chunk(bat(sb), Pb, KA, Ko, KX, 50, True, **o)
print("kb=kr=0 vs eco3: maxdiff A", float(jnp.abs(r3["A"] - rb["A"]).max()), "Q", float(jnp.abs(r3["Q"] - rb["Q"]).max()))
Pb2 = jax.tree.map(lambda a: a[None], B.params(dict(cc, kb=1.0, kr=10.0)))
t0 = time.time(); r2 = B.run_chunk(bat(sb), Pb2, KA, Ko, KX, 50, True, **o); jax.block_until_ready(r2["A"]); print("brain on: mass", float(r2["A"].sum()), "vs off", float(rb["A"].sum()), "t", time.time() - t0)
D = np.asarray(r2["D"][0]); A = np.asarray(r2["A"][0]); print("D sums / mass:", [float(D[i].sum() / A.sum()) for i in range(4)])
print("diff A brain on/off", float(jnp.abs(r2["A"] - rb["A"]).max()))
w = np.asarray(r2["Q"][0, 2:].sum(axis=(1,2,3,4)) / A.sum()); print("brain(w, s=0)", B.brain(w, np.zeros(3), 2.0))
