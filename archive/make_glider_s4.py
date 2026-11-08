import _path  # noqa: F401
_root = _path._root
_archive = str(_root / "archive") if _root is not None else ""
import json, copy, itertools
S1 = {c["seed"]: c for call in json.load(open(f"{_archive}/specs/glider_s1.json")) for c in call[0]["cands"]}
S2 = {c["seed"]: c for call in json.load(open(f"{_archive}/specs/glider_s2.json")) for c in call[0]["cands"]}
S3 = {c["seed"]: c for call in json.load(open(f"{_archive}/specs/glider_s3.json")) for c in call[0]["cands"]}
i = 30000; calls = []
def mk(base, **o):
    global i
    c = copy.deepcopy(base); c.update(o); c["seed"] = i; i += 1; return c
for base in (S1[23], S1[37], S2[10119]):
    g = [mk(base, lag=l, tau_c=t, gam=0.0, beta=0.0, chi=0.0, asym=0.0) for l, t in itertools.product((0.3, 0.6, 1.0, 1.5), (5.0, 20.0, 50.0, 100.0))]
    calls.append([dict(cands=g, N=4, L=32, steps=4000, every=25)])
json.dump(calls, open(f"{_archive}/specs/glider_s4.json", "w"))
# масштабирование: навязанный дрейф (10119) и лучший спонтанный Flow+χ (20069)
big = []
for name, base in (("drift", S2[10119]), ("flowchi", S3[20069])):
    for N, L in ((4, 64), (5, 24)):
        c = mk(base, n_blobs=12 if N == 4 else 8); c["tag"] = f"{name}_{N}D{L}"
        big.append([dict(cands=[c], N=N, L=L, steps=4000, every=50, vol_every=8, vol_ds=2)])
json.dump(big, open(f"{_archive}/specs/glider_big1.json", "w")); print(len(calls), len(big))
