import _path  # noqa: F401
import json, numpy as np, sys
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
PEAKS = [[1.0], [1.0, 0.5], [0.5, 1.0], [1.0, 1.0], [1/3, 2/3, 1.0]]
def cand(mode, chi_on, i):
    c = dict(mode=mode, R=float(rng.choice([5.0, 6.0])), peaks=PEAKS[rng.integers(len(PEAKS))], seed=int(i))
    if mode == "lenia":
        c["mu"] = float(rng.uniform(0.1, 0.35)); c["sigma"] = float(c["mu"] * rng.uniform(0.08, 0.3)); c["dtA"] = float(rng.choice([0.1, 0.2]))
    else:
        c["mu"] = float(rng.uniform(0.1, 0.45)); c["sigma"] = float(c["mu"] * rng.uniform(0.1, 0.5)); c["dtA"] = float(rng.choice([0.1, 0.2]))
        c["theta"] = float(rng.choice([1.0, 2.0]))
    c["asym"] = float(rng.choice([0.0, 0.0, 0.0, 0.1, 0.2]))
    c["beta"] = float(rng.choice([0.0, 0.0, 0.3, -0.3]))
    c["chi"] = float(rng.choice([0.5, 2.0])) if chi_on else 0.0
    c["tau_c"] = float(rng.choice([2.0, 20.0]))
    c["n_blobs"] = int(rng.choice([3, 5])); c["density"] = float(rng.choice([0.4, 0.6]))
    return c
calls = []; i = 0
for mode in ("lenia", "flow"):
    for chi_on in (False, True):
        for rep in range(int(sys.argv[1])):
            cs = []
            for _ in range(16): cs.append(cand(mode, chi_on, i)); i += 1
            calls.append([dict(cands=cs, N=4, L=32, steps=2000, every=25)])
json.dump(calls, open(sys.argv[3], "w")); print(len(calls), "calls", i, "cands")
