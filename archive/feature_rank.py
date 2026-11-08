import sys; _R = __import__("pathlib").Path(__file__).resolve().parents[1]; [sys.path.insert(0, str(p)) for p in [_R, _R / "src"] + list((_R / "experiments").glob("part*"))]
"""Эффективная размерность признаков резервуара (participation ratio спектра PCA) на входе Mackey-Glass."""
import json, numpy as np
from reservoir_experiment import BatchFL, FL_DEFAULT, make_input
def pr(X):
    X = X - X.mean(0); X = X / (X.std(0) + 1e-8); ev = np.clip(np.linalg.eigvalsh(X.T @ X / len(X)), 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum()), float((np.cumsum(ev[::-1]) / ev.sum() < 0.99).sum() + 1)
u = make_input("mg", 900); out = {}
for a, tag in [(0.2, "a=0.2"), (-0.05, "a=-0.05")]:
    P = dict(FL_DEFAULT); P["a"] = a
    fl = BatchFL(3, 4, 8, 3.0, 0.1, 2.0, [2.5, 2.8, 5.0], [False] * 3, 1.0, 0, P)
    sel = np.random.default_rng(0).choice(4096, 400, replace=False); F = np.zeros((3, 900, 1200))
    for _ in range(1200): fl.step(0.0, 0.05)
    for k in range(900):
        acc = 0
        for _ in range(10):
            fl.step(u[k], 0.05); acc = acc + np.concatenate([fl.x.reshape(3, -1)[:, sel], fl.y.reshape(3, -1)[:, sel], fl.z.reshape(3, -1)[:, sel]], 1)
        F[:, k] = acc / 10
    for i, c in enumerate([2.5, 2.8, 5.0]): out[f"FL {tag} c={c}"] = pr(F[i, 300:])
rng = np.random.default_rng(0); N = 1200
W = rng.standard_normal((N, N)) * (rng.random((N, N)) < 10 / N); W *= 1.1 / np.max(np.abs(np.linalg.eigvals(W)))
win = rng.uniform(-1, 1, N); b = rng.uniform(-.2, .2, N); h = np.zeros(N); H = np.zeros((900, N))
for k in range(900): h = np.tanh(W @ h + 0.1 * win * u[k] + b); H[k] = h
out["ESN rho=1.1"] = pr(H[300:])
for k, v in out.items(): print(f"{k:22s} PR={v[0]:6.1f}  #PC(99%)={v[1]:.0f}")
json.dump(out, open("results/feature_rank.json", "w"), indent=1)
