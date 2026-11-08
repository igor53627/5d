import json, numpy as np, sys, copy
rng = np.random.default_rng(2)
S1 = {c["seed"]: c for call in json.load(open("specs/glider_s1.json")) for c in call[0]["cands"]}
def pert(base, i, **over):
    c = copy.deepcopy(base); c["seed"] = 10000 + i
    c["mu"] = float(base["mu"] * rng.uniform(0.9, 1.1)); c["sigma"] = float(base["sigma"] * rng.uniform(0.8, 1.25))
    c.update(over); return c
calls = []; i = 0
# A: Lenia-солитон #23 + фазовая адвекция / обратная связь / слабая асимметрия
b = S1[23]
grp_nochi, grp_chi = [], []
for k in range(32):
    grp_nochi.append(pert(b, i, beta=float(rng.choice([0, 0.3, -0.3, 0.6, -0.6])), asym=float(rng.choice([0, 0, 0.02, 0.05])),
                          tau_c=float(rng.choice([2.0, 20.0])), n_blobs=int(rng.choice([3, 5])), dtA=float(rng.choice([0.1, 0.2])))); i += 1
for k in range(32):
    grp_chi.append(pert(b, i, chi=float(rng.choice([0.25, 0.5, 1.0, 2.0, 4.0])), beta=float(rng.choice([0, 0.3, -0.3])), asym=0.0,
                        tau_c=float(rng.choice([2.0, 20.0])), n_blobs=int(rng.choice([3, 5])), dtA=float(rng.choice([0.1, 0.2])))); i += 1
for g in (grp_nochi[:16], grp_nochi[16:], grp_chi[:16], grp_chi[16:]):
    calls.append([dict(cands=g, N=4, L=32, steps=3000, every=25)])
# B: Flow-Lenia + χ вокруг 150/159/146/134
fl = []
for base in (S1[150], S1[159], S1[146], S1[134]):
    for k in range(12):
        fl.append(pert(base, i, chi=float(rng.choice([0.5, 1.0, 2.0, 4.0])), beta=float(rng.choice([0, 0.3, -0.3, 0.6])), asym=0.0,
                       tau_c=float(rng.choice([2.0, 20.0])))); i += 1
for j in range(3): calls.append([dict(cands=fl[16 * j:16 * j + 16], N=4, L=32, steps=3000, every=25)])
# C: навязанный дрейф (#37) — эталон «движущейся» структуры с заданным направлением
dr = [pert(S1[37], i + k, asym=float(rng.choice([0.05, 0.1, 0.15])), beta=float(rng.choice([0, -0.3, -0.6]))) for k in range(16)]; i += 16
calls.append([dict(cands=dr, N=4, L=32, steps=3000, every=25)])
json.dump(calls, open("specs/glider_s2.json", "w")); print(len(calls), i)
