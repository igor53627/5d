"""Part 15b: what the evolved micro-brains encode. Uses saved Part 15 logs only."""
import pickle, glob, re, numpy as np, fl_brain as B, analyze12 as A12, analyze_eco as AE
K0, NW, iD = B.K0, B.NW, B.K0 + B.NW
SSTAR = np.array([-0.6, -0.4, 0.06])
RANGES = [("food F̃−1", np.linspace(-1, 0, 21)), ("own density Ub/U1−1", np.linspace(-0.7, 0.5, 21)), ("neighbours Ufar/U1", np.linspace(0, 0.3, 21))]
def load():
    R = {}
    for f in sorted(glob.glob("results/eco15/b15m_*.pkl")) + sorted(glob.glob("results/eco15/b15d5_*.pkl")):
        for t, r in pickle.load(open(f, "rb"))["out"][0].items():
            if t == "summary" or not isinstance(r, dict) or not r.get("runs"): continue
            rr = r["runs"][0]; R[t] = dict(log=rr["log"], eco=rr["eco"], tot=rr["tot"], L=int(re.search(r"_\dD(\d+)", t).group(1)), N=int(re.search(r"_(\d)D", t).group(1)))
    return R
def cond(t): return re.sub(r"_s\d+$", "", t)
def plateau_W(r, frac=0.5, every=5):
    """weights and masses of individuals on the plateau (t ≥ frac·T), every 5th snapshot."""
    T = r["log"][-1][0]; W = []; M = []; S = []
    for k, (t, objs) in enumerate(r["log"]):
        if t < frac * T or k % every: continue
        for o in objs: W.append(o[2][K0:iD]); M.append(o[0]); S.append(t)
    return np.array(W, float), np.array(M, float), np.array(S)
def brain_batch(W, s):
    """W: (n, NW), s: (NIN,) or (n_s, NIN) → o (n, n_s, NOUT)."""
    s = np.atleast_2d(s); W1, b1, W2, b2 = B.unpack(W.T, 2.0)      # W1 (H,NIN,n) ...
    h = np.tanh(np.einsum("hin,si->snh", W1, s) + b1.T[None])
    return np.transpose(np.tanh(np.einsum("ohn,snh->sno", W2, h) + b2.T[None]), (1, 0, 2))
def curves(W, M):
    out = {}
    for j, (nm, xs) in enumerate(RANGES):
        S = np.repeat(SSTAR[None], len(xs), 0); S[:, j] = xs
        o = brain_batch(W, S); out[nm] = (xs, np.average(o, axis=0, weights=M))   # (n_s, NOUT)
    return out
