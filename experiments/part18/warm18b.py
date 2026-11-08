import sys; from pathlib import Path; _root = next(p for p in [Path(__file__).resolve().parent] + list(Path(__file__).resolve().parents) if (p / "pyproject.toml").exists()); sys.path.insert(0, str(_root / "src"))
"""Part 18b: map Part-17 evolved brains (4→3→3) onto the Part-18 architecture (7→3→4) as warm-start founders."""
import pickle, glob, numpy as np, fl_poison as P17, fl_season as Z
def best17(tag):
    for f in glob.glob("results/eco17/b17m_*.pkl"):
        for t, r in pickle.load(open(f, "rb"))["out"][0].items():
            if t == tag:
                log = r["runs"][0]["log"]; objs = log[-1][1]; o = max(objs, key=lambda o: o[0]); return np.array(o[2][:P17.KB])
def map_w(q17):
    w = q17[P17.K0:P17.KB]; W1, b1, W2, b2 = P17.unpack(w, 1.0)               # raw (unscaled) values
    W1n = np.zeros((Z.H, Z.NIN)); W1n[:, 0], W1n[:, 1], W1n[:, 5], W1n[:, 6] = W1[:, 0], W1[:, 1], W1[:, 2], W1[:, 3]
    b1n = b1 - W1[:, 0]; W2n = np.zeros((Z.NOUT, Z.H)); W2n[:3] = W2; b2n = np.r_[b2, 0.0]
    wn = np.r_[W1n.ravel(), b1n, W2n.ravel(), b2n]; clip = float(np.abs(wn).max())
    p = np.zeros(Z.KB); p[:Z.K0] = q17[:P17.K0]; p[Z.K0:Z.iM] = np.clip(wn, -1, 1); return p, clip
if __name__ == "__main__":
    out = {}
    for s in (511, 512, 513, 514):
        q = best17(f"evo_s{s}"); p, cl = map_w(q); out[s] = p
        w17 = q[P17.K0:P17.KB]; s17 = np.array([-0.3, 0.1, -0.4, 0.06]); o17 = P17.brain(w17, s17, 2.0)
        s18 = np.array([0.7, 0.1, 0.0, 0.0, 0.0, -0.4, 0.06]); o18 = Z.brain(p[Z.K0:Z.iM], s18, 2.0)
        print(s, "max|w|", round(cl, 2), "P17 out", np.round(o17, 3), "P18 out", np.round(o18, 3))
    pickle.dump(out, open("results/warm18b.pkl", "wb"))
