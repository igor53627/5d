import sys; _R = __import__("pathlib").Path(__file__).resolve().parents[2]; [sys.path.insert(0, str(p)) for p in [_R, _R / "src"] + list((_R / "experiments").glob("part*"))]
import pickle, glob, numpy as np, analyze_eco as AE
from scipy import stats
def quad_log(log):
    out = []
    for t, O in log:
        if not O: out.append((t, O)); continue
        m = np.array([o[0] for o in O]); p = np.array([o[2][0] for o in O]); pb = np.average(p, weights=m)
        out.append((t, [(o[0], o[1], [(o[2][0] - pb) ** 2] + list(o[2][1:])) for o in O]))
    return out
Q = {}
for f in sorted(glob.glob('results/eco12/eco12m_*.pkl')):
    o = pickle.load(open(f, 'rb'))['out'][0]
    for t, r in o.items():
        if t == 'summary' or '_96_' not in t: continue
        log = r['runs'][0]['log']; ql = quad_log(log)
        Cg = AE.selection_diff([x for x in ql if x[0] <= 7000], 96, lag=5)[0]; Cp = AE.selection_diff([x for x in ql if x[0] >= 12500], 96, lag=5)[0]
        Q.setdefault(t.split('_')[0], []).append((Cg, Cp))
for c, v in Q.items():
    v = np.array(v); print(c, 'C_quad growth %.4f±%.4f' % (v[:, 0].mean(), v[:, 0].std(ddof=1)), 'plateau %.4f±%.4f p_vs0=%.3f' % (v[:, 1].mean(), v[:, 1].std(ddof=1), stats.ttest_1samp(v[:, 1], 0).pvalue), np.round(v[:, 1], 4))
a = np.array(Q['res2k'])[:, 1]; b = np.array(Q['neu'])[:, 1]; s = np.array(Q['sel'])[:, 1]
print('res2k vs neu p=%.3f, res2k vs sel p=%.3f' % (stats.ttest_ind(a, b, equal_var=False).pvalue, stats.ttest_ind(a, s, equal_var=False).pvalue))
pickle.dump(Q, open('results/eco12/quad.pkl', 'wb'))
