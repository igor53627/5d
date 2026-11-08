import sys; _R = __import__("pathlib").Path(__file__).resolve().parents[2]; [sys.path.insert(0, str(p)) for p in [_R, _R / "src"] + list((_R / "experiments").glob("part*"))]
import pickle, numpy as np, analyze_eco as AE
from scipy import stats
L = 64; R = {}
for i in range(3):
    o = pickle.load(open(f'results/p11/val11_{i}.pkl', 'rb'))['out'][0]
    for vn, oo in o.items():
        r = oo['runs'][0]; s = AE.summarize(r, L); ps = AE.pstats(r['log'], 0); log = r['log']
        Se = AE.selection_diff([x for x in log if x[0] <= 7000], L, lag=5)
        n = np.array(s['n']); t = np.array([x[0] for x in log])
        R.setdefault(vn, []).append(dict(n_end=n[-10:].mean(), n_max=n.max(), M_end=np.mean(s['M'][-10:]), M_max=max(s['M']), div=s['n_div'], h=AE.herit(s['pairs'], 0)[0],
            p0_6k=ps[30][1], p0_end=np.mean([x[1] for x in ps[-5:]]), dp0=np.mean([x[1] for x in ps[-5:]]) - ps[0][1], sd_end=ps[-1][2], Se=Se[0], wall=oo['wall'], T=t[-1],
            n_t=n, p0_t=np.array([x[1] for x in ps]), M_t=np.array(s['M'])))
keys = ['n_end', 'n_max', 'M_end', 'M_max', 'div', 'h', 'p0_6k', 'dp0', 'sd_end', 'Se', 'wall']
print('variant | ' + ' | '.join(keys))
for vn, rs in R.items():
    print(vn, ' | '.join(f"{np.mean([r[k] for r in rs]):.3g}±{np.std([r[k] for r in rs], ddof=1):.2g}" for k in keys))
print('\npaired t-test vs base (3 seeds); rel diff of mean; |diff|/sd_between_seeds(base)')
for vn in R:
    if vn == 'base': continue
    row = []
    for k in keys[:-1]:
        a = np.array([r[k] for r in R['base']]); b = np.array([r[k] for r in R[vn]])
        p = stats.ttest_rel(a, b).pvalue if np.std(a - b) > 0 else np.nan
        row.append(f"{k}: {np.mean(b)-np.mean(a):+.3g} (p={p:.2f}, z={abs(np.mean(b)-np.mean(a))/(np.std(a,ddof=1)+1e-12):.1f})")
    print(vn, '; '.join(row))
# trajectory difference
for vn in R:
    if vn == 'base': continue
    dn = [np.abs(R[vn][i]['n_t'] - R['base'][i]['n_t']).max() / R['base'][i]['n_t'].max() for i in range(3)]
    dp = [np.abs(R[vn][i]['p0_t'] - R['base'][i]['p0_t']).max() for i in range(3)]
    sp = [np.abs(R['base'][i]['p0_t'] - R['base'][j]['p0_t']).max() for i in range(3) for j in range(i + 1, 3)]
    print(vn, 'max|Δn|/n_max', np.round(dn, 2), 'max|Δp0(t)|', np.round(dp, 3), '(between-seed base max|Δp0|', np.round(sp, 2), ')')
pickle.dump(R, open('results/p11/val11_summary.pkl', 'wb'))
