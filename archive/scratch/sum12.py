import _path  # noqa: F401
import pickle, glob, sys, numpy as np, analyze12 as A
L = int(sys.argv[1]); R = {}
for f in sys.argv[2:]:
    o = pickle.load(open(f, 'rb'))['out'][0]
    for t, r in o.items():
        if t == 'summary' or not r.get('runs'): continue
        rr = r['runs'][0]; m = A.metrics(rr, L, eco=rr['eco']); R[t] = m
        n = m['n']; idx = np.linspace(0, len(n) - 1, 6).astype(int)
        print(f"{t:12s} T{m['T']} n{list(n[idx])} M{[round(x) for x in m['M'][idx]]} p0{list(np.round(m['p0'][idx],2))} sd{list(np.round(m['sd'][idx],2))} div{m['div']} h{m['h']:.2f} "
              f"S_pl{m['S_plateau']:+.3f}(c{m['corr_plateau']:+.2f}) sd_pl{m['sd_pl']:.2f} k{m['k_mode']} lin{m['lin_n'][-1]}/{m['lin_eff'][-1]:.1f} "
              f"Sw{[round(x[1],3) for x in m['Sw']]} {('LA %.2f' % m['local_adapt']) if 'local_adapt' in m else ''} {m.get('season_track','')}")
pickle.dump(R, open(f'results/eco12/sum_{sys.argv[2].split("/")[-1].split("_")[0]}.pkl', 'wb'))
