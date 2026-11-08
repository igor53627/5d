import sys, pickle, numpy as np
R=pickle.load(open(sys.argv[1],'rb'))['out'][0]
print('partial', R['partial'])
for k,v in R['runs'].items():
    if isinstance(v, list):
        for x in v:
            print(k, 'cand', x['cand'], 'seed', x['seed'], f"net={x['net']:.1f} v={np.linalg.norm(x['vel']):.4f} str={x['straight']:.3f} r50={x['r50']:.3f} r250={x['r250']:.3f} p10={x['r50_p10']:.3f} loc0={x['loc0']:.2f} locw={x['locw']:.2f} ok={x['ok']} vel={np.round(x['vel'],3).tolist()}")
    else:
        print(k, 'axis', v['axis'], 'gap', v['gap'], [(l['t'], l['n'], l['masses'][:3]) for l in v['log']])
