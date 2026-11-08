import _path  # noqa: F401
_root = _path._root
_archive = str(_root / "archive") if _root is not None else ""
import sys, pickle, subprocess, os, numpy as np
sys.path.insert(0,'.'); import fl_glider as FG
prefix=sys.argv[1]; ids=[int(x) for x in sys.argv[2].split(',')]; os.makedirs(f"{_archive}/results/ver", exist_ok=True)
names="ABCDE"
for i in ids:
    f=f"{_archive}/results/ver/{prefix}_{i}.pkl"
    subprocess.run(["/workspace/.venv-aca/bin/modal","volume","get","--force","fl-results",f"{prefix}_{i}.pkl",f],capture_output=True)
    if not os.path.exists(f): print(i,'missing'); continue
    R=pickle.load(open(f,"rb"))["out"][0]
    print(f"--- {prefix}_{i} partial={R.get('partial')}")
    for r in R["res"]:
        if r["type"]=="iso":
            for k,o in enumerate(r["out"]):
                tr=o.get("traj",[]); L=r["L"]
                if len(tr)>2:
                    X=np.array([x for _,x in tr]); d=np.array([FG.pdelta(X[j],X[j+1],L) for j in range(len(X)-1)]); tot=d.sum(0); T=tr[-1][0]-tr[0][0]
                    net=np.linalg.norm(tot); path=np.linalg.norm(d,axis=1).sum()
                    print(f" iso N{r['N']} seed{r['seed']} flip{int(r['flip'])} k{k}(chi={r['cands'][k]['chi']:.2f},b={r['cands'][k]['beta']:.2f},pk={len(r['cands'][k]['peaks'])}): alive={o['alive']} T={T} net_all={net:.1f} v={net/max(T,1):.4f} straight_all={net/max(path,1e-9):.2f} dir={np.round(tot/max(net,1e-9),2)} | 2/3-window: net={o['net']:.1f} str={o['straight']:.2f} corr={o['corr']:.2f} mcv={o.get('mass_cv',0):.2f}")
                else:
                    print(f" iso N{r['N']} seed{r['seed']} flip{int(r['flip'])} {k}: dead/split, alive={o.get('alive')} n={len(tr)}")
        else:
            for o in r["out"]:
                tl=[t for t in o["tracks"] if t["life"]>=0.25*r["steps"]]
                print(f" soup N{r['N']} L{r['L']}: n_glider={o['n_glider']} ncomp_end={o['ncomp_end']} massfrac={o['mass'][-1]/r['L']**r['N']:.4f} long tracks={len(tl)}")
                for t in sorted(tl,key=lambda t:-t['net'])[:8]: print(f"   life={t['life']} net={t['net']:.1f} v={t['speed']:.4f} str={t['straight']:.2f} corr={t['corr']:.2f} mcv={t['mass_cv']:.2f} size={t['size_max']:.0f}")
