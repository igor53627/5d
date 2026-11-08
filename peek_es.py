import sys, pickle, subprocess, os
prefix=sys.argv[1]; n=int(sys.argv[2]); os.makedirs("results/es", exist_ok=True)
for i in range(n):
    f=f"results/es/{prefix}_{i}.pkl"
    subprocess.run(["/workspace/.venv-aca/bin/modal","volume","get","--force","fl-results",f"{prefix}_{i}.pkl",f],capture_output=True)
    r=pickle.load(open(f,"rb"))["out"][0]
    if r.get("partial"):
        s=r["summ"]; b=s["best"]; print(i, "gen", s["gen"], "best_fit", round(s["best_fit"],3), {k:(round(v,3) if isinstance(v,float) else v) for k,v in b.items() if k!='vel'}, "sig", round(s["sigma"],3), f"{s['t']:.0f}s")
        import collections; g=collections.Counter(round(h['fit'],2)>0 for h in r['hist']); print("   nonzero fit:", g)
    else:
        print(i, "DONE", r["summary"])
