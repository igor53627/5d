import sys; from pathlib import Path; _root = next(p for p in [Path(__file__).resolve().parent] + list(Path(__file__).resolve().parents) if (p / "pyproject.toml").exists()); sys.path.insert(0, str(_root / "src"))
_archive = str(_root / "archive")
import sys, pickle, subprocess, os
prefix=sys.argv[1]; n=int(sys.argv[2]); os.makedirs(f"{_archive}/results/es", exist_ok=True)
for i in range(n):
    f=f"{_archive}/results/es/{prefix}_{i}.pkl"
    subprocess.run(["/workspace/.venv-aca/bin/modal","volume","get","--force","fl-results",f"{prefix}_{i}.pkl",f],capture_output=True)
    r=pickle.load(open(f,"rb"))["out"][0]
    if r.get("partial"):
        s=r["summ"]; b=s["best"]; print(i, "gen", s["gen"], "best_fit", round(s["best_fit"],3), {k:(round(v,3) if isinstance(v,float) else v) for k,v in b.items() if k!='vel'}, "sig", round(s["sigma"],3), f"{s['t']:.0f}s")
        import collections; g=collections.Counter(round(h['fit'],2)>0 for h in r['hist']); print("   nonzero fit:", g)
    else:
        print(i, "DONE", r["summary"])
