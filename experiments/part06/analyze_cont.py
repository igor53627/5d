import sys; from pathlib import Path; _root = next(p for p in [Path(__file__).resolve().parent] + list(Path(__file__).resolve().parents) if (p / "pyproject.toml").exists()); sys.path.insert(0, str(_root / "src"))
"""Скачать cont*-pkl с Volume и напечатать/сохранить таблицу стадий по веткам."""
import sys, os, pickle, json, subprocess, numpy as np
prefix = sys.argv[1]; n = int(sys.argv[2]); fetch = len(sys.argv) < 4 or sys.argv[3] != "nofetch"
os.makedirs("results/cont", exist_ok=True)
M = "/workspace/.venv-aca/bin/modal"
rows = []
for i in range(n):
    f = f"results/cont/{prefix}_{i}.pkl"
    if fetch:
        subprocess.run([M, "volume", "get", "--force", "fl-results", f"{prefix}_{i}.pkl", f], capture_output=True)
    if not os.path.exists(f): print("missing", f); continue
    r = pickle.load(open(f, "rb"))["out"][0]
    c = r["cand"]; print(f"\n=== {prefix}_{i}: seed {c['seed']} {c['mode']} chi={c.get('chi')} partial={r.get('partial')} t_gpu={r.get('t_gpu',0):.0f}s info={ {k:(round(v,3) if isinstance(v,float) else v) for k,v in r['info'].items() if k!='iso_pick'} }")
    for b, br in enumerate(r["branches"]):
        line = []
        for s in r["stats"][b]:
            line.append(f"{s['asym']:+.3f}:{s.get('v_axis_med', float('nan')):+.4f}/{s.get('speed_med', float('nan')):.4f} m{s.get('frac_moving', 0):.1f} n{s.get('n_long', 0)} c{s.get('corr_med', float('nan')):.2f} M{s.get('mass_frac', float('nan')):.3f}")
            rows.append(dict(run=f"{prefix}_{i}", seed=c["seed"], branch=br["name"], **{k: v for k, v in s.items()}))
        print(f"{br['name']:10s} | " + " | ".join(line))
json.dump(rows, open(f"results/cont/{prefix}_stats.json", "w"), default=float)
