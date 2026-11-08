import pickle, sys, glob, numpy as np, json, subprocess, os
def fetch(prefix, n):
    os.makedirs("results/glider", exist_ok=True)
    for i in range(n):
        f = f"results/glider/{prefix}_{i}.pkl"
        if not os.path.exists(f):
            subprocess.run(["/workspace/.venv-aca/bin/modal", "volume", "get", "--force", "fl-results", f"{prefix}_{i}.pkl", f], capture_output=True)
def load(prefix):
    rows = []
    for f in sorted(glob.glob(f"results/glider/{prefix}_*.pkl")):
        r = pickle.load(open(f, "rb"))
        for job in r["out"]:
            for o in job["out"]:
                o["_N"], o["_L"], o["_gpu"], o["_steps"] = job["N"], job["L"], r["gpu"], job["steps"]; rows.append(o)
    return rows
def summarize(rows, top=25):
    out = []
    for o in rows:
        c = o["cand"]; L, N = o["_L"], o["_N"]; V = L ** N
        tr = o["tracks"]
        # лучший "движущийся" трек: долгоживущий, локализованный
        mv = [t for t in tr if t["life"] >= 0.25 * o.get("_steps", 2000) and t["size_max"] < 0.1 * V]
        best = max(mv, key=lambda t: t["net"]) if mv else None
        out.append(dict(i=c["seed"], mode=c["mode"], chi=c["chi"], asym=c["asym"], beta=c["beta"], mu=round(c["mu"], 3), sig=round(c["sigma"], 4),
                        R=c["R"], pk=len(c["peaks"]), mfrac=o["mass"][-1] / V, ncomp=o["ncomp_end"], ng=o["n_glider"], ns=o["n_static"],
                        nloc=len(mv), best_net=best["net"] if best else 0, best_str=best["straight"] if best else 0,
                        best_corr=best["corr"] if best else 0, best_cv=best["mass_cv"] if best else 0, best_speed=best["speed"] if best else 0))
    out.sort(key=lambda d: (-d["ng"], -d["nloc"] * (d["best_net"] > 0), -d["best_net"]))
    print("i mode chi asym beta mu sig R pk | mfrac ncomp ngl nstat nloc | net straight corr cv speed")
    for d in out[:top]:
        print(f"{d['i']:4d} {d['mode']:5s} {d['chi']:.1f} {d['asym']:.1f} {d['beta']:+.1f} {d['mu']:.3f} {d['sig']:.4f} {d['R']:.0f} {d['pk']} | {d['mfrac']:.3f} {d['ncomp']:3d} {d['ng']:3d} {d['ns']:3d} {d['nloc']:3d} | {d['best_net']:5.1f} {d['best_str']:.2f} {d['best_corr']:.2f} {d['best_cv']:.2f} {d['best_speed']:.4f}")
    import collections
    cnt = collections.Counter(("dead" if d["mfrac"] < 1e-3 else "full" if d["mfrac"] > 0.3 else "loc") + f"/{d['mode']}" for d in out)
    print(cnt)
    return out
if __name__ == "__main__":
    fetch(sys.argv[1], int(sys.argv[2])); rows = load(sys.argv[1]); s = summarize(rows, int(sys.argv[3]) if len(sys.argv) > 3 else 25)
    json.dump(s, open(f"results/glider/{sys.argv[1]}_summary.json", "w"))
