"""Part 13: dimension comparison. python analyze13.py results/eco13/eco13m_*.pkl"""
import pickle, sys, re, json, numpy as np, analyze12 as A
from scipy import stats
def lfromtag(t): return int(re.match(r"\w+?_(\d)D(\d+)", t).group(2))
def one(r, L):
    m = A.metrics13(r, L); t = m["t"]; tp = m["t_plateau"]
    Sw = np.array([(a, s) for a, s, _ in m["Sw"]], float)
    g = Sw[(Sw[:, 0] < max(tp, 2000)), 1]; g = g[np.isfinite(g)]
    pl = t >= t[-1] // 2
    sl = np.polyfit(t[pl] / 1e4, m["sd"][pl], 1)[0] if pl.sum() > 3 and np.isfinite(m["sd"][pl]).all() else np.nan
    i5 = min(np.searchsorted(t, 5000), len(t) - 1)
    return dict(n_pl=m["n_pl"], n_end=float(m["n"][-1]), sd0=float(m["sd"][0]), sd5k=float(m["sd"][i5]), sd_pl=m["sd_pl"], sd_slope_pl=float(sl),
                dp0=float(m["p0"][-1] - m["p0"][0]), dp0_5k=float(m["p0"][i5] - m["p0"][0]), S_growth=float(g.mean()) if len(g) else np.nan,
                S_pl=float(m["S_plateau"]), h=float(m["h"]), div_rate_k=1000 * m["div_rate"], merge_rate_k=1000 * m["merge_rate"], merges=m["merges"],
                t_plateau=tp, lin_eff=float(m["lin_eff"][-1]), extinct=bool(m["n"][-1] == 0), k_mode=m["k_mode"],
                ts=dict(t=t, n=m["n"], p0=m["p0"], sd=m["sd"], M=m["M"]))
if __name__ == "__main__":
    R = {}
    for f in sys.argv[1:]:
        o = pickle.load(open(f, "rb"))["out"][0]
        for tag, r in o.items():
            if tag == "summary" or not isinstance(r, dict) or not r.get("runs") or r["runs"][0].get("partial"): continue
            R[tag] = one(r["runs"][0], lfromtag(tag)); print(tag, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in R[tag].items() if k != "ts"})
    pickle.dump(R, open("results/eco13/main13.pkl", "wb"))
    groups = sorted({re.sub(r"_s\d+$", "", t) for t in R}); keys = ["n_pl", "sd_pl", "sd_slope_pl", "dp0", "dp0_5k", "S_growth", "S_pl", "h", "div_rate_k", "merge_rate_k", "t_plateau", "lin_eff"]
    lines = ["| group | N | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 2)]; G = {}
    for g in groups:
        rs = [v for t, v in R.items() if re.sub(r"_s\d+$", "", t) == g]; G[g] = {k: np.array([r[k] for r in rs], float) for k in keys}
        lines.append(f"| {g} | {len(rs)} | " + " | ".join(f"{np.nanmean(G[g][k]):.3g}±{np.nanstd(G[g][k], ddof=1) / np.sqrt(np.isfinite(G[g][k]).sum()):.2g}" for k in keys) + " |")
    st = {}
    for g in groups:
        if g.startswith("sel_"):
            h = "neu_" + g[4:]
            if h in G: st[g[4:]] = {k: float(stats.ttest_ind(G[g][k], G[h][k], equal_var=False, nan_policy="omit").pvalue) for k in keys}
    lines.append("\nWelch p (sel vs neu):\n```\n" + json.dumps(st, indent=1) + "\n```")
    open("results/part13_table.md", "w").write("\n".join(lines)); print("\n".join(lines))
