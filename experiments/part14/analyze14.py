import _path  # noqa: F401
"""Part 14: 6D vs Part 13 (3D–5D). python analyze14.py results/eco14_0.pkl results/eco14_1.pkl"""
import pickle, sys, re, numpy as np, analyze13 as A, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R13 = pickle.load(open("results/eco13/main13.pkl", "rb")); R6 = {}
for f in sys.argv[1:]:
    for t, r in pickle.load(open(f, "rb"))["out"][0].items():
        if t == "summary" or not isinstance(r, dict) or not r.get("runs"): continue
        R6[t] = A.one(r["runs"][0], 28); R6[t]["partial"] = bool(r["runs"][0].get("partial")); R6[t]["M1"] = r.get("M1")
pickle.dump(R6, open("results/eco14/main14.pkl", "wb"))
keys = ["n_pl", "sd_pl", "sd_slope_pl", "dp0_5k", "dp0", "S_growth", "S_pl", "h", "div_rate_k", "merge_rate_k", "t_plateau", "lin_eff"]
G = {}
for t, v in R13.items():
    m = re.match(r"(sel|neu)_(\d)D\d+_s", t)
    if m and m.group(2) in "345": G.setdefault((m.group(1), int(m.group(2))), []).append(v)
for t, v in R6.items(): G[(t.split("_")[0], 6)] = [v]
lines = ["| cond | D | n seeds | T | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 4)]
for (cn, D) in sorted(G):
    rs = G[(cn, D)]; T = int(rs[0]["ts"]["t"][-1])
    lines.append(f"| {cn} | {D} | {len(rs)} | {T} | " + " | ".join(f"{np.nanmean([r[k] for r in rs]):.3g}" + (f"±{np.nanstd([r[k] for r in rs], ddof=1) / np.sqrt(len(rs)):.2g}" if len(rs) > 1 else "") for k in keys) + " |")
open("results/part14_table.md", "w").write("\n".join(lines)); print("\n".join(lines))
fig, ax = plt.subplots(2, 4, figsize=(17, 8)); ax = ax.ravel()
for a, k, lab in zip(ax, ["div_rate_k", "merge_rate_k", "dp0_5k", "sd_pl", "h", "lin_eff", "t_plateau", "n_pl"],
                     ["divisions/ind/1000 steps", "merges/ind/1000 steps (upper bnd)", "Δ⟨p0⟩ first 5k", "sd(p0) plateau", "heritability h", "eff. founder lineages (end)", "t_plateau", "n on plateau"]):
    for cn, c, o in (("sel", "C3", -.08), ("neu", "C0", .08)):
        D = [d for d in (3, 4, 5, 6) if (cn, d) in G]; m = [np.nanmean([r[k] for r in G[(cn, d)]]) for d in D]
        se = [np.nanstd([r[k] for r in G[(cn, d)]], ddof=1) / np.sqrt(len(G[(cn, d)])) if len(G[(cn, d)]) > 1 else 0 for d in D]
        a.errorbar(np.array(D) + o, m, yerr=se, fmt="o-", color=c, label=cn, capsize=3)
    a.set_xticks([3, 4, 5, 6]); a.set_title(lab, fontsize=10); a.axvspan(5.5, 6.5, color="y", alpha=.12)
ax[0].legend(); fig.suptitle("Part 14: 6D (1 seed, 28⁶, shaded) vs 3D–5D (8 seeds, Part 13)"); fig.tight_layout(); fig.savefig("glider_part14_dims.png", dpi=90)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
for t, v in R6.items():
    for a, k in zip(ax, ["n", "p0", "sd"]): a.plot(v["ts"]["t"], v["ts"][k], "-" if t.startswith("sel") else "--", color="C3" if t.startswith("sel") else "C0", label=t)
for a, k in zip(ax, ["number of objects", "⟨p0⟩", "sd(p0)"]): a.set_title("6D 28⁶: " + k); a.set_xlabel("step")
ax[0].legend(); fig.tight_layout(); fig.savefig("glider_part14_6D_timeseries.png", dpi=90)
