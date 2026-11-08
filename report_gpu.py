#!/usr/bin/env python3
"""Сводка GPU-эксперимента: таблицы (markdown) + рисунки."""
import json, glob, os, sys, collections
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import reservoir_gpu as RG

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
COLS = [("mg", "h1", "MG h=1"), ("mg", "h10", "MG h=10"), ("mg", "h50", "MG h=50"),
        ("lorenz", "h1", "Lorenz h=1"), ("lorenz", "h10", "Lorenz h=10"), ("mc", "MC", "MC ↑")]
FAMS = ["cond-adapt | Рёсслер a=0.2", "cond-freeze | Рёсслер a=0.2", "cond-freeze-shuf | Рёсслер a=0.2",
        "fixed | Рёсслер a=0.2", "fixed-hetero | Рёсслер a=0.2",
        "cond-adapt | затух. a=-0.05", "cond-freeze | затух. a=-0.05", "cond-freeze-shuf | затух. a=-0.05",
        "fixed | затух. a=-0.05", "fixed-hetero | затух. a=-0.05"]

def cfg_of(r):
    return {k: r[k] for k in ("method", "a", "gamma", "eps", "lam_target", "kappa", "c0") if k in r}

def load_final():
    runs = []   # (L, task, run)
    for f in sorted(glob.glob(os.path.join(RES, "final_L*.json"))):
        for job in json.load(open(f)):
            for o in job["out"]:
                for r in o["runs"]: runs.append((o["L"], o["task"], r))
    return runs

def esn_lin():
    out = {}
    for task in ("mg", "lorenz", "mc"):
        r = json.load(open(os.path.join(RES, f"v2_{task}_esn.json")))
        for run in r["runs"]:
            nm = "ESN (1200 нейронов)" if run["cond"] == "ESN" else run["cond"].replace("Linear-delay", "Линейная регрессия, окно ")
            d = out.setdefault(nm, {})
            if task == "mc": d.setdefault(("mc", "MC"), []).append(run["MC"])
            else:
                for tn, v in zip(r["targets"], run["nrmse"]): d.setdefault((task, tn), []).append(v)
    return out

def fmt(v):
    if not v: return "—"
    m, s = np.mean(v), np.std(v)
    if abs(m) >= 10: return f"{m:.1f} ± {s:.1f}"
    if abs(m) >= 0.1: return f"{m:.3f} ± {s:.3f}"
    return f"{m:.2g} ± {s:.1g}"

def main():
    tune = [o for f in ("tune2_L8.json",) for job in json.load(open(os.path.join(RES, f))) for o in job["out"]]
    best = RG.select_best(tune)
    runs = load_final()
    Ls = sorted({L for L, _, _ in runs})
    # cell[(fam, L, task, tn)] = list over seeds
    cell, twin, diag = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    for L, task, r in runs:
        fam = RG.family(r); c = cfg_of(r)
        tns = r["targets"] if task != "mc" else ["MC"]
        for j, tn in enumerate(tns):
            bk = (task, tn, fam if not fam.startswith("cond-freeze-shuf") else fam.replace("cond-freeze-shuf", "cond-freeze"))
            if bk not in best: continue
            want = dict(best[bk][1])
            if fam.startswith("cond-freeze-shuf"): want["method"] = "cond-freeze-shuf"
            if c != want: continue
            v = r["MC"] if task == "mc" else r["nrmse"][j]
            if r["diverged"]: v = np.nan
            cell[(fam, L, task, tn)].append(v)
            if "twin_corr" in r: twin[(fam, L, task, tn)].append(r["twin_corr"])
            diag[(fam, L, task, tn)].append((r["lamc"], r["c_mean"]))
    EL = esn_lin()
    lines = ["# GPU-эксперимент: гомеостаз по условному показателю Ляпунова (NRMSE на test, mean ± std по 5 сидам; MC — сумма r², задержки 1..80)", ""]
    for L in Ls:
        lines += [f"## Решётка 4D {L}^4 = {L**4} клеток (ридаут: 400 клеток × (x,y,z) = 1200 признаков)", "",
                  "| режим | " + " | ".join(c[2] for c in COLS) + " | twin‑corr (MG h=10) | ⟨λc⟩ | ⟨c⟩ |", "|" + "---|" * (len(COLS) + 4)]
        for fam in FAMS:
            vals = [fmt([x for x in cell[(fam, L, t, tn)] if np.isfinite(x)]) + ("" if all(np.isfinite(cell[(fam, L, t, tn)])) else " (NaN!)") for t, tn, _ in COLS]
            if all(v.startswith("—") for v in vals): continue
            tw = twin[(fam, L, "mg", "h10")]; dg = diag[(fam, L, "mg", "h10")]
            lines.append(f"| {fam.replace(' | ', ' / ')} | " + " | ".join(vals) + f" | {np.mean(tw):.3f} | {np.mean([d[0] for d in dg]):+.3f} | {np.mean([d[1] for d in dg]):.2f} |" if dg else
                         f"| {fam.replace(' | ', ' / ')} | " + " | ".join(vals) + " | — | — | — |")
        lines.append("")
    lines += ["## Бейзлайны (не зависят от решётки, 5 сидов)", "", "| режим | " + " | ".join(c[2] for c in COLS) + " |", "|" + "---|" * (len(COLS) + 1)]
    for nm, d in EL.items():
        lines.append(f"| {nm} | " + " | ".join(fmt(d.get((t, tn), [])) for t, tn, _ in COLS) + " |")
    lines += ["", "## Выбранные по val гиперпараметры (тюнинг на 8^4, сид 0)", "", "| задача | цель | семейство | val | конфиг |", "|---|---|---|---|---|"]
    for k in sorted(best):
        lines.append(f"| {k[0]} | {k[1]} | {k[2].replace(' | ', ' / ')} | {best[k][0]:.4g} | " + ", ".join(f"{a}={b}" for a, b in best[k][1].items() if a not in ("method",)) + " |")
    open(os.path.join(HERE, "gpu_results.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:60]))
    figure_main(cell, EL, Ls)
    figure_sweep(tune)

def figure_main(cell, EL, Ls):
    fig, axs = plt.subplots(2, 3, figsize=(16, 9)); axs = axs.ravel()
    colors = {"cond-adapt": "tab:red", "cond-freeze": "tab:green", "cond-freeze-shuf": "tab:olive", "fixed": "tab:blue", "fixed-hetero": "tab:cyan"}
    for ax, (t, tn, title) in zip(axs, COLS):
        for fam in FAMS:
            m = fam.split(" | ")[0]; ls = "-" if "0.2" in fam else "--"
            ys = [np.nanmean(cell[(fam, L, t, tn)]) if cell[(fam, L, t, tn)] else np.nan for L in Ls]
            es = [np.nanstd(cell[(fam, L, t, tn)]) if cell[(fam, L, t, tn)] else np.nan for L in Ls]
            if np.all(np.isnan(ys)): continue
            ax.errorbar(Ls, ys, yerr=es, color=colors[m], ls=ls, marker="o", ms=4, capsize=2, label=fam)
        for nm, d in EL.items():
            if (t, tn) in d:
                ax.axhline(np.mean(d[(t, tn)]), color="k" if nm.startswith("ESN") else "gray", ls=":" if nm.startswith("ESN") else "-.", label=nm)
        ax.set_title(title); ax.set_xticks(Ls); ax.set_xticklabels([f"{L}⁴" for L in Ls])
        if t != "mc": ax.set_yscale("log"); ax.set_ylabel("NRMSE (test) ↓")
        else: ax.set_ylabel("ёмкость памяти ↑")
        ax.grid(alpha=.3)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("Feigenbaum‑Lenia как резервуар: гомеостаз по условному λc vs фиксированный c vs ESN (сплошные — Рёсслер a=0.2, пунктир — затухающая клетка a=−0.05)")
    fig.tight_layout(rect=(0, 0.1, 1, 0.96)); fig.savefig(os.path.join(HERE, "gpu_reservoir.png"), dpi=110); plt.close(fig)

def figure_sweep(tune):
    """λc, twin-неважно (нет реплик в тюнинге), val vs c0 у fixed; c и λc у cond-adapt vs цель."""
    o = [x for x in tune if x["task"] == "mg"][0]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    for a, ls in ((0.2, "-"), (-0.05, "--")):
        for (g, e), col in zip([(1.0, 0.3), (2.0, 0.3), (3.0, 0.3), (3.0, 0.1)], ["tab:blue", "tab:orange", "tab:green", "tab:purple"]):
            rs = sorted([r for r in o["runs"] if r["method"] == "fixed" and r["a"] == a and r["gamma"] == g and r["eps"] == e], key=lambda r: r["c0"])
            if not rs: continue
            axs[0].plot([r["c0"] for r in rs], [r["lamc"] for r in rs], ls=ls, color=col, marker="o", ms=3, label=f"a={a}, γ={g}, ε={e}")
            axs[1].plot([r["c0"] for r in rs], [r["val"][1] for r in rs], ls=ls, color=col, marker="o", ms=3)
            ra = sorted([r for r in o["runs"] if r["method"] == "cond-freeze" and r["a"] == a and r["gamma"] == g and r["eps"] == e and r.get("kappa", 0) == 0], key=lambda r: r["lam_target"])
            axs[2].plot([r["lam_target"] for r in ra], [r["lamc"] for r in ra], ls=ls, color=col, marker="s", ms=3)
    axs[0].axhline(0, color="k", lw=.5); axs[0].set_xlabel("фиксированный c"); axs[0].set_ylabel("⟨λc⟩ под входом MG"); axs[0].legend(fontsize=7)
    axs[0].set_title("Условный показатель Ляпунова растёт с c")
    axs[1].set_yscale("log"); axs[1].set_xlabel("фиксированный c"); axs[1].set_ylabel("val NRMSE, MG h=10"); axs[1].set_title("Качество резервуара vs c (8⁴, сид 0)")
    lo = min(r["lam_target"] for r in o["runs"] if "lam_target" in r)
    axs[2].plot([lo, 0], [lo, 0], "k:", lw=.8); axs[2].set_xlabel("цель λ*"); axs[2].set_ylabel("достигнутый ⟨λc⟩ (после washout)")
    axs[2].set_title("Гомеостаз: достигнутый λc vs цель")
    for ax in axs: ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "gpu_condlyap_sweep.png"), dpi=110); plt.close(fig)

if __name__ == "__main__":
    main()
