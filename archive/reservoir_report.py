import _path  # noqa: F401
_root = _path._root
_archive = str(_root / "archive") if _root is not None else ""
"""Сводка результатов reservoir-эксперимента: markdown-таблица + рисунок."""
import glob, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
COLS = [("mg", "h1", "MG h=1"), ("mg", "h10", "MG h=10"), ("mg", "h50", "MG h=50"),
        ("lorenz", "h1", "Lorenz h=1"), ("lorenz", "h10", "Lorenz h=10"), ("mc", "MC", "MC ↑")]

def load(tag):
    data = {}
    for f in glob.glob(os.path.join(RES, f"{tag}_*.json")):
        r = json.load(open(f)); t = r["task"]
        for run in r["runs"]:
            key = run["cond"] if f.endswith("_esn.json") or tag == "main" else f"{run['cond']} [{tag}]"
            d = data.setdefault(key, {})
            if t == "mc": d.setdefault(("mc", "MC"), []).append(run["MC"])
            else:
                for nm, v in zip(r["targets"], run["nrmse"]): d.setdefault((t, nm), []).append(v)
            d.setdefault(("meta", t), []).append(run)
    return data

def report(argv):
    argv = argv if argv else ["main"]
    if "--fig" in argv:
        i = argv.index("--fig"); tags, ftags = argv[:i], argv[i + 1:]
    else: tags, ftags = argv, argv
    data = {}
    for t in tags: data.update(load(t))
    order = sorted(data, key=lambda k: (0 if k.startswith("FL") else 1 if k.startswith("ESN") else 2, k))
    lines = ["| режим | " + " | ".join(c[2] for c in COLS) + " | ⟨λ⟩ | ⟨c⟩ | доля хаоса |",
             "|---|" + "---|" * (len(COLS) + 3)]
    for k in order:
        d = data[k]; cells = []
        for t, nm, _ in COLS:
            v = d.get((t, nm))
            cells.append("—" if not v else (f"{np.mean(v):.3g} ± {np.std(v):.2g}" if len(v) > 1 else f"{v[0]:.3g}"))
        meta = d.get(("meta", "mg"), []) or d.get(("meta", "mc"), [])
        if meta and "lam_mean" in meta[0]:
            ex = [f"{np.mean([m['lam_mean'] for m in meta]):+.3f}", f"{np.mean([m['c_mean'] for m in meta]):.2f}",
                  f"{np.mean([m['chaos'] for m in meta]):.2f}"]
        else: ex = ["—"] * 3
        lines.append(f"| {k} | " + " | ".join(cells) + " | " + " | ".join(ex) + " |")
    md = "\n".join(lines); print(md)
    open(os.path.join(HERE, "reservoir_results.md"), "w").write(
        "# Результаты reservoir computing (NRMSE на test, mean ± std по сидам; MC — сумма r² по задержкам 1..80)\n\n" + md + "\n")
    # ---------- figure (только выбранные теги)
    data = {}
    for t in ftags: data.update(load(t))
    order = sorted(data, key=lambda k: (0 if k.startswith("FL") else 1 if k.startswith("ESN") else 2, k))
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 6)
    pal = plt.get_cmap("tab20")
    for j, (t, nm, lab) in enumerate(COLS):
        ax = fig.add_subplot(gs[0, j])
        rows = [k for k in order]                              # одинаковые строки во всех панелях
        ok = lambda k: (t, nm) in data[k] and not (t == "mc" and k.startswith("Linear"))
        m = [np.mean(data[k][(t, nm)]) if ok(k) else 0 for k in rows]; s = [np.std(data[k][(t, nm)]) if ok(k) else 0 for k in rows]
        ax.barh(range(len(rows)), m, xerr=s, color=[pal(i % 20) for i in range(len(rows))])
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows if j == 0 else [], fontsize=7)
        ax.invert_yaxis(); ax.set_title(lab + (" (NRMSE ↓)" if t != "mc" else "\n(лин. окно = 80 тривиально, не показано)"), fontsize=9)
        if t != "mc": ax.set_xscale("log")
        ax.grid(axis="x", alpha=.3)
    # traces
    def best_fl(t, nm):
        ks = [k for k in order if k.startswith("FL") and (t, nm) in data[k]]
        f = max if t == "mc" else min
        return f(ks, key=lambda k: np.mean(data[k][(t, nm)]))
    for j, (t, nm, ti) in enumerate([("mg", "h10", 0), ("lorenz", "h1", 0), ("mc", None, 0)]):
        ax = fig.add_subplot(gs[1, 2 * j: 2 * j + 2])
        if t == "mc":
            for k in order:
                runs = data[k].get(("meta", "mc"))
                if runs and k in ("ESN", best_fl("mc", "MC"), "FL-adapt", "FL-c5.0 (хаос)", "FL-c2.5 (порядок)") :
                    ax.plot(range(1, 81), np.mean([r["mc_curve"] for r in runs], 0), label=k)
            ax.set_xlabel("задержка k"); ax.set_ylabel("r²(k)"); ax.set_title("Кривая памяти"); ax.legend(fontsize=8)
            continue
        col = {"mg": ["h1", "h10", "h50"], "lorenz": ["h1", "h10"]}[t].index(nm)
        kf = best_fl(t, nm)
        for k, st in [(kf, "-"), ("ESN", "--")]:
            r = data[k][("meta", t)][0]
            pt = np.array(r["pred_test"])[:, col]; yt = np.array(r["true_test"])[:, col]
            if k == kf: ax.plot(yt[:250], c="k", lw=2, alpha=.4, label="истина")
            ax.plot(pt[:250], st, lw=1.2, label=f"{k}")
        ax.set_title(f"{'Mackey-Glass, прогноз на 10' if t=='mg' else 'Lorenz x, прогноз на 1'} шаг(ов), test"); ax.legend(fontsize=8)
    fig.suptitle("Feigenbaum-Lenia как резервуар vs ESN", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "reservoir_results.png"), dpi=110)
    print("saved")

if __name__ == "__main__":
    report(sys.argv[1:])
