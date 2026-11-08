"""Таблица и рисунок бенчмарка."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results")
src = {"numpy CPU (1 поток FFT)": json.load(open(f"{R}/bench_numpy_local.json")),
       "JAX CPU (8 vCPU)": json.load(open(f"{R}/bench_jaxcpu_local.json")),
       "JAX L4": json.load(open(f"{R}/bench_L4.json"))["out"], "JAX B200": json.load(open(f"{R}/bench_B200.json"))["out"]}
key = lambda r: (r["N"], r["L"], r["B"], r["rule"])
cases = []
for v in src.values():
    for r in v:
        if key(r) not in cases: cases.append(key(r))
tab = {k: {nm: next((r for r in v if key(r) == k and "steps_per_s" in r), None) for nm, v in src.items()} for k in cases}
lines = ["| решётка | батч | правило | клеток | " + " | ".join(src) + " | ускорение B200/numpy |", "|" + "---|" * (len(src) + 5)]
for k, d in tab.items():
    N, L, B, rule = k
    vals = [f"{d[nm]['steps_per_s']:.3g}" if d[nm] else "—" for nm in src]
    sp = (f"×{d['JAX B200']['steps_per_s'] / d['numpy CPU (1 поток FFT)']['steps_per_s']:.0f}" if d["JAX B200"] and d["numpy CPU (1 поток FFT)"] else "—")
    lines.append(f"| {N}D {L}^{N} | {B} | {rule} | {B * L ** N:,} | " + " | ".join(vals) + f" | {sp} |")
md = "Шагов/с (один шаг = вся решётка или весь батч). own = исходное правило (касательный λ), cond = с двойником (условный λc, ~2× работы).\n\n" + "\n".join(lines) + "\n"
open(os.path.join(HERE, "gpu_bench.md"), "w").write(md); print(md)
fig, ax = plt.subplots(figsize=(11, 4.5))
lab = [f"{N}D {L}^{N}\nB={B} {rule}" for N, L, B, rule in cases]; x = np.arange(len(cases)); w = 0.2
for i, nm in enumerate(src):
    ax.bar(x + (i - 1.5) * w, [tab[k][nm]["cell_steps_per_s"] if tab[k][nm] else 0 for k in cases], w, label=nm)
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7); ax.set_ylabel("клетко‑шагов/с"); ax.legend(fontsize=8)
ax.set_title("Feigenbaum‑Lenia: пропускная способность движка"); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "gpu_bench.png"), dpi=110)
