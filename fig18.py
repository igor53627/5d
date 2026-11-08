"""Part 18 figures/table/mp4 from results/eco18/main18.pkl and b18m logs."""
import pickle, re, glob, os, subprocess, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import fl_season as Z, os
TAG = os.environ.get("TAG18", "18"); LBOX = int(os.environ.get("LBOX", "56")); MAIN = os.environ.get("OUT18", "results/eco18/main18.pkl"); RAWGLOB = os.environ.get("RAW18", "results/eco18/b18m_*.pkl")
M = pickle.load(open(MAIN, "rb")); cond = lambda t: re.sub(r"_s\d+$", "", t)
CONDS = [c for c in ["mem", "nomem", "cue", "nobrain"] if any(cond(t) == c for t in M)]; COL = dict(mem="tab:red", nomem="tab:orange", cue="tab:green", nobrain="tab:blue")
g = lambda r, *ks: (lambda x: x if np.isscalar(x) else np.nan)(__import__("functools").reduce(lambda a, k: a.get(k, np.nan) if isinstance(a, dict) else np.nan, ks, r))
def lagm(r): l = r["lags"]; return np.nanmean(l) if np.isfinite(l).any() else np.nan
def fPs(r):   # mean poison fraction in seasons after the first (re-adaptation needed)
    s = r["seasons"]; return np.nanmean(s[1:, 2]) if len(s) > 1 else np.nan
def ext(r):
    ts = r["ts"]; z = np.where((ts[:, 2] == 0) | (ts[:, 3] < 0.05))[0]; return ts[z[0], 0] if len(z) else np.inf
def mass_at(r, t): ts = r["ts"]; return ts[np.searchsorted(ts[:, 0], t), 3]
def sea(r, sg, col):
    s = r["seasons"]; s = s[(s[:, 1] == sg) & (s[:, 4] > 0.05)]; return np.nanmean(s[:, col]) if len(s) else np.nan
def corr_alive(r, a, b):
    ts = r["ts"]; ok = (ts[:, 2] > 0) & np.isfinite(ts[:, a]); return np.corrcoef(ts[ok, a], ts[ok, b])[0, 1] if ok.sum() > 5 else np.nan
def dg_corr(r):
    ts = r["ts"]; ok = (ts[:, 2] > 0) & np.isfinite(ts[:, 5]); return np.corrcoef(ts[ok, 5] - ts[ok, 6], ts[ok, 1])[0, 1] if ok.sum() > 5 else np.nan
def lagm(r):
    l = r["lags"]; return np.nanmean(l) if np.isfinite(l).any() else np.nan
cols = [("ext", "вымирание (шаг; ∞ = дожил)", 0, ext), ("m4", "масса 4k", 1, lambda r: mass_at(r, 4000)), ("m8", "масса 8k", 2, lambda r: mass_at(r, 8000)), ("m16", "масса 16k", 2, lambda r: mass_at(r, 16000)), ("mend", "масса в конце", 1, lambda r: r["ts"][-1, 3]),
        ("fPA", "доля яда, сезоны A-хор.", 2, lambda r: sea(r, 1, 2)), ("fPB", "доля яда, сезоны B-хор.", 2, lambda r: sea(r, -1, 2)),
        ("aA", "σ·Δg, сезоны A", 2, lambda r: sea(r, 1, 3)), ("aB", "σ·Δg, сезоны B", 2, lambda r: sea(r, -1, 3)),
        ("aBe", "σ·Δg B: 1-я четверть", 2, lambda r: sea(r, -1, 5) if r["seasons"].shape[1] > 5 else np.nan), ("aBl", "σ·Δg B: последняя четверть", 2, lambda r: sea(r, -1, 6) if r["seasons"].shape[1] > 5 else np.nan),
        ("fBe", "яд B: 1-я четв.", 2, lambda r: sea(r, -1, 7) if r["seasons"].shape[1] > 5 else np.nan), ("fBl", "яд B: посл. четв.", 2, lambda r: sea(r, -1, 8) if r["seasons"].shape[1] > 5 else np.nan),
        ("aA2", "σ·Δg A (2-й раз и далее)", 2, lambda r: (lambda s: np.nanmean(s[2:, 3][s[2:, 1] > 0]) if len(s) > 2 and (s[2:, 1] > 0).any() else np.nan)(r["seasons"])), ("cdg", "corr(g_A−g_B, σ)", 2, dg_corr),
        ("lag", "лаг после смены", 0, lagm), ("ch", "corr(h, σ)", 2, lambda r: corr_alive(r, 7, 1)), ("me", "|Δ поведения| при h→0", 3, lambda r: r.get("mem_effect", np.nan)),
        ("nl", "1−R² (лин.)", 2, lambda r: g(r, "nonlin", "lin")), ("nli", "1−R² (с взаимод.)", 2, lambda r: g(r, "nonlin", "inter")), ("pS", "Price S σ·Δg", 3, lambda r: r["price"]["adapt"]["S"]),
        ("p0", "⟨p0⟩ (плато)", 2, lambda r: r["p0"])]
L = ["| условие | " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * (len(cols) + 1)]
fmt = lambda x, d: "∞" if x == np.inf else ("—" if not np.isfinite(x) else f"{x:.{d}f}")
for c in CONDS:
    R = [v for t, v in sorted(M.items()) if cond(t) == c]
    def ms(f, d):
        x = np.array([f(r) for r in R], float)
        if np.isinf(x).any(): return " / ".join(fmt(u, d) for u in x)
        x = x[np.isfinite(x)]
        return "—" if not len(x) else (f"{x.mean():.{d}f} ± {x.std(ddof=1):.{d}f}" if len(x) > 1 else f"{x[0]:.{d}f}")
    L.append(f"| **{c}** (n={len(R)}) | " + " | ".join(ms(f, d) for _, _, d, f in cols) + " |")
for t, r in sorted(M.items()): L.append(f"| {t}{' (partial)' if r['partial'] else ''} | " + " | ".join(fmt(f(r), d) for _, _, d, f in cols) + " |")
L += ["", "Welch t-тест по сидам:"]
for a, b in (("mem", "nomem"), ("mem", "nobrain"), ("nomem", "nobrain"), ("cue", "mem")):
    if a not in CONDS or b not in CONDS: continue
    s = []
    for k, nm, _, f in cols:
        x = [f(v) for t, v in M.items() if cond(t) == a]; y = [f(v) for t, v in M.items() if cond(t) == b]
        x = [u for u in x if np.isfinite(u)]; y = [u for u in y if np.isfinite(u)]
        if len(x) > 1 and len(y) > 1: s.append(f"{nm} p={ttest_ind(x, y, equal_var=False).pvalue:.2g}")
    L.append(f"- {a} vs {b}: " + "; ".join(s))
L += ["", "Повреждение памяти (σ·Δg: как есть / h→0 / h→−h), h по сезонам (σ>0, σ<0), Price p0 (S / Δ / передача):"]
for t, r in sorted(M.items()):
    L.append(f"- {t}: " + (" / ".join(f"{x:+.3f}" for x in r["adapt_lesion"]) if "adapt_lesion" in r else "—") + "; h: " + (", ".join(f"{x:+.2f}" for x in r["h_season"]) if "h_season" in r else "—")
             + f"; p0: {r['price']['p0']['S']:+.3f} / {r['price']['p0']['dz']:+.3f} / {r['price']['p0']['T']:+.3f}; лаги: {np.round(r['lags']).tolist()}")
L += ["", "По сезонам (начало, σ, доля яда, σ·Δg, средняя масса):"]
L[-1] = "По сезонам (начало, какой корм хороший, доля яда, σ·Δg, средняя масса; [σ·Δg 1-я → посл. четверть; доля яда 1-я → посл.]):"
for t, r in sorted(M.items()): L.append(f"- {t}: " + "; ".join(f"{s[0]:.0f} {'A' if s[1] > 0 else 'B'}-хор: {s[2]:.2f}, {s[3]:+.2f}, {s[4]:.1f}" + (f" [{s[5]:+.2f}→{s[6]:+.2f}; {s[7]:.2f}→{s[8]:.2f}]" if len(s) > 5 else "") for s in r["seasons"]))
open(f"results/part{TAG}_table.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
# --- dynamics figure ---
fig, ax = plt.subplots(2, 3, figsize=(16, 8.5)); ax = ax.ravel()
for t, r in sorted(M.items()):
    c = cond(t); ts = r["ts"]; kw = dict(color=COL[c], lw=1, alpha=.8, label=c if t.endswith(sorted(M)[0][-4:]) else None)
    ax[0].plot(ts[:, 0], ts[:, 2], **kw); ax[1].plot(ts[:, 0], ts[:, 3], **kw)
    sm = lambda y: np.convolve(np.nan_to_num(y), np.ones(3) / 3, "same")
    ax[2].plot(ts[:, 0], sm(ts[:, 9]), **kw); ax[3].plot(ts[:, 0], ts[:, 5] - ts[:, 6], **kw); ax[4].plot(ts[:, 0], ts[:, 7], **kw); ax[5].plot(ts[:, 0], ts[:, 10], **kw)
ts0 = next(iter(M.values()))["ts"]
for a in ax:
    for k in range(len(ts0) - 1):
        if ts0[k, 1] < 0: a.axvspan(ts0[k, 0], ts0[k + 1, 0], color="0.85", lw=0, zorder=0)
for a, tt in zip(ax, ["число особей n", "общая масса (M1)", "доля яда в рационе", "g_A − g_B (реализованное)", "память ⟨h⟩", "σ·(g_A − g_B): >0 = правильный выбор"]): a.set_title(tt); a.set_xlabel("шаг"); a.grid(alpha=.3)
ax[0].legend(); fig.suptitle(f"Часть {TAG}: сезоны, 4D {LBOX}⁴ (серый фон — сезон «B хороший»)"); fig.tight_layout(); fig.savefig(f"glider_part{TAG}_dynamics.png", dpi=80)
# --- brains figure: Δg vs reward R in A / B patch, h = ±0.5 ---
fig, ax = plt.subplots(1, len(CONDS), figsize=(4.5 * len(CONDS), 4), squeeze=False); ax = ax[0]
for a, c in zip(ax, CONDS):
    for t, r in sorted(M.items()):
        if cond(t) != c or "curves" not in r: continue
        for (patch, hv), (xs, y) in r["curves"].items():
            a.plot(xs, y, color="tab:green" if patch == "A" else "tab:purple", ls="-" if hv > 0 else "--", lw=1, alpha=.7)
    a.axhline(0, color="k", lw=.5); a.set_title(f"{c}: g_A−g_B от награды R\n(зелёный — в пятне A, фиол. — в пятне B; сплошн. h=+.5, пунктир h=−.5)", fontsize=9); a.set_xlabel("R"); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"glider_part{TAG}_brains.png", dpi=80)
# --- mp4: seed of first mem run, all conditions ---
seed = sorted(t for t in M if cond(t) == "mem")[0].split("_")[-1]; raw = {}
for f in sorted(glob.glob(RAWGLOB)):
    for t, r in pickle.load(open(f, "rb"))["out"][0].items():
        if isinstance(r, dict) and r.get("runs") and t.endswith(seed): raw[cond(t)] = (r["runs"][0]["proj"], r["runs"][0]["eco"])
raw = {c: raw[c] for c in CONDS if c in raw}; nf = min(len(v[0]) for v in raw.values()); vmx = max(float(np.percentile(np.asarray(v[0], float), 99.7)) for v in raw.values())
os.makedirs("/tmp/f18", exist_ok=True); [os.remove(p) for p in glob.glob("/tmp/f18/*.png")]
for k in range(nf):
    fig, ax = plt.subplots(1, len(raw), figsize=(3.6 * len(raw), 4)); t = k * float(os.environ.get("PROJ18", "200"))
    for a, (c, (pj, e)) in zip(np.atleast_1d(ax), raw.items()):
        P = dict(Z.BDEF, **{kk: e[kk] for kk in Z.BKEYS if kk in e}); cc = Z.pattern((LBOX, LBOX), P, t, np); sg = float(Z.season(t, P["Ts"], np))
        a.imshow(np.maximum(cc, 0).T, origin="lower", cmap="Greens", alpha=.5, vmin=0, vmax=1); a.imshow(np.ma.masked_less(np.maximum(-cc, 0).T, .05), origin="lower", cmap="Purples", alpha=.5, vmin=0, vmax=1)
        a.imshow(np.ma.masked_less(pj[k].astype(float).T, .03 * vmx), origin="lower", cmap="Reds", vmin=0, vmax=vmx, alpha=.9)
        a.set_title(f"{c} t={t:.0f}  хороший: {'A (зел.)' if sg > 0 else 'B (фиол.)'}", fontsize=9); a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(f"/tmp/f18/{k:04d}.png", dpi=60); plt.close(fig)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "12", "-i", "/tmp/f18/%04d.png", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "30", f"glider_part{TAG}_{seed}.mp4"], check=True)
print("frames", nf, "seed", seed)
