import _path  # noqa: F401
"""Фигуры, таблица и анимация для больших прогонов глайдеров."""
import pickle, glob, sys, os, json, subprocess, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import fl_glider as FG
HERE = os.path.dirname(os.path.abspath(__file__))

def events(tracks, L, R, nfr):
    """Слияния/распады: трек кончается (рождается) в ≤1.5R от другого живого трека."""
    pos = {}
    for k, t in enumerate(tracks):
        for j, c in enumerate(t["cens"]): pos.setdefault(t["t0"] + j, []).append((k, np.asarray(c)))
    merge = split = 0
    for k, t in enumerate(tracks):
        if t["t1"] < nfr - 1 and len(t["cens"]) > 2:
            end = np.asarray(t["cens"][-1])
            if any(kk != k and FG.pdist(end, c, L) < 1.5 * R for kk, c in pos.get(t["t1"] + 1, [])): merge += 1
        if t["t0"] > 0 and len(t["cens"]) > 2:
            st = np.asarray(t["cens"][0])
            if any(kk != k and FG.pdist(st, c, L) < 1.5 * R for kk, c in pos.get(t["t0"] - 1, [])): split += 1
    return merge, split

def unwrap(C, L):
    C = np.asarray(C); d = np.array([FG.pdelta(C[i], C[i + 1], L) for i in range(len(C) - 1)])
    return np.vstack([C[:1], C[:1] + np.cumsum(d, 0)]) if len(C) > 1 else C

def analyze_run(o, N, L, steps, every):
    c = o["cand"]; R = c["R"]; tr = o["tracks_all"] or o["tracks"]; nfr = len(o["ncomp"])
    V = L ** N
    longt = [t for t in tr if t["life"] >= 0.25 * steps and t["size_max"] < 0.1 * V]
    gl = [t for t in longt if t["net"] >= 1.5 * R and t["straight"] > 0.6 and t["mass_cv"] < 0.3 and t["corr"] > 0.6]
    m, s = events(tr, L, R, nfr)
    vel = np.array([t["vel"] for t in gl]) if gl else np.zeros((0, N))
    dirs = vel / (np.linalg.norm(vel, axis=1, keepdims=True) + 1e-12) if len(vel) else vel
    pol = float(np.linalg.norm(dirs.mean(0))) if len(dirs) else np.nan     # 1 = все в одну сторону
    row = dict(tag=c.get("tag", str(c["seed"])), N=N, L=L, cells=V, steps=steps, ncomp_end=o["ncomp_end"], n_long=len(longt), n_glider=len(gl),
               speed_med=float(np.median([t["speed"] for t in gl])) if gl else 0.0,
               speed_iqr=[float(np.percentile([t["speed"] for t in gl], q)) for q in (25, 75)] if gl else [0, 0],
               straight=float(np.mean([t["straight"] for t in gl])) if gl else 0.0, corr=float(np.mean([t["corr"] for t in gl])) if gl else 0.0,
               life_med=float(np.median([t["life"] for t in gl])) if gl else 0.0, net_max=float(max([t["net"] for t in gl], default=0)),
               genome_offset=float(np.nanmean([t["genome_offset"] for t in gl])) if gl else np.nan, polarization=pol,
               merges=m, splits=s, mass_end_frac=o["mass"][-1] / V)
    return row, gl, longt

def figure(o, N, L, steps, every, gl, longt, out_png, title):
    fig = plt.figure(figsize=(17, 8.5))
    P = o["mproj"].astype(np.float32); T = len(P)
    for j, t in enumerate(np.linspace(0, T - 1, 5).astype(int)):
        ax = fig.add_subplot(2, 5, j + 1); ax.imshow(P[t], cmap="inferno", vmin=0, vmax=max(P[t].max(), 1e-3))
        for g in gl:
            k = t - g["t0"]
            if 0 <= k < len(g["cens"]): ax.plot(g["cens"][k][1], g["cens"][k][0], "c+", ms=5)
        ax.set_title(f"среднее по осям 3..{N}, шаг {t * every}", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(2, 5, 6)
    for g in (gl or longt)[:60]:
        U = unwrap(g["cens"], L); ax.plot(U[:, 1], U[:, 0], lw=1); ax.plot(U[-1, 1], U[-1, 0], "k.", ms=3)
    ax.set_title(f"траектории (оси 1–2, развёрнуты), {len(gl)} глайдеров", fontsize=8); ax.set_aspect("equal")
    ax = fig.add_subplot(2, 5, 7)
    sp = [g["speed"] for g in gl] or [t["speed"] for t in longt]
    ax.hist(sp, bins=20, color="tab:orange"); ax.set_xlabel("скорость, клеток/шаг"); ax.set_title("гистограмма скоростей", fontsize=8)
    ax = fig.add_subplot(2, 5, 8)
    if gl:
        V = np.array([g["vel"] for g in gl]); D = V / np.linalg.norm(V, axis=1, keepdims=True)
        ax.boxplot([D[:, k] for k in range(N)]); ax.set_xticklabels([f"x{k + 1}" for k in range(N)])
    ax.set_title("направления (компоненты ед. вектора)", fontsize=8)
    ax = fig.add_subplot(2, 5, 9)
    lifes = [t["life"] for t in longt]; ax.hist(lifes, bins=20, color="tab:green"); ax.set_xlabel("время жизни трека, шагов"); ax.set_title("время жизни (долгие треки)", fontsize=8)
    ax = fig.add_subplot(2, 5, 10)
    # срез 2D через центр лучшего глайдера (по прореженному объёму)
    if gl and o.get("vols") is not None and len(o["vols"]):
        g = max(gl, key=lambda g: g["net"]); Vs = o["vols"].astype(np.float32); ds = L // Vs.shape[1]
        fr_per_vol = (len(o["ncomp"]) - 1) / max(len(Vs) - 1, 1)
        kv = min(len(Vs) - 1, int(round((g["t0"] + len(g["cens"]) // 2) / fr_per_vol)))
        cen = np.asarray(g["cens"][min(len(g["cens"]) - 1, int(round(kv * fr_per_vol)) - g["t0"])]) / ds
        idx = tuple(int(round(x)) % Vs.shape[1] for x in cen[2:])
        ax.imshow(Vs[kv][(slice(None), slice(None)) + idx], cmap="inferno", vmin=0, vmax=1)
        ax.set_title(f"2D‑срез через глайдер (½ разрешения), шаг≈{int(kv * fr_per_vol) * every}", fontsize=8)
    else:
        C = o["cproj"].astype(np.float32); ax.imshow(C[-1], cmap="viridis"); ax.set_title("геном c (среднее), конец", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(out_png, dpi=85); plt.close(fig)

def animation(o, out_mp4, every, fps=15):
    import matplotlib.animation as A
    use_slice = o.get("slices") is not None and len(o["slices"])
    P = (o["slices"] if use_slice else o["mproj"]).astype(np.float32); C = o["cproj"].astype(np.float32)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    vm = np.percentile(P, 99.5) + 1e-6
    im1 = axs[0].imshow(P[0], cmap="inferno", vmin=0, vmax=vm); im2 = axs[1].imshow(C[0], cmap="viridis", vmin=2.5, vmax=5.0)
    axs[0].set_title("масса A: 2D‑срез (x1,x2) через центр" if use_slice else "масса A (среднее по 3..N)", fontsize=9); axs[1].set_title("геном c (среднее по 3..N)", fontsize=9)
    for a in axs: a.set_xticks([]); a.set_yticks([])
    tx = fig.suptitle("")
    def upd(t):
        im1.set_data(P[t]); im2.set_data(C[t]); tx.set_text(f"шаг {t * every}"); return im1, im2
    an = A.FuncAnimation(fig, upd, frames=len(P), interval=1000 / fps)
    an.save(out_mp4, writer=A.FFMpegWriter(fps=fps, bitrate=600)); plt.close(fig)

if __name__ == "__main__":
    prefix = sys.argv[1]; rows = []
    for f in sorted(glob.glob(f"results/glider/{prefix}_*.pkl")):
        r = pickle.load(open(f, "rb"))
        for job in r["out"]:
            for o in job["out"]:
                row, gl, longt = analyze_run(o, job["N"], job["L"], job["steps"], job["every"])
                row["gpu"] = r["gpu"].split(",")[0]; row["t_gpu"] = job["t_gpu"]; row["t_total"] = job["t_total"]
                rows.append(row); tag = row["tag"]
                figure(o, job["N"], job["L"], job["steps"], job["every"], gl, longt, os.path.join(HERE, f"glider_{tag}.png"),
                       f"{tag}: {job['N']}D {job['L']}^{job['N']} = {job['L'] ** job['N']:,} клеток, {job['steps']} шагов; " + json.dumps({k: o['cand'][k] for k in ('mode', 'mu', 'sigma', 'R', 'asym', 'chi', 'beta', 'lag', 'tau_c') if k in o['cand']}))
                if len(sys.argv) > 2 and tag in sys.argv[2].split(","):
                    animation(o, os.path.join(HERE, f"glider_{tag}.mp4"), job["every"])
                print(json.dumps(row))
    json.dump(rows, open(f"results/glider/{prefix}_table.json", "w"), indent=1)
