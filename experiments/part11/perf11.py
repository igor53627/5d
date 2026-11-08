import sys; _R = __import__("pathlib").Path(__file__).resolve().parents[2]; [sys.path.insert(0, str(p)) for p in [_R, _R / "src"] + list((_R / "experiments").glob("part*"))]
"""Часть 11: профиль по ядрам (jax.profiler → perfetto), скорость вариантов fl_eco3, ошибка поля на коротком времени, e2e логирование, валидация."""
import time, glob, gzip, json, os, shutil, numpy as np

VARS = {"base": {}, "coarse": dict(coarse=True), "fnoise": dict(fnoise=True), "m5": dict(m=5), "fn_m5": dict(fnoise=True, m=5),
        "all5": dict(coarse=True, fnoise=True, m=5), "all10": dict(coarse=True, fnoise=True, m=10)}

def _state(L, N, c, eco, opts, B=1, nblob=0, seed=0):
    import jax, jax.numpy as jnp, fl_glider as FG, fl_eco3 as E, perf_eco as PE
    s, P, KA, Ko, _ = PE.make(L, N, B, c, eco, nblob=nblob, seed=seed)
    return s, P, KA, Ko, E.kernels(L, N, **opts)

def _sync(x):
    import jax; return jax.block_until_ready(x)

def speed(c, eco, L, N, variants, n=50):
    import jax, fl_eco3 as E
    out = {}
    for vn in variants:
        o = VARS[vn]
        try:
            s, P, KA, Ko, KX = _state(L, N, c, eco, o)
            f = lambda st: E.run_chunk(st, P, KA, Ko, KX, n, True, **o)
            t0 = time.perf_counter(); r = _sync(f(s)); tc = time.perf_counter() - t0; ts = []
            for _ in range(3):
                t0 = time.perf_counter(); r = _sync(f(r)); ts.append(time.perf_counter() - t0)
            st = jax.devices()[0].memory_stats() or {}
            out[vn] = dict(steps_s=n / float(np.median(ts)), ms_step=1000 * float(np.median(ts)) / n, t_compile=tc, peak_GB=st.get("peak_bytes_in_use", 0) / 1e9)
            del s, r, KX
        except Exception as e:
            out[vn] = dict(err=repr(e)[:200])
        jax.clear_caches()
    return out

def profile(c, eco, L, N, vn, n=10, tag=""):
    import jax, fl_eco3 as E
    o = VARS[vn]; s, P, KA, Ko, KX = _state(L, N, c, eco, o)
    f = lambda st: E.run_chunk(st, P, KA, Ko, KX, n, True, **o)
    r = _sync(f(s)); d = f"/tmp/prof_{tag}"
    shutil.rmtree(d, ignore_errors=True)
    with jax.profiler.trace(d, create_perfetto_trace=True):
        r = _sync(f(r))
    fs = glob.glob(f"{d}/**/*.trace.json.gz", recursive=True)
    if not fs: return dict(err="no trace", files=glob.glob(f"{d}/**/*", recursive=True)[:20])
    try:
        os.makedirs("/vol/prof", exist_ok=True); shutil.copy(fs[0], f"/vol/prof/{tag}.trace.json.gz")
    except Exception: pass
    ev = json.load(gzip.open(fs[0]))["traceEvents"]
    pn = {e["pid"]: e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"}
    gp = {p for p, nm in pn.items() if "GPU" in nm or "gpu" in nm}
    agg = {}; tot = 0.0
    for e in ev:
        if e.get("ph") == "X" and e.get("pid") in gp and "dur" in e:
            nm = e["name"]; a = agg.setdefault(nm, [0.0, 0]); a[0] += e["dur"]; a[1] += 1; tot += e["dur"]
    top = sorted(agg.items(), key=lambda kv: -kv[1][0])
    return dict(gpu_procs=[pn[p] for p in gp], total_us=tot, us_per_step=tot / n, n_kernels=sum(v[1] for v in agg.values()) / n,
                top=[(k, v[0] / n, v[1] / n) for k, v in top[:60]], n_names=len(agg))

def micro(L, N):
    import jax, jax.numpy as jnp, perf_eco as PE
    sh = (L,) * N; x = jax.random.normal(jax.random.PRNGKey(0), sh); xc = (x + 1j * x).astype(jnp.complex64)
    r = dict(t_rfftn=PE._t(jax.jit(jnp.fft.rfftn), x), t_irfftn=PE._t(jax.jit(lambda X: jnp.fft.irfftn(X, s=sh)), jnp.fft.rfftn(x)),
             t_fftn_c2c=PE._t(jax.jit(jnp.fft.fftn), xc), t_copy=PE._t(jax.jit(lambda a: a * 1.0001 + 1.0), x),
             t_pool2=PE._t(jax.jit(lambda a: a.reshape(sum(((L // 2, 2) for _ in range(N)), ())).mean(axis=tuple(range(1, 2 * N, 2)))), x))
    import fl_eco3 as E
    r["t_up2"] = PE._t(jax.jit(E.up2), x[tuple(slice(0, L // 2) for _ in range(N))])
    n = x.size; r["bw_copy_TBs"] = 8 * n / r["t_copy"] / 1e12
    return r

def field_err(c, eco, N, L, seed, variants, steps=(200, 1000)):
    """От одного реального солитона (+ eco-состояние): ошибка поля вариантов относительно base."""
    import jax, jax.numpy as jnp, fl_glider as FG, fl_eco3 as E, glider_coll as GC
    st, v = GC.soliton(c, N, L, seed, pre=600)
    cc = dict(c, **eco); U1 = E.U1_of(st["A"], L, N)
    s0 = E.make_state(st, dict(cc, U1=U1), K=2, p0=np.array([0.3, 0.0]), seed=seed)
    res = {}
    for vn in ["base"] + variants:
        o = VARS[vn]; U1o = E.U1_of(st["A"], L, N, **o)
        P = jax.tree.map(lambda a: jnp.stack([a]), E.params(dict(cc, U1=U1o)))
        s = jax.tree.map(lambda a: jnp.stack([jnp.asarray(a)]), s0)
        KA = jnp.stack([jnp.asarray(FG.ring_kernel_np(L, N, c["R"], tuple(c["peaks"])))]); Ko = jnp.asarray(FG.ring_kernel_np(L, N, 3.0)); KX = E.kernels(L, N, **o)
        snaps = {}; t = 0
        for T in steps:
            s = E.run_chunk(s, P, KA, Ko, KX, T - t, True, **o); t = T; snaps[T] = (np.asarray(s["A"][0]), np.asarray(s["Q"][0, 0]))
        res[vn] = dict(U1=U1o, snaps=snaps)
    out = {}
    for vn in variants:
        out[vn] = {T: dict(relL2_A=float(np.linalg.norm(res[vn]["snaps"][T][0] - res["base"]["snaps"][T][0]) / np.linalg.norm(res["base"]["snaps"][T][0])),
                           mass_rel=float(res[vn]["snaps"][T][0].sum() / res["base"]["snaps"][T][0].sum() - 1),
                           relL2_Q=float(np.linalg.norm(res[vn]["snaps"][T][1] - res["base"]["snaps"][T][1]) / (np.linalg.norm(res["base"]["snaps"][T][1]) + 1e-12)))
                   for T in steps}
        out[vn]["U1"] = res[vn]["U1"]
    out["U1_base"] = res["base"]["U1"]; out["mass_base"] = {T: float(res["base"]["snaps"][T][0].sum()) for T in steps}
    return out

def e2e_log(c, eco, L, N, nblob=500, chunks=6, every=200, proj_every=400):
    """Старое (синхронно + проекция через хост) против нового (асинхронно + проекция на GPU) на состоянии с ~сотнями объектов."""
    import jax, fl_eco2 as E, glider_eco as GE
    s0, P, KA, Ko, KX = _state(L, N, c, eco, {}, nblob=nblob)
    KX = E.kernels(L, N)
    adv = lambda s: E.run_chunk(s, P, KA, Ko, KX, every, True)
    _sync(adv(s0)); out = {}
    for mode in ("old", "new"):
        s = s0; t0 = time.perf_counter(); nob = []; s_next = None
        for k in range(chunks + 1):
            if k: s = s_next
            Ap = np.asarray(GE._pool(s["A"], N, L)); Qp = np.asarray(GE._pool(s["Q"], N, L))
            if (k * every) % proj_every == 0:
                pj = np.asarray(s["A"]).mean(axis=tuple(range(3, N + 1))) if mode == "old" else np.asarray(s["A"].mean(axis=tuple(range(3, N + 1))))
            if mode == "new" and k < chunks: s_next = adv(s)
            ob = GE._objects(Ap[0], Qp[0], 80.0, 2, N); nob.append(len(ob))
            if mode == "old" and k < chunks: s_next = _sync(adv(s))
        _sync(s); dt = time.perf_counter() - t0
        out[mode] = dict(wall=dt, steps_s_e2e=chunks * every / dt, n_obj=nob)
    t0 = time.perf_counter(); s = s0
    for k in range(chunks): s = adv(s)
    _sync(s); out["gpu_only"] = dict(wall=time.perf_counter() - t0, steps_s=chunks * every / (time.perf_counter() - t0))
    return out

def run_perf11(c, eco, jobs):
    import jax
    res = dict(gpu=str(jax.devices()[0].device_kind))
    for j in jobs:
        j = dict(j); k = j.pop("job"); tag = j.pop("tag", k)
        if k == "profile": j["tag"] = tag
        try:
            t0 = time.time(); r = (globals()[k](c, eco, **j) if k != "micro" else micro(**j)); res[tag] = r; res[tag + "_wall"] = time.time() - t0
        except Exception as e:
            res[tag] = dict(err=repr(e)[:300])
        jax.clear_caches()
        print(tag, "done", flush=True)
    return res

def run_val(c, eco, seed, variants, **kw):
    import glider_eco as GE
    out = {}
    for vn in variants:
        t0 = time.time(); o = GE.run_eco(c, [eco], seed=seed, model="eco3", opts=VARS[vn], **kw); o["wall"] = time.time() - t0; out[vn] = o
        print(vn, "done", round(o["wall"]), flush=True)
    return out
