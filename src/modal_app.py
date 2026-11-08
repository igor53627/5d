"""
Modal-обвязка: GPU-бенчмарк и reservoir-прогоны Feigenbaum-Lenia (JAX).
  modal run modal_app.py::smoke --gpu L4
  modal run modal_app.py::bench --gpu B200
  modal run modal_app.py::res --gpu B200 --spec specs/tune_L8.json --out results/tune_L8.json
Запуски эфемерные (modal run): приложение останавливается само, когда локальный entrypoint завершился.
Учёт GPU-времени: каждый вызов пишет wall-time функции в results/modal_usage.jsonl.
"""
import json, os, time
import modal


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

def _src(name):
    return os.path.join(HERE, name)

def _archive(name):
    return os.path.join(ROOT, "archive", name)

def _exp(part, name):
    return os.path.join(ROOT, "experiments", part, name)

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("jax[cuda12]==0.12.0.1", "numpy", "scipy")
         .env({"XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
         # Core library (src/)
         .add_local_file(_src("fl_jax.py"), "/root/fl_jax.py")
         .add_local_file(_src("feigenbaum_lenia.py"), "/root/feigenbaum_lenia.py")
         .add_local_file(_src("alife_jax.py"), "/root/alife_jax.py")
         .add_local_file(_src("fl_glider.py"), "/root/fl_glider.py")
         .add_local_file(_src("glider_coll.py"), "/root/glider_coll.py")
         .add_local_file(_src("fl_eco.py"), "/root/fl_eco.py")
         .add_local_file(_src("fl_eco2.py"), "/root/fl_eco2.py")
         .add_local_file(_src("fl_eco3.py"), "/root/fl_eco3.py")
         .add_local_file(_src("fl_brain.py"), "/root/fl_brain.py")
         .add_local_file(_src("fl_poison.py"), "/root/fl_poison.py")
         .add_local_file(_src("fl_season.py"), "/root/fl_season.py")
         .add_local_file(_src("helpers.py"), "/root/helpers.py")
         .add_local_file(_src("analyze_eco.py"), "/root/analyze_eco.py")
         .add_local_file(os.path.join(ROOT, "_path.py"), "/root/_path.py")
         # Archive (moved from root)
         .add_local_file(_archive("reservoir_experiment.py"), "/root/reservoir_experiment.py")
         .add_local_file(_archive("reservoir_gpu.py"), "/root/reservoir_gpu.py")
         .add_local_file(_archive("bench.py"), "/root/bench.py")
         .add_local_file(_archive("glider_search.py"), "/root/glider_search.py")
         .add_local_file(_archive("glider_es.py"), "/root/glider_es.py")
         # Experiments
         .add_local_file(_exp("part05", "glider_cont.py"), "/root/glider_cont.py")
         .add_local_file(_exp("part06", "glider_es5.py"), "/root/glider_es5.py")
         .add_local_file(_exp("part08", "glider_rep.py"), "/root/glider_rep.py")
         .add_local_file(_exp("part10", "glider_eco.py"), "/root/glider_eco.py")
         .add_local_file(_exp("part10", "perf_eco.py"), "/root/perf_eco.py")
         .add_local_file(_exp("part11", "perf11.py"), "/root/perf11.py"))
app = modal.App("feigenbaum-lenia", image=image)
vol = modal.Volume.from_name("fl-results", create_if_missing=True)   # результаты переживают обрыв клиента

@app.function(gpu="L4", cpu=8, memory=65536, timeout=7200, volumes={"/vol": vol})
def gpu_job(kind, payload, save_as=None):
    import sys, subprocess; sys.path.insert(0, "/root")
    t0 = time.time()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    import jax
    if kind == "smoke":
        out = dict(devices=[str(d) for d in jax.devices()])
    elif kind == "bench":
        import bench; out = bench.bench_jax(**payload)
    elif kind == "res":
        import reservoir_gpu as RG
        out = [RG.run_batch(**p) for p in payload]
    elif kind == "glider":
        import glider_search as GS; out = [GS.run_candidates(**p) for p in payload]
    elif kind in ("cont", "es", "ver", "es5", "ver5", "ctrl5", "eval5", "iso5", "coll", "rep", "repeval", "repver", "cascade", "eco", "ecolist"):
        import pickle
        mod = __import__({"cont": "glider_cont", "es5": "glider_es5", "ver5": "glider_es5", "ctrl5": "glider_es5", "eval5": "glider_es5", "iso5": "glider_es5", "coll": "glider_coll", "rep": "glider_rep", "repeval": "glider_rep", "repver": "glider_rep", "cascade": "glider_rep", "eco": "glider_eco", "ecolist": "glider_eco"}.get(kind, "glider_es"))
        fn = {"cont": getattr(mod, "continuation", None), "es": getattr(mod, "run_es", None), "ver": getattr(mod, "run_verify", None),
              "es5": getattr(mod, "run_cma", None), "ver5": getattr(mod, "run_verify5", None), "ctrl5": getattr(mod, "run_ctrl5", None), "eval5": getattr(mod, "run_eval5", None), "iso5": getattr(mod, "run_iso", None), "coll": getattr(mod, "run_coll", None), "rep": getattr(mod, "run_rep_cma", None), "repeval": getattr(mod, "run_rep_eval", None), "repver": getattr(mod, "run_rep_verify", None), "cascade": getattr(mod, "run_cascade", None), "eco": getattr(mod, "run_eco", None), "ecolist": getattr(mod, "run_eco_list", None)}[kind]
        def cb(r):
            if save_as:
                pickle.dump(dict(kind=kind, gpu=gpu, wall=time.time() - t0, out=[r]), open(f"/vol/{save_as}", "wb")); vol.commit()
        r = fn(**payload, save_cb=cb); cb(r)
        return dict(kind=kind, gpu=gpu, wall=time.time() - t0, out=[], summary=str(r.get("summary"))[:2000], saved=save_as)
    elif kind == "perf":
        import perf_eco; out = [perf_eco.run_perf(**p) for p in payload]
    elif kind == "perf11":
        import perf11; out = [perf11.run_perf11(**payload)]
    elif kind == "val":
        import perf11; out = [perf11.run_val(**payload)]
    elif kind == "alife":
        import alife_jax; out = [alife_jax.run_alife(**p) for p in payload]
    res = dict(kind=kind, gpu=gpu, wall=time.time() - t0, out=out)
    if save_as:
        import pickle
        pickle.dump(res, open(f"/vol/{save_as}", "wb")); vol.commit()
        summ = [(x['task'], x['L'], x['B'], round(x['t_sim'])) for x in out] if kind == "res" else ([(x["N"], x["L"], x["B"], round(x["t_gpu"]), round(x["t_total"]), [o["n_glider"] for o in x["out"]]) for x in out] if kind == "glider" else [])
        return dict(kind=kind, gpu=gpu, wall=res["wall"], out=[], summary=summ, saved=save_as)
    return res

@app.function(gpu=["L4", "A10", "L40S"], cpu=8, memory=65536, timeout=7200, volumes={"/vol": vol})
def gpu_job_auto(kind, payload, save_as=None):
    """То же, что gpu_job, но с фолбэком по доступности GPU (B200 на Starter-плане бывает недоступен)."""
    return gpu_job.local(kind, payload, save_as)

def JOB(gpu):
    return gpu_job_auto if gpu == "auto" else gpu_job.with_options(gpu=gpu)

RESULTS = os.path.join(ROOT, "archive", "results")

def _log(fn_gpu, res, t_client, tag):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "modal_usage.jsonl"), "a") as f:
        f.write(json.dumps(dict(tag=tag, gpu_req=fn_gpu, gpu=res.get("gpu"), wall_fn=res["wall"], wall_client=t_client,
                                time=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")

def call(gpu, kind, payload, tag):
    t = time.time()
    res = JOB(gpu).remote(kind, payload)
    _log(gpu, res, time.time() - t, tag)
    return res

@app.local_entrypoint()
def smoke(gpu: str = "L4"):
    print(call(gpu, "smoke", None, "smoke"))

@app.local_entrypoint()
def bench(gpu: str = "L4", max_cells: int = 0):
    res = call(gpu, "bench", dict(max_cells=max_cells or None), f"bench_{gpu}")
    json.dump(res, open(os.path.join(RESULTS, f"bench_{gpu}.json"), "w"), indent=1)
    print(res["gpu"], res["wall"])

@app.local_entrypoint()
def res(gpu: str = "B200", spec: str = "", out: str = ""):
    """spec: json со списком вызовов (каждый — список kwargs для run_batch; вызовы идут параллельно в разных контейнерах)."""
    calls = json.load(open(spec))
    t = time.time()
    handles = [JOB(gpu).spawn("res", c) for c in calls]
    results = []
    for h in handles:
        r = h.get(); results.append(r)
        _log(gpu, r, time.time() - t, f"res_{os.path.basename(spec)}")
        print(r["gpu"], f"{r['wall']:.0f}s", r.get("summary") or [(x['task'], x['L'], x['B'], round(x['t_sim'])) for x in r["out"]], flush=True)
        json.dump(results, open(out, "w"))

@app.local_entrypoint()
def alife(gpu: str = "L4", size: int = 48, steps: int = 8000):
    L = size
    import pickle
    payload = [dict(L=L, steps=steps, lam_target=-0.005), dict(L=L, steps=steps, lam_target=-0.02),
               dict(L=L, steps=steps, lam_target=-0.005, kappa=0.7)]
    res = call(gpu, "alife", payload, f"alife_{gpu}")
    pickle.dump(res, open(os.path.join(RESULTS, f"alife_L{L}.pkl"), "wb"))
    print(res["gpu"], res["wall"], [r["sec"] for r in res["out"]])

@app.local_entrypoint()
def spawn(gpu: str = "B200", spec: str = "", prefix: str = ""):
    """Только запускает вызовы (с --detach переживают обрыв клиента); результаты -> Volume fl-results/<prefix>_<i>.pkl."""
    calls = json.load(open(spec))
    t = time.time()
    hs = [JOB(gpu).spawn("res", c, save_as=f"{prefix}_{i}.pkl") for i, c in enumerate(calls)]
    print("spawned", len(calls), flush=True)
    for h in hs:   # ждём (если клиент оборвётся, --detach оставит вызовы работать; результаты всё равно в Volume)
        r = h.get(); _log(gpu, r, time.time() - t, prefix)
        print(r["gpu"], f"{r['wall']:.0f}s", r.get("summary") or [(x['task'], x['L'], x['B'], round(x['t_sim'])) for x in r["out"]], flush=True)

@app.local_entrypoint()
def spawn_alife(gpu: str = "B200", size: int = 48, steps: int = 8000):
    payload = [dict(L=size, steps=steps, lam_target=-0.005), dict(L=size, steps=steps, lam_target=-0.02),
               dict(L=size, steps=steps, lam_target=-0.005, kappa=0.7)]
    t = time.time()
    r = JOB(gpu).spawn("alife", payload, save_as=f"alife_L{size}.pkl").get()
    _log(gpu, r, time.time() - t, f"alife_L{size}")
    print(r["gpu"], r["wall"])

@app.local_entrypoint()
def spawn_glider(gpu: str = "auto", spec: str = "", prefix: str = ""):
    """spec: json-список вызовов; каждый вызов — список kwargs для glider_search.run_candidates. Результаты -> Volume."""
    calls = json.load(open(spec)); t = time.time()
    hs = [JOB(gpu).spawn("glider", c, save_as=f"{prefix}_{i}.pkl") for i, c in enumerate(calls)]
    print("spawned", len(calls), flush=True)
    for h in hs:
        r = h.get(); _log(gpu, r, time.time() - t, prefix); print(r["gpu"], f"{r['wall']:.0f}s", r.get("summary"), flush=True)

@app.function(gpu=["L40S", "A10", "L4"], cpu=4, memory=24576, timeout=7200, volumes={"/vol": vol})
def gpu_job_lean(kind, payload, save_as=None):
    """Дешевле по CPU/памяти (для глайдеров): только L40S/A10/L4."""
    return gpu_job.local(kind, payload, save_as)

@app.local_entrypoint()
def spawn_glider_lean(spec: str = "", prefix: str = ""):
    calls = json.load(open(spec)); t = time.time()
    hs = [gpu_job_lean.spawn("glider", c, save_as=f"{prefix}_{i}.pkl") for i, c in enumerate(calls)]
    print("spawned", len(calls), flush=True)
    for h in hs:
        r = h.get(); _log("lean", r, time.time() - t, prefix); print(r["gpu"], f"{r['wall']:.0f}s", r.get("summary"), flush=True)

@app.local_entrypoint()
def spawn_kind(kind: str = "cont", spec: str = "", prefix: str = ""):
    """spec: json-список payload-ов (kwargs для glider_cont.continuation / glider_es.run_es); по контейнеру на payload."""
    calls = json.load(open(spec)); t = time.time()
    hs = [gpu_job_lean.spawn(kind, c, save_as=f"{prefix}_{i}.pkl") for i, c in enumerate(calls)]
    print("spawned", len(calls), flush=True)
    for h in hs:
        r = h.get(); _log("lean", r, time.time() - t, prefix); print(r["gpu"], f"{r['wall']:.0f}s", r.get("summary"), flush=True)

@app.function(gpu=["A10", "L4"], cpu=4, memory=24576, timeout=7200, volumes={"/vol": vol})
def gpu_job_cheap(kind, payload, save_as=None):
    """Самые дешёвые GPU (A10/L4) — для проверок."""
    return gpu_job.local(kind, payload, save_as)

@app.local_entrypoint()
def spawn_cheap(kind: str = "ver", spec: str = "", prefix: str = ""):
    calls = json.load(open(spec)); t = time.time()
    hs = [gpu_job_cheap.spawn(kind, c, save_as=f"{prefix}_{i}.pkl") for i, c in enumerate(calls)]
    print("spawned", len(calls), flush=True)
    for h in hs:
        r = h.get(); _log("cheap", r, time.time() - t, prefix); print(r["gpu"], f"{r['wall']:.0f}s", r.get("summary"), flush=True)

@app.function(gpu=["L40S", "H100"], cpu=4, memory=32768, timeout=7200, volumes={"/vol": vol})
def gpu_job_big(kind, payload, save_as=None):
    """Тяжёлые проверки: L40S (H100 как фолбэк; одобрено Игорем)."""
    return gpu_job.local(kind, payload, save_as)

@app.local_entrypoint()
def spawn_big(kind: str = "ver", spec: str = "", prefix: str = ""):
    calls = json.load(open(spec)); t = time.time()
    hs = [gpu_job_big.spawn(kind, c, save_as=f"{prefix}_{i}.pkl") for i, c in enumerate(calls)]
    print("spawned", len(calls), flush=True)
    for h in hs:
        r = h.get(); _log("big", r, time.time() - t, prefix); print(r["gpu"], f"{r['wall']:.0f}s", r.get("summary"), flush=True)

@app.function(gpu="H100", cpu=4, memory=32768, timeout=4 * 3600, volumes={"/vol": vol})
def gpu_job_h100(kind, payload, save_as=None):
    """Длинный 5D CMA-ES (H100; Modal может выдать H200)."""
    return gpu_job.local(kind, payload, save_as)

@app.local_entrypoint()
def spawn_h100(kind: str = "es5", spec: str = "", prefix: str = ""):
    calls = json.load(open(spec)); t = time.time()
    hs = [gpu_job_h100.spawn(kind, c, save_as=f"{prefix}_{i}.pkl") for i, c in enumerate(calls)]
    print("spawned", len(calls), flush=True)
    for h in hs:
        r = h.get(); _log("h100", r, time.time() - t, prefix); print(r["gpu"], f"{r['wall']:.0f}s", r.get("summary"), flush=True)

@app.function(gpu="B300", cpu=4, memory=65536, timeout=4 * 3600, volumes={"/vol": vol})
def gpu_job_b300(kind, payload, save_as=None):
    """Часть 10: попытка B300 (по просьбе Игоря)."""
    return gpu_job.local(kind, payload, save_as)

@app.local_entrypoint()
def spawn_b300(kind: str = "eco", spec: str = "", prefix: str = ""):
    calls = json.load(open(spec)); t = time.time()
    hs = [gpu_job_b300.spawn(kind, c, save_as=f"{prefix}_{i}.pkl") for i, c in enumerate(calls)]
    print("spawned", len(calls), flush=True)
    for h in hs:
        r = h.get(); _log("b300", r, time.time() - t, prefix); print(r["gpu"], f"{r['wall']:.0f}s", r.get("summary"), flush=True)


@app.local_entrypoint()
def spawn_gpu(kind: str = "perf11", spec: str = "", prefix: str = "", gpu: str = "H100"):
    """Часть 11: любой тип GPU (H100/H200/B300) через with_options."""
    calls = json.load(open(spec)); t = time.time(); fn = gpu_job_h100.with_options(gpu=gpu, memory=65536)
    hs = [fn.spawn(kind, c, save_as=f"{prefix}_{i}.pkl") for i, c in enumerate(calls)]
    print("spawned", len(calls), gpu, flush=True)
    for h in hs:
        r = h.get(); print(r["gpu"], f"{r['wall']:.0f}s", flush=True)
