import gzip, json, collections, sys, re
def cat(e):
    n = e["name"].lower(); op = str(e.get("args", {}).get("hlo_op", "")).lower()
    if "fft" in n or "fft" in op or "cufft" in n or "radix" in n: return "FFT (cuFFT kernels)"
    if "transpose" in n or "transpose" in op: return "transpose (FFT layout)"
    if "memcpy" in n or "memset" in n: return "memcpy/memset"
    if "reduce" in n or "reduce" in op: return "reductions (std, sum)"
    if "fusion" in n or "fusion" in op or "loop" in n: return "XLA fusions (elementwise, stencils, MUSCL)"
    return "other"
for t in sys.argv[1:]:
    ev = json.load(gzip.open(t))["traceEvents"]; nsteps = 10
    pn = {e["pid"]: e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"}
    gp = {p for p, nm in pn.items() if "GPU" in nm}
    K = [e for e in ev if e.get("ph") == "X" and e.get("pid") in gp]
    t0 = min(e["ts"] for e in K); t1 = max(e["ts"] + e["dur"] for e in K); busy = sum(e["dur"] for e in K)
    print(f"== {t.split('/')[-1]}: span {(t1-t0)/1e3/nsteps:.1f} ms/step, GPU busy {busy/1e3/nsteps:.1f} ms/step ({100*busy/(t1-t0):.0f}%), {len(K)/nsteps:.0f} kernels/step")
    C = collections.defaultdict(lambda: [0, 0])
    for e in K: c = C[cat(e)]; c[0] += e["dur"]; c[1] += 1
    for k, (d, n) in sorted(C.items(), key=lambda kv: -kv[1][0]): print(f"   {k:45s} {d/1e3/nsteps:7.2f} ms {100*d/busy:5.1f}%  {n/nsteps:5.0f}/step")
    Nm = collections.defaultdict(lambda: [0, 0, ""])
    for e in K: c = Nm[e["name"][:70]]; c[0] += e["dur"]; c[1] += 1; c[2] = str(e["args"].get("hlo_op", ""))[:40]
    print("   top kernels:")
    for k, (d, n, op) in sorted(Nm.items(), key=lambda kv: -kv[1][0])[:14]: print(f"     {d/1e3/nsteps:6.2f} ms {100*d/busy:5.1f}%  x{n/nsteps:4.0f}  {k}  [{op}]")
    fk = [e for e in K if cat(e).startswith("FFT")]
    if fk: print("   sample FFT kernel_details:", str(fk[0]["args"].get("kernel_details", ""))[:300])
