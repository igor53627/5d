"""Часть 9: анализ эко-прогонов — число объектов, масса, деления, наследуемость p (родитель→потомок), сдвиг распределения p."""
import sys, pickle, json, numpy as np
def wd(a, b, L): d = (np.asarray(a) - np.asarray(b) + L / 2) % L - L / 2; return float(np.linalg.norm(d))
def nearest(B, A, L):
    """Векторизованно: для каждого объекта B индекс ближайшего (периодически) объекта A."""
    cb = np.array([o[1] for o in B], float); ca = np.array([o[1] for o in A], float)
    d = (cb[:, None, :] - ca[None, :, :] + L / 2) % L - L / 2
    return np.argmin((d * d).sum(-1), axis=1)
def lineage(log, L, mmin=0.3):
    """Сопоставление снимков: потомок → ближайший объект на прошлом снимке. Деление: у родителя ≥2 потомков ≥mmin."""
    ev = []; pairs = []
    for (t0, A), (t1, B) in zip(log[:-1], log[1:]):
        if not A or not B: continue
        ch = {}
        for j, i in enumerate(nearest(B, A, L)):
            ch.setdefault(int(i), []).append(j)
        for i, js in ch.items():
            big = [j for j in js if B[j][0] >= mmin]
            if len(big) >= 2 and A[i][0] >= mmin:
                ev.append(dict(t=t1, parent=A[i][0], kids=[B[j][0] for j in big], p_parent=A[i][2], p_kids=[B[j][2] for j in big]))
                for j in big: pairs.append((A[i][2], B[j][2]))
    return ev, pairs
def summarize(r, L, every=100):
    log = r["log"]; n = [len([o for o in x[1] if o[0] >= 0.3]) for x in log]; M = [x[0] for x in r["tot"]]
    ev, pairs = lineage(log, L)
    pm = [np.average([o[2][0] for o in x[1]], weights=[o[0] for o in x[1]]) if x[1] else np.nan for x in log]
    return dict(n=n, M=M, F=[x[1] for x in r["tot"]], n_div=len(ev), events=ev, pairs=pairs, p0_mean=pm)
if __name__ == "__main__":
    L = int(sys.argv[1])
    for f in sys.argv[2:]:
        o = pickle.load(open(f, "rb"))["out"][0]
        for r in o["runs"]:
            s = summarize(r, L); e = r["eco"]
            print(f.split("/")[-1], {k: e[k] for k in ("keat", "dec", "rF") if k in e}, "tr" if r["transport"] else "NOtr", "n(t):", s["n"][::10], "M:", [round(x, 1) for x in s["M"][::20]], "div:", s["n_div"])

def herit(pairs, k=0):
    if len(pairs) < 3: return np.nan, len(pairs)
    P = np.array([(a[k], b[k]) for a, b in pairs]); 
    if P[:, 0].std() < 1e-9: return np.nan, len(pairs)
    return float(np.corrcoef(P[:, 0], P[:, 1])[0, 1]), len(pairs)

def pstats(log, k=0):
    """массо-взвешенные среднее и std p_k по объектам"""
    out = []
    for t, objs in log:
        if not objs: out.append((t, np.nan, np.nan)); continue
        w = np.array([o[0] for o in objs]); x = np.array([o[2][k] for o in objs])
        m = np.average(x, weights=w); out.append((t, float(m), float(np.sqrt(np.average((x - m) ** 2, weights=w)))))
    return out

def parents(log, L):
    """Для каждого снимка k>0: индекс родителя (ближайший объект на снимке k-1) для каждого объекта."""
    par = [None]
    for (t0, A), (t1, B) in zip(log[:-1], log[1:]):
        par.append([int(i) for i in nearest(B, A, L)] if (A and B) else [-1] * len(B))
    return par

def selection_diff(log, L, lag=10, k=0, t_min=0):
    """Ковариация p_k объекта и его относительного «потомства по массе» через lag снимков (уравнение Прайса, член отбора).
    Возвращает средний по времени S = cov(p, w)/mean(w), корреляцию и число пар."""
    par = parents(log, L); Ss = []; ps_all = []; ws_all = []
    for t in range(1, len(log) - lag):
        if log[t][0] < t_min: continue
        A = log[t][1]
        if len(A) < 2: continue
        idx = list(range(len(log[t + lag][1])))
        anc = idx[:]
        for s in range(t + lag, t, -1):
            anc = [par[s][a] if par[s] and a >= 0 and a < len(par[s]) else -1 for a in anc]
        w = np.zeros(len(A))
        for j, a in enumerate(anc):
            if 0 <= a < len(A): w[a] += log[t + lag][1][j][0]
        m = np.array([o[0] for o in A]); w = w / np.maximum(m, 1e-9); p = np.array([o[2][k] for o in A])
        if w.mean() <= 0: continue
        Ss.append(float(np.cov(p, w / w.mean(), bias=True)[0, 1])); ps_all += list(p); ws_all += list(w / w.mean())
    if not Ss: return np.nan, np.nan, 0
    return float(np.mean(Ss)), float(np.corrcoef(ps_all, ws_all)[0, 1]) if len(ps_all) > 3 else np.nan, len(ps_all)
