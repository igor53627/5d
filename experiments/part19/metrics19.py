"""Part 19 metrics: coexistence, cycles, overlap, arms-race indicators."""
import math
import numpy as np

THRESH = 0.15  # in units of founder mass M1


def _xy(series):
    t = np.asarray([s[0] for s in series], float)
    y = np.asarray([s[1] for s in series], float)
    return t, y


def prey_mass(run):
    """tot entries are (M_A/M1, mean F)."""
    tot = run["tot"]
    t = np.asarray([i for i, _ in enumerate(tot)], float)
    # times are implicit if we only have tot; prefer log times
    if run.get("log"):
        t = np.asarray([frame[0] for frame in run["log"]], float)
    m = np.asarray([row[0] for row in tot], float)
    return t, m


def pred_mass(run):
    if not run.get("ptot"):
        t, _ = prey_mass(run)
        return t, np.zeros_like(t)
    t = np.asarray([frame[0] for frame in run.get("plog") or run["log"]], float)
    m = np.asarray([row[0] for row in run["ptot"]], float)
    return t, m


def counts(run, key="log"):
    frames = run.get(key) or []
    t = np.asarray([fr[0] for fr in frames], float)
    n = np.asarray([len(fr[1]) for fr in frames], int)
    return t, n


def kill_rate(run):
    """Prey loss / M1 at each log frame (snapshot, not a window mean)."""
    if not run.get("ptot"):
        t, _ = prey_mass(run)
        return t, np.zeros_like(t)
    t = np.asarray([frame[0] for frame in run.get("plog") or run["log"]], float)
    k = np.asarray([row[1] for row in run["ptot"]], float)
    return t, k


def coexistence_time(mA, mP, t, thresh=THRESH):
    alive = (np.asarray(mA) > thresh) & (np.asarray(mP) > thresh)
    if not np.any(alive):
        return 0.0
    # last time both were still above threshold (even if they later recover)
    # use last contiguous prefix? "coexistence time" = last t where both alive,
    # which equals how long the run stayed coexisting if they don't revive.
    return float(np.asarray(t)[np.where(alive)[0][-1]])


def first_extinction(mA, mP, t, thresh=THRESH):
    """('prey'|'pred'|'both'|None, time or None). First frame a species is down and stays down."""
    t = np.asarray(t); mA = np.asarray(mA); mP = np.asarray(mP)
    prey_down = mA <= thresh
    pred_down = mP <= thresh
    def first_stay(down):
        if not np.any(down):
            return None
        # first index from which it never recovers
        idx = np.where(down)[0]
        for i in idx:
            if np.all(down[i:]):
                return i
        return None
    iA, iP = first_stay(prey_down), first_stay(pred_down)
    if iA is None and iP is None:
        return None, None
    if iA is None:
        return "pred", float(t[iP])
    if iP is None:
        return "prey", float(t[iA])
    if iA == iP:
        return "both", float(t[iA])
    if iA < iP:
        return "prey", float(t[iA])
    return "pred", float(t[iP])


def final_status(mA, mP, thresh=THRESH):
    a, p = float(mA[-1]) > thresh, float(mP[-1]) > thresh
    if a and p:
        return "coexist"
    if a:
        return "prey_only"
    if p:
        return "pred_only"
    return "both_extinct"


def _detrend(y):
    n = len(y)
    if n < 3:
        return np.asarray(y, float) - np.mean(y)
    x = np.arange(n, dtype=float)
    c = np.polyfit(x, y, 1)
    return np.asarray(y, float) - np.polyval(c, x)


def peak_count(y):
    """Signed interior peaks of a detrended series (local maxima of |not used|, just maxima)."""
    y = np.asarray(y, float)
    if len(y) < 3:
        return 0
    return int(np.sum((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])))


def dominant_period(t, y):
    """Period of the largest FFT bin after a linear detrend. None if not oscillatory."""
    t = np.asarray(t, float); y = _detrend(y)
    n = len(y)
    if n < 8:
        return None
    dt = np.median(np.diff(t))
    if dt <= 0:
        return None
    spec = np.fft.rfft(y)
    freq = np.fft.rfftfreq(n, d=dt)
    mag = np.abs(spec)
    mag[0] = 0
    # ignore periods longer than 80% of the window
    ok = freq > 1.0 / (0.8 * (t[-1] - t[0] + dt))
    if not np.any(ok):
        return None
    k = np.argmax(np.where(ok, mag, 0.0))
    if mag[k] < 1e-12 or freq[k] <= 0:
        return None
    return float(1.0 / freq[k])


def best_lag(mA, mP, t, max_frac=0.4):
    """Lag (in time units) maximising corr(M_A(t), M_P(t+lag)). Positive = predator lags prey."""
    a, p = _detrend(mA), _detrend(mP)
    n = len(a)
    if n < 8 or np.std(a) < 1e-12 or np.std(p) < 1e-12:
        return None, None
    dt = float(np.median(np.diff(t)))
    max_lag = max(1, int(max_frac * n))
    scores = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            c = np.corrcoef(a[:n - lag] if lag else a, p[lag:] if lag else p)[0, 1]
        else:
            c = np.corrcoef(a[-lag:], p[:n + lag])[0, 1]
        if np.isfinite(c):
            scores.append((lag, float(c)))
    if not scores:
        return None, None
    best_c = max(c for _, c in scores)
    # Prefer the smallest |lag| among near-ties (periodic signals have aliases).
    near = [(lag, c) for lag, c in scores if c >= best_c - 0.02]
    lag, c = min(near, key=lambda kv: (abs(kv[0]), -kv[1]))
    return float(lag * dt), float(c)


def mean_g_av(run, out_av_offset=None):
    """Mass-weighted mean g_Av from object phenotypes, if the genome is in the log.

    Each object is (mass, center, q). q concatenates channels then D. The avoidance
    output is not stored as a raw output, so this uses D3 / mass when D is appended
    (D3 = A·g_Av, so the last-but-related diagnostic lives in the D block). Prefer
    the diagnostic path: if phenotype length is K+ND, index K+3 is mean g_Av.
    """
    if not run.get("log"):
        return None
    vals = []
    for _, objs in run["log"]:
        if not objs:
            continue
        num = den = 0.0
        for mass, _cen, q in objs:
            q = np.asarray(q, float)
            if out_av_offset is not None and out_av_offset < len(q):
                num += mass * q[out_av_offset]; den += mass
            elif len(q) >= 4:
                # fall back: last of a trailing ND=4 block if present
                num += mass * q[-1]; den += mass
        if den > 0:
            vals.append(num / den)
    return float(np.mean(vals)) if vals else None


def bodies_alive(n, m, thresh=THRESH):
    """True where the species has at least one object of mass >= 0.15 M1."""
    return (np.asarray(n) >= 1) & (np.asarray(m) > thresh)


def both_bodies(nA, nP, mA, mP, thresh=THRESH):
    return bodies_alive(nA, mA, thresh) & bodies_alive(nP, mP, thresh)


def first_body_loss(nA, nP, mA, mP, t, thresh=THRESH):
    """First log time both species do not both have a body (may later recover)."""
    t = np.asarray(t)
    alive = both_bodies(nA, nP, mA, mP, thresh)
    if len(t) == 0:
        return None
    if bool(alive[0]) is False:
        return float(t[0])
    lost = np.where(~alive)[0]
    return None if len(lost) == 0 else float(t[lost[0]])


def body_time_to_loss(nA, nP, mA, mP, t, thresh=THRESH):
    """First time a species drops all bodies and never recovers.

    Returns (who, t) with who in ('prey','pred','both', None). t is None if
    both still have a body at the end (possibly after a temporary dip).
    """
    t = np.asarray(t)
    a_down = ~bodies_alive(nA, mA, thresh)
    p_down = ~bodies_alive(nP, mP, thresh)

    def first_stay(down):
        if not np.any(down):
            return None
        for i in np.where(down)[0]:
            if np.all(down[i:]):
                return int(i)
        return None

    iA, iP = first_stay(a_down), first_stay(p_down)
    if iA is None and iP is None:
        return None, None
    if iA is None:
        return "pred", float(t[iP])
    if iP is None:
        return "prey", float(t[iA])
    if iA == iP:
        return "both", float(t[iA])
    if iA < iP:
        return "prey", float(t[iA])
    return "pred", float(t[iP])


def autocorr_period(t, y, min_peaks=1, min_r=0.2):
    """Period from the first positive peak of the detrended autocorrelation."""
    t = np.asarray(t, float); y = _detrend(y)
    n = len(y)
    if n < 16 or float(np.std(y)) < 1e-12:
        return None, None
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return None, None
    ac = np.correlate(y, y, mode="full")[n - 1:]
    if ac[0] == 0:
        return None, None
    ac = ac / ac[0]
    # Ignore lag 0 and very short lags (< 3 frames). Search to half the window.
    lo, hi = 3, n // 2
    if hi <= lo:
        return None, None
    seg = ac[lo:hi]
    peaks = np.where((seg[1:-1] > seg[:-2]) & (seg[1:-1] > seg[2:]) & (seg[1:-1] >= min_r))[0]
    if len(peaks) < min_peaks:
        return None, None
    k = int(peaks[0]) + lo
    return float(k * dt), float(ac[k])


def bootstrap_ci(values, n=10000, alpha=0.05, seed=0):
    """Percentile CI for the mean, resampling seeds with replacement."""
    xs = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not xs:
        return dict(mean=None, lo=None, hi=None, n=0)
    mean = float(np.mean(xs))
    if len(xs) == 1:
        return dict(mean=mean, lo=None, hi=None, n=1)
    rng = np.random.default_rng(seed)
    arr = np.asarray(xs, float)
    draws = rng.choice(arr, size=(n, len(arr)), replace=True).mean(axis=1)
    lo = float(np.percentile(draws, 100 * alpha / 2))
    hi = float(np.percentile(draws, 100 * (1 - alpha / 2)))
    return dict(mean=mean, lo=lo, hi=hi, n=len(xs))


def summarize_run(run, thresh=THRESH):

    t, mA = prey_mass(run)
    _, mP = pred_mass(run)
    _, nA = counts(run, "log")
    _, nP = counts(run, "plog") if run.get("plog") else (t, np.zeros_like(t))
    _, kill = kill_rate(run)
    who, when = first_extinction(mA, mP, t, thresh)
    lag, r = best_lag(mA, mP, t)
    # Body-level: an object must be at least 0.15 M1 (same cut as _objects).
    nA = np.asarray(nA); nP = np.asarray(nP)
    bodies = (nA >= 1) & (nP >= 1) & (np.asarray(mA) > thresh) & (np.asarray(mP) > thresh)
    coexist_bodies_t = float(t[np.where(bodies)[0][-1]]) if np.any(bodies) else 0.0
    aB, pB = (int(nA[-1]) >= 1 and float(mA[-1]) > thresh) if len(nA) else False, (int(nP[-1]) >= 1 and float(mP[-1]) > thresh) if len(nP) else False
    if aB and pB:
        status_bodies = "coexist"
    elif aB:
        status_bodies = "prey_only"
    elif pB:
        status_bodies = "pred_only"
    else:
        status_bodies = "no_bodies"
    perA_acf, rA_acf = autocorr_period(t, mA)
    perP_acf, rP_acf = autocorr_period(t, mP)
    loss_who, loss_t = body_time_to_loss(nA, nP, mA, mP, t, thresh)
    return dict(
        t_final=float(t[-1]) if len(t) else 0.0,
        mA_final=float(mA[-1]) if len(mA) else 0.0,
        mP_final=float(mP[-1]) if len(mP) else 0.0,
        nA_final=int(nA[-1]) if len(nA) else 0,
        nP_final=int(nP[-1]) if len(nP) else 0,
        status_bodies=status_bodies,
        coexist_bodies_t=coexist_bodies_t,
        mA_mean=float(np.mean(mA)) if len(mA) else 0.0,
        mP_mean=float(np.mean(mP)) if len(mP) else 0.0,
        status=final_status(mA, mP, thresh),
        coexist_t=coexistence_time(mA, mP, t, thresh),
        extinct=who, extinct_t=when,
        peaks_A=peak_count(_detrend(mA)),
        peaks_P=peak_count(_detrend(mP)),
        period_A=dominant_period(t, mA),
        period_P=dominant_period(t, mP),
        period_A_acf=perA_acf,
        period_P_acf=perP_acf,
        period_A_acf_r=rA_acf,
        period_P_acf_r=rP_acf,
        pred_lag=lag, pred_lag_r=r,
        kill_mean=float(np.mean(kill)) if len(kill) else 0.0,
        nA_mean=float(np.mean(nA)) if len(nA) else 0.0,
        nP_mean=float(np.mean(nP)) if len(nP) else 0.0,
        frac_bodies=float(np.mean(bodies)) if len(bodies) else 0.0,
        body_loss_first=first_body_loss(nA, nP, mA, mP, t, thresh),
        body_loss_who=loss_who,
        body_loss_t=loss_t,
        held_bodies=status_bodies == "coexist",
    )


def jsonable(x):
    """Convert numpy scalars for JSON: bool before int, non-finite floats to None."""
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, (np.integer, int)):
        return int(x)
    return x

