"""Part 18c founder genomes.

The Part 17 weights mapped in Part 18b are loaded from archive/specs/b18bm.json.
Old channels stay at those values except in the planted control, which is allowed to
move the cue column and the g_A / g_B output biases. See part18c_notes.md.
"""
import json
from pathlib import Path
import numpy as np
import fl_season as Z

ROOT = Path(__file__).resolve().parents[2]
SPEC_18B = ROOT / "archive" / "specs" / "b18bm.json"
# Same body-size inputs analyze18.py freezes when it reconstructs a brain.
PROBE_UB, PROBE_UFAR = -0.4, 0.06
N_INIT = 12
N_PLANTED = 6
# Switch acceptance at the mixed-food probe, and a weaker check on an A-rich patch.
DG_ON, DG_OFF, DG_RICH = 0.4, -0.4, 0.2
PLANT_TRIALS = 2500
# Independent of the placement RNG inside run_eco, so min_gap retries cannot move weights.
WEIGHT_STREAM = 10007
# 18c seeds and the Part 18b run whose mapped genome they inherit.
SEED_OF_FOUNDER = {731: 621, 732: 622, 733: 623, 734: 624}
ARMS = ("lo", "inputs", "cue", "planted", "nobrain")


def load_founders(path=SPEC_18B):
    """seed -> length-KB genome used as p0 in the Part 18b mem runs."""
    out = {}
    for item in json.loads(Path(path).read_text()):
        run = item["runs"][0]
        if run["tag"].startswith("mem_"):
            out[int(run["seed"])] = np.asarray(run["p0"], float)
    return out


def _weights(p):
    return np.asarray(p[Z.K0:Z.iM], float).copy()


def _write_weights(p, w):
    q = np.asarray(p, float).copy()
    q[Z.K0:Z.iM] = np.clip(w, -1.0, 1.0)
    return q


def probe_dg(w, cue, a=0.55, b=0.45, reward=0.0, h=0.0, ws=2.0):
    s = np.array([a, b, reward, h, cue, PROBE_UB, PROBE_UFAR], np.float32)
    o = Z.brain(np.asarray(w, np.float32), s, ws)
    return float(o[0] - o[1])


def is_switcher(w):
    """Cue +1 keeps the A reflex; cue -1 flips it. A-rich food must not undo the A season."""
    on = probe_dg(w, 1.0) > DG_ON and probe_dg(w, 1.0, a=0.85, b=0.15) > DG_RICH
    off = probe_dg(w, -1.0) < DG_OFF
    return bool(on and off)


def _views(w):
    return [np.array(x, float) for x in Z.unpack(np.asarray(w, np.float32), 1.0)]


def _pack(W1, b1, W2, b2):
    return np.concatenate([np.asarray(W1, float).ravel(), np.asarray(b1, float).ravel(), np.asarray(W2, float).ravel(), np.asarray(b2, float).ravel()])


def noise_new_channels(p0, rng, std):
    """Keep every trained weight. Add N(0, std) on R, h, cue, and the o_h readout, then clip."""
    w = _weights(p0)
    idx = np.array(Z.new_channel_flat())
    w[idx] = w[idx] + rng.normal(0.0, std, size=idx.size)
    return _write_weights(p0, w)


def plant_switcher(p0, rng, trials=PLANT_TRIALS):
    """Smallest RMS edit that passes is_switcher. Touches only the cue column and b2 of g_A, g_B."""
    w0 = _weights(p0)
    best = None
    hits = 0
    for _ in range(trials):
        W1, b1, W2, b2 = _views(w0)
        W1[:, Z.IN_CUE] = rng.uniform(-1.0, 1.0, size=Z.H)
        b2[Z.OUT_GA] = np.clip(b2[Z.OUT_GA] + rng.uniform(-0.9, 0.0), -1.0, 1.0)
        b2[Z.OUT_GB] = np.clip(b2[Z.OUT_GB] + rng.uniform(-0.2, 0.5), -1.0, 1.0)
        w = np.clip(_pack(W1, b1, W2, b2), -1.0, 1.0)
        if not is_switcher(w):
            continue
        hits += 1
        dist = float(np.sqrt(np.mean((w - w0) ** 2)))
        if best is None or dist < best[0]:
            best = (dist, w, hits)
    if best is None:
        raise RuntimeError(f"no cue-switcher in {trials} trials (hits={hits})")
    return _write_weights(p0, best[1]), dict(rms=best[0], hits=hits, trials=trials)


def population(p0, seed, arm, n=N_INIT, std_lo=0.05, std_new=0.5):
    """n genomes for one arm. cue shares the inputs draw; nobrain shares the lo draw."""
    if arm in ("cue", "inputs"):
        stream = "inputs"
    elif arm in ("nobrain", "lo"):
        stream = "lo"
    elif arm == "planted":
        stream = "planted"
    else:
        raise ValueError(arm)
    rng = np.random.default_rng([int(seed), WEIGHT_STREAM, {"lo": 1, "inputs": 2, "planted": 3}[stream]])
    rows = []
    plant_info = []
    for i in range(n):
        if stream == "lo":
            noise = rng.normal(0.0, std_lo, size=p0.size)
            rows.append(np.clip(np.asarray(p0, float) + noise, -1.0, 1.0))
        elif stream == "inputs":
            rows.append(noise_new_channels(p0, rng, std_new))
        else:
            if i < n - N_PLANTED:
                rows.append(np.asarray(p0, float).copy())
                plant_info.append(None)
            else:
                g, info = plant_switcher(p0, rng)
                rows.append(g)
                plant_info.append(info)
    return np.stack(rows, 0), plant_info


def genome_report(rows, p0=None):
    rows = np.asarray(rows, float)
    w = rows[:, Z.K0:Z.iM]
    new = np.array(Z.new_channel_flat())
    old = np.ones(w.shape[1], dtype=bool)
    old[new] = False
    flags = [is_switcher(wi) for wi in w]
    out = dict(
        n=int(len(rows)),
        n_switch=int(sum(flags)),
        switch=[bool(x) for x in flags],
        std_new=float(w[:, new].std()),
        std_old=float(w[:, old].std()),
        max_abs_new=float(np.abs(w[:, new]).max()) if new.size else 0.0,
        mean_abs_new=float(np.abs(w[:, new]).mean()) if new.size else 0.0,
        dg_cue_pos=[float(probe_dg(wi, 1.0)) for wi in w],
        dg_cue_neg=[float(probe_dg(wi, -1.0)) for wi in w],
    )
    if p0 is not None:
        d = w - np.asarray(p0, float)[Z.K0:Z.iM]
        out["max_abs_delta_old"] = float(np.abs(d[:, old]).max())
        out["max_abs_delta_new"] = float(np.abs(d[:, new]).max())
        out["std_delta_old"] = float(d[:, old].std())
        out["std_delta_new"] = float(d[:, new].std())
    return out
