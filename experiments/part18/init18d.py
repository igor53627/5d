"""Part 18d founder genomes: 18c init plus N(0, sigma_b) on g_A / g_B biases.

Reproduction already mutates all NW weights at mutw (including biases). 18c only
froze biases at init by omitting them from new_channel_flat(). 18d adds standing
variation on b2[g_A], b2[g_B] and an optional mut_bias ecology override.
"""
import numpy as np
import fl_season as Z
import init18c as I

# Fresh seeds so HF / Modal prefixes do not collide with 18c (731–734).
SEED_OF_FOUNDER = {841: 621, 842: 622, 843: 623, 844: 624}
ARMS = ("bias_cue", "bias_nocue", "inputs_cue", "planted", "nobrain")
STD_BIAS = 0.3
# Census: allnew0.5 + bias0.3 → ~1% switchers, ~87% forage-OK averaged over founders.
WEIGHT_STREAM = 10008  # distinct from 18c's 10007


def noise_biases(p0, rng, std=STD_BIAS):
    """Add N(0, std) to b2 of g_A and g_B only; clip to [-1, 1]."""
    w = I._weights(p0)
    W1, b1, W2, b2 = I._views(w)
    b2[Z.OUT_GA] = np.clip(b2[Z.OUT_GA] + rng.normal(0.0, std), -1.0, 1.0)
    b2[Z.OUT_GB] = np.clip(b2[Z.OUT_GB] + rng.normal(0.0, std), -1.0, 1.0)
    return I._write_weights(p0, I._pack(W1, b1, W2, b2))


def noise_new_and_bias(p0, rng, std_new=0.5, std_bias=STD_BIAS):
    """18c new-channel noise, then bias noise (order matches census)."""
    return noise_biases(I.noise_new_channels(p0, rng, std_new), rng, std_bias)


def population(p0, seed, arm, n=I.N_INIT, std_lo=0.05, std_new=0.5, std_bias=STD_BIAS):
    """n genomes for one 18d arm. bias_nocue shares bias_cue; nobrain shares lo draw."""
    if arm in ("bias_cue", "bias_nocue"):
        stream = "bias"
    elif arm == "inputs_cue":
        stream = "inputs"
    elif arm == "nobrain":
        stream = "lo"
    elif arm == "planted":
        stream = "planted"
    else:
        raise ValueError(arm)
    rng = np.random.default_rng([int(seed), WEIGHT_STREAM, {"lo": 1, "inputs": 2, "planted": 3, "bias": 4}[stream]])
    rows = []
    plant_info = []
    for i in range(n):
        if stream == "lo":
            noise = rng.normal(0.0, std_lo, size=p0.size)
            rows.append(np.clip(np.asarray(p0, float) + noise, -1.0, 1.0))
        elif stream == "inputs":
            rows.append(I.noise_new_channels(p0, rng, std_new))
        elif stream == "bias":
            rows.append(noise_new_and_bias(p0, rng, std_new, std_bias))
        else:
            if i < n - I.N_PLANTED:
                rows.append(np.asarray(p0, float).copy())
                plant_info.append(None)
            else:
                g, info = I.plant_switcher(p0, rng)
                rows.append(g)
                plant_info.append(info)
    return np.stack(rows, 0), plant_info


def bias_indices():
    return (Z.b2_flat(Z.OUT_GA), Z.b2_flat(Z.OUT_GB))


def genome_report(rows, p0=None):
    """Extend 18c report with bias-channel deltas."""
    out = I.genome_report(rows, p0)
    rows = np.asarray(rows, float)
    w = rows[:, Z.K0:Z.iM]
    bi = list(bias_indices())
    out["std_bias"] = float(w[:, bi].std())
    out["mean_abs_bias"] = float(np.abs(w[:, bi]).mean())
    if p0 is not None:
        d = w - np.asarray(p0, float)[Z.K0:Z.iM]
        out["std_delta_bias"] = float(d[:, bi].std())
        out["max_abs_delta_bias"] = float(np.abs(d[:, bi]).max())
    return out
