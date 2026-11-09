# Part 18d — mutable output biases (setup)

**Status:** specs + CPU census ready; no Modal run yet.
**Parent:** Part 18c full (`bfdb70e` / `d248d3b`): only planted switchers flipped within bodies; `inputs`/`cue` did not. Switching phenotype sits at bias RMS ~0.10–0.17 from founders; 18c init left `g_A`/`g_B` at founder values.

## How mutation works (pre-18d)

In `fl_season.step`, every `m=5` steps, alive cells get Gaussian noise:

| channel | amplitude |
|---------|-----------|
| ecology `p[0:K0]` | `mutn` (0.002) |
| all `NW` brain weights (W1, b1, W2, **b2 including g_A/g_B**) | `mutw` (0.004) |
| memory `p[iM]` | **0** (never mutates) |

There is no separate reproduction-time / division-only mutation: inheritance is continuous spatial homogenization (`hom`) plus this per-step weight noise. **Biases were not frozen during 18c evolution** — they already mutated at `mutw`. They were frozen **at init** in the `inputs`/`cue` arms because `noise_new_channels` only touches `new_channel_flat()` (R, h, cue, o_h), which excludes `b2[g_A]` and `b2[g_B]`.

## Offline switcher census (CPU)

Same `is_switcher` as 18c. N=200 genomes × 4 founders (621–624). Foraging OK = `probe_dg(cue=+1)>0.2` and A-rich `>0`.

| σ_b on g_A/g_B | + inputs N(0,0.5) | switch % | forage OK % |
|----------------|-------------------|----------|-------------|
| 0.05 | no | 0.0 | 100 |
| 0.05 | yes | 0.5 | 94.5 |
| 0.10 | no | 0.0 | 100 |
| 0.10 | yes | 0.8 | 92.0 |
| 0.20 | no | 0.0 | 98.5 |
| 0.20 | yes | 1.0 | 86.9 |
| 0.30 | no | 0.0 | 93.2 |
| 0.30 | yes | 2.0 | 86.1 |
| inputs only (18c) | — | 0.9 | — |

Bias noise alone never yields cue-switchers (founder cue column is 0). Chosen init: **σ_b=0.3 + new-channel N(0,0.5)** → a few percent standing switchers without wrecking A-season foraging (~87% OK). Asymmetric plant-style bias edits give ~5% switchers but only ~54% forage OK — too harsh for a random init.

## Design

| arm | init | cue | `mut_bias` | role |
|-----|------|-----|------------|------|
| `bias_cue` | new N(0,0.5) + bias N(0,0.3) | 1 | **0.02** | test: standing bias variation + faster bias drift + oracle |
| `bias_nocue` | same genomes as `bias_cue` | 0 | **0.02** | control: bias variation without oracle |
| `inputs_cue` | new N(0,0.5) only (18c) | 1 | default (−1 → `mutw`) | 18c cue baseline |
| `planted` | 6 hand switchers + 6 founders | 1 | default | positive control |
| `nobrain` | lo noise | 0 | default | floor (`kb=kr=mem=0`) |

- World = 18c (`krep=90`, `Ts=8000`, `season_rule=1`, L=48, n_init=12).
- Seeds **841–844** inherit founders 621–624 (no collision with 18c 731–734).
- New ecology key `mut_bias`: `<0` (BDEF −1) → bias amp = `mutw` (legacy bit-identical values); `≥0` overrides only `b2[g_A]` and `b2[g_B]`. Absent from 18/18b/18c specs.

## Cost (from 18c actuals)

| | runs | steps | ~USD |
|--|------|-------|------|
| pilot | 5 (1 seed × 5 arms) | 12k | **~$5.0** |
| full | 20 (4 × 5) | 20k | **~$30.4** |

## Launch (after Igor approves GPU)

```bash
# regenerate specs / smoke
PYTHONPATH=src:.:experiments/part18 python experiments/part18/run18d.py smoke

modal run src/modal_app.py::spawn_h100 --kind ecolist --spec archive/specs/b18d_pilot.json --prefix b18dp
modal run src/modal_app.py::spawn_h100 --kind ecolist --spec archive/specs/b18d_full.json --prefix b18d

# analyze (reuse 18c metrics)
PYTHONPATH=src:.:experiments/part18 python experiments/part18/analyze18c.py 'results/eco18d/b18dp_*.pkl'
PYTHONPATH=src:.:experiments/part18 python experiments/part18/analyze18c.py 'results/eco18d/b18d_*.pkl'
```

## Files

- `src/fl_season.py` — `mut_bias` in BKEYS/BDEF; mutation amp override for g_A/g_B
- `experiments/part18/init18d.py`, `run18d.py`
- `archive/specs/b18d_pilot.json`, `b18d_full.json`
- `tests/test_season18d.py`
- `experiments/part18/results/b18d_init_report.json`, `part18d_notes.md`
