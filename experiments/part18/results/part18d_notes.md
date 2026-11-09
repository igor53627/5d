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
| `bias_cue` | new N(0,0.5) + bias N(0,0.3) | 1 | **0.05** | test: standing bias variation + faster bias drift + oracle |
| `bias_nocue` | same genomes as `bias_cue` | 0 | **0.05** | control: bias variation without oracle |
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

## Mut_bias tuning (post-pilot)

Pilot at `mut_bias=0.02` drifted bias-diff toward 0 in **both** `bias_cue` and `bias_nocue` (mutation/hom, not selection). Naive RW std = `amp·√T` (step uses `amp·√m`); hom≈4× damps (pilot body RMS ~0.55 vs naive ~2.2 at 12k).

Offline RW census (48 genomes, no hom): mb=0.05 still yields switchers with forage ≳40%; mb≥0.1 hurts forage; mb=0.2 ≈ useless for switchers.

Short CPU ecology (L=24, 800 steps, n_init=2) — **too short for selection**, hygiene only: mass floor identical; A-season poison rises with mb (`poiA` 0.18→0.26→0.43 at 0.02/0.05/0.10). Chose **0.05**.

**No re-pilot.** Go full (~$30). Fusion/survival already OK at 0.02; 0.05 is a moderate raise inside the safe offline band.

Greptile P2: `tests/test_mut_bias_amp.py` — `step()` with `mut_bias≥0` mutates only `b2[g_A]/b2[g_B]` when `mutw=0`; `mut_bias<0` uses `mutw` for those channels.

## Pilot 2 results (b18dp2, mut_bias=0.05, seed 841, ~$4.0, ~13 min wall)

Same world as pilot 1 (krep=90, Ts=8000, 12k steps) except `mut_bias=0.05` on bias_* arms. Prefix `b18dp2` (pilot 1 kept as `b18dp`).

| arm | n_end | mass | max_frac | fus | sw init→end (mass%) | mean g A/B | poison A/B | Δbias-diff | within@t8000 |
|-----|-------|------|----------|-----|---------------------|------------|------------|------------|--------------|
| bias_cue | 11 | 13.3 | 0.16 | none | 0→0 (0%) | **+0.36 / −0.37** | 0.39 / **0.43** | −0.95 | **+0.69** |
| bias_nocue | 12 | 12.9 | 0.13 | none | 0→1 (13%) | +0.35 / −0.09 | 0.41 / 0.48 | −1.16 | +0.17 |
| inputs_cue | 12 | 20.2 | 0.11 | none | 0→0 | +1.45 / +0.94 | 0.19 / 0.72 | +0.00 | +0.47 |
| planted | 12 | 21.1 | 0.10 | none | 6→3 (23%) | +1.26 / +0.11 | 0.20 / 0.48 | −0.06 | +1.22 |
| nobrain | 12 | 14.8 | 0.10 | none | 0→0 | +1.60 / +1.63 | 0.49 / 0.50 | −0.02 | 0.00 |

### vs pilot 1 (mut_bias=0.02)

| | bias_cue p1 | bias_cue p2 | bias_nocue p1 | bias_nocue p2 |
|--|-------------|-------------|---------------|---------------|
| mean g A/B | +0.99/+0.81 | **+0.36/−0.37** | +0.94/+0.85 | +0.35/−0.09 |
| poison B | 0.73 | **0.43** | 0.71 | 0.48 |
| within@8k | +0.06 | **+0.69** | −0.02 | +0.17 |
| mass | 17.3 | 13.3 | 18.3 | 12.9 |
| cue−nocue Δ(gA−gB) | +0.10 | **+0.29** | — | — |

**Readout:** raising mut_bias to 0.05 makes `bias_cue` **diverge** from `bias_nocue`: season-aligned mean g (A+/B−), larger within-body Δg at the B→A switch, and lower B-season poison. Mild mass cost (~25%). Hard `is_switcher` count at end is still ~0 (phenotype is continuous). Planted control still switches. No early fusion.

**Go for full run (~$30)** with mut_bias=0.05. Logs: `results/eco18d/b18dp2_*.pkl`, `pilot18d2_summary.json`.

