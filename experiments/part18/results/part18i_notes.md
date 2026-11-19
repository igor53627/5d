# Part 18i — more seeds for the 18g confirmatory assay

**Status:** done (`b18i`, ~$15.05). Pooled with 18g: **8 seeds/arm**.
**Parent:** 18g (seeds 851–854): bias_cue within_al +0.07, cue−nocue Δ(gA−gB) −0.25. 18f (841–842) had looked much stronger.

## Design

Same world as 18g: `{bias_cue, bias_nocue, planted}`, mut_bias=0.05, 20k, 4 switches. New seeds **855–858** → founders 621–624. No new knobs.

Cost **~$15.05** (3.81 GPU-h).

## 18i only (4 seeds)

| arm | within_al (95% CI) | A→B | B→A | repl_al | crossed | poiB |
|-----|-------------------:|----:|----:|--------:|--------:|-----:|
| bias_cue | **+0.33 [0.17, 0.47]** | +0.08 | +0.59 | +0.34 | 2/4 | 0.45 |
| bias_nocue | +0.03 [−0.03, 0.09] | −0.01 | +0.08 | +0.01 | 2/4 | 0.54 |
| planted | +0.48 [0.18, 0.75] | +0.26 | +0.71 | +0.50 | 2/4 | 0.48 |

cue−nocue: Δ(gA−gB)=+0.24 [−0.06, 0.54]; **Δwithin_al=+0.30 [0.13, 0.49]**; ΔpoiB=−0.10 [−0.15, −0.03].

## Pooled 18g+18i (8 seeds) — primary

Seed-level means, bootstrap 95% CIs (10,000 resamples).

| arm | within_al | A→B | B→A | repl_al | crossed | poiB |
|-----|----------:|----:|----:|--------:|--------:|-----:|
| bias_cue | **+0.20 [0.07, 0.34]** | +0.02 [−0.06, 0.08] | **+0.39 [0.19, 0.62]** | +0.26 [0.15, 0.37] | 4/8 | 0.48 [0.44, 0.51] |
| bias_nocue | +0.03 [−0.01, 0.07] | −0.01 [−0.05, 0.03] | +0.07 [0.03, 0.12] | +0.01 [−0.02, 0.05] | 3/8 | 0.54 [0.47, 0.61] |
| planted | **+0.45 [0.24, 0.68]** | +0.25 [0.06, 0.49] | +0.65 [0.37, 0.95] | +0.51 [0.39, 0.63] | **6/8** | 0.50 [0.46, 0.53] |

### cue − nocue (8 pairs)

| | mean | 95% CI | n_pos | t-test p | Wilcoxon p |
|--|-----:|--------|------:|---------:|-----------:|
| Δwithin_al | **+0.173** | **[+0.039, +0.316]** | 7/8 | 0.057 | 0.055 |
| Δ(gA−gB) | −0.003 | [−0.229, +0.249] | 4/8 | 0.983 | 1.000 |
| ΔpoiB | **−0.060** | **[−0.115, −0.005]** | 2/8 (6/8 lower poiB) | 0.084 | 0.148 |
| crossed | cue **4/8**, nocue 3/8 | | | | |

p-values are two-sided against 0 on the 8 paired differences: one-sample t-test and exact Wilcoxon signed-rank test. `pool18i.py` rebuilds this section from the per-run records (bootstrap seed 0) and writes `pool18i_summary.json`.

### vs earlier parts

| | bias_cue wal | planted wal | Δ(gA−gB) | Δwithin_al |
|--|-------------:|------------:|---------:|-----------:|
| 18d full (841–844) | +0.26 | +0.68 | **+0.33** (4/4 +) | +0.04 (raw) |
| 18f pilot (841–842) | +0.45 | +0.51 | — | — |
| 18g (851–854) | +0.07 | +0.42 | −0.25 (1/4 +) | +0.05 |
| **18g+18i pool (8)** | **+0.20 [0.07, 0.34]** | +0.45 [0.24, 0.68] | **−0.00 [−0.23, 0.25]** | **+0.17 [0.04, 0.32]** |

## Readout

1. **Suggestive: cue may raise within-body seasonal Δg.** Pooled Δwithin_al is +0.17 (7/8 pairs positive). The bootstrap CI excludes 0, but the t-test (p=0.057) and the Wilcoxon signed-rank test (p=0.055) do not reach 0.05. nocue stays near 0. 18g alone gave +0.05.
2. **Planted is still stronger** (+0.45 vs +0.20) and more often crossed (6/8 vs 4/8). Evolved bias_cue does not reliably match the planted control.
3. **End-point Δ(gA−gB) is not a stable cue effect** (pool CI includes 0 and is far from 18d’s +0.33). Season-matched *mean* biases can come from replacement; within_al is the cleaner switching metric.
4. **Asymmetry persists:** B→A within_al is large for cue; A→B is ~0.
5. **Slightly lower B-season poison with cue, not significant** (ΔpoiB −0.06; the bootstrap CI just excludes 0, t-test p=0.084, Wilcoxon p=0.148).

**What within_al covers.** `metrics18c.match_bodies` pairs a body with the only body within 8 cells one snapshot later. In the 18g and 18i logs, 99.7–99.8% of these pairs agree with the ancestor identified by heritable traits (`lineage_tracking.assign_ancestors`; table in `experiments/tracking_fix/results/tracking_validation.md`). The agreement shows a line of descent. It does not show one undivided body, because both rules accept a parent and a daughter that stays within 8 cells. within_al is thus the change of g along one line of descent in 200 steps, and it can include a division. The pairs cover 37–49% of the old mass; every other body enters the `fusion`, `uncertain` or `replacement` term.

**Limits of the pool.** The pool was formed after the confirmatory 18g run (4 seeds) gave a null result (Δwithin_al +0.05), so the number of seeds was not fixed in advance. The primary metric also changed: 18d used the end-point difference Δ(gA−gB), and within_al replaced it after 18d. The 8 seeds use 4 founders: 851–854 and 855–858 both map to founders 621–624 (`run18g.py`, `run18i.py`), with new genome-noise and run seeds. A confirmatory claim needs a preregistered replicate.

**Verdict:** Across 8 seeds, the result is **suggestive, not confirmatory**: oracle cue may select for more within-body switching than nocue (Δwithin_al +0.17, bootstrap CI excludes 0, t-test p=0.057, Wilcoxon p=0.055). The effect is moderate, direction-asymmetric, and much noisier than planted. 18d’s strong Δ(gA−gB) does not hold in this pool.

Logs: `results/eco18i/b18i_{0..11}.pkl` plus 18g `b18g_*`; summary `full18i_summary.json`; pooled statistics `pool18i_summary.json` (from `pool18i.py`).
