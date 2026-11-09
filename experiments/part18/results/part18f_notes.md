# Part 18f — longer-horizon within-body pilot

**Status:** pilot done (`b18fp`, ~$8.32). Full on mb/hom/lead variants **not** recommended.
**Parent:** 18e (`cff3d53`): single A→B switch; lowdec hurt within_aligned; cue_lead flat on bias_cue.

## Goal

Measure **within_aligned** (+ = toward new season) over **both** A→B and B→A; test higher `mut_bias`, weaker `hom`, and `cue_lead` sanity on planted.

## Design

| | |
|--|--|
| Steps | **12000** (2 switches @ Ts=8000: A→B ~4.2k, B→A ~8.2k) |
| Seeds | 841, 842 |
| Arms | `bias_cue` (mb=0.05, hom=0.05); `bias_cue_mb10` (mb=0.10); `bias_cue_lowhom` (hom=0.0125); `planted`; `planted_lead` (cue_lead=400) |
| Dropped | lowdec (18e) |
| Cost | **~$8.32** (2.11 GPU-h; 8×H100 + 2×H200) |

## Results

### Overall means (2 seeds × 2 switches)

| arm | within_al | repl_al | matched | crossed | poiB | mass | gA / gB |
|-----|----------:|--------:|--------:|--------:|-----:|-----:|---------|
| **bias_cue** | **+0.447** | +0.106 | 0.56 | 50% | 0.45 | 14.9 | +0.05 / −0.40 |
| bias_cue_mb10 | +0.208 | +0.089 | 0.55 | **100%** | 0.50 | 15.5 | +0.10 / −0.21 |
| bias_cue_lowhom | +0.142 | +0.049 | 0.53 | 50% | 0.47 | 13.5 | +0.10 / −0.35 |
| planted | **+0.511** | +0.492 | 0.46 | 50% | 0.46 | 20.2 | +0.48 / −0.36 |
| planted_lead | −0.015 | +0.021 | 0.41 | 50% | 0.45 | 19.5 | +0.06 / +0.04 |

### within_aligned by direction

| arm | A→B | B→A |
|-----|----:|----:|
| bias_cue | +0.36 | +0.53 |
| bias_cue_mb10 | +0.24 | +0.17 |
| bias_cue_lowhom | +0.13 | +0.15 |
| planted | +0.13 | +0.89 |
| planted_lead | −0.01 | −0.02 |

### vs 18d / 18e

| | bias_cue within_al | planted within_al |
|--|------------------:|------------------:|
| 18d full (16 sw) | +0.26 | +0.68 |
| 18e pilot (1 A→B) | +0.14 | +0.07 |
| **18f pilot (2 sw)** | **+0.45** | **+0.51** |

## Readout

1. **Longer horizon changes the story:** with both switch directions, baseline `bias_cue` already shows strong within-body seasonal Δg (+0.45), nearly matching planted (+0.51) and beating the 18d full mean (+0.26). 18e’s single A→B window understated this.
2. **Higher mut_bias (0.10)** and **weaker hom (0.0125)** both **lower** within_aligned vs baseline. mb10 does raise crossed% (100%) but via noisier / less coherent within.
3. **`cue_lead` on planted fails the sanity check** (−0.015 vs planted +0.511): advance cue desynchronizes genomes wired to current σ. Do not use cue_lead with planted-style switchers.
4. Replacement still contributes for planted (repl_al +0.49); bias_cue’s Δg is mostly within (repl_al only +0.11).

**Verdict:** neither mb↑ nor hom↓ nor cue_lead improves within_aligned. Baseline at ≥12k is already close to planted on within_al.

## Next

- Do **not** full-run the mb/hom/lead variants.
- Optional confirmatory full: 4 seeds × {bias_cue, bias_nocue, planted} × 20k, report within_aligned by direction (no new knobs) — to confirm cue-selected within-body switching vs nocue.
- Or treat within-body pathway as demonstrated for bias_cue and move on.

Logs: `results/eco18f/b18fp_{0..9}.pkl`; summary `pilot18f_summary.json`.
