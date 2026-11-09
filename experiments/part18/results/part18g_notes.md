# Part 18g — confirmatory full (within_aligned, cue vs nocue)

**Status:** full done (`b18g`, ~$15.17).
**Parent:** 18f pilot (`ec0b39a`): on seeds 841–842, bias_cue within_al≈+0.45 ≈ planted +0.51 over 2 switches. Recommended confirmatory full with new seeds + nocue.

## Design

| | |
|--|--|
| Seeds | **851–854** → founders 621–624 (new; not 841–844) |
| Arms | `bias_cue` (cue=1, mb=0.05), `bias_nocue` (cue=0, mb=0.05, **shared genomes**), `planted` |
| Steps | **20000** (4 switches: A→B, B→A, A→B, B→A at Ts=8000) |
| Knobs | none new (18d/18f world, krep=90) |
| Cost | **~$15.17** (3.84 GPU-h) |

## Results

### Per-arm means (4 seeds × 4 switches)

| arm | within_al | A→B | B→A | repl_al | matched | crossed | poiB | mass | gA / gB |
|-----|----------:|----:|----:|--------:|--------:|--------:|-----:|-----:|---------|
| bias_cue | +0.073 | −0.05 | +0.20 | +0.19 | 0.51 | 50% | 0.51 | 12.4 | −0.11 / −0.35 |
| bias_nocue | +0.028 | −0.01 | +0.06 | +0.02 | 0.54 | 25% | 0.53 | 14.1 | −0.01 / −0.49 |
| planted | **+0.422** | +0.25 | +0.60 | +0.52 | 0.43 | **100%** | 0.51 | 19.5 | **+0.42 / −0.68** |

### Per-seed within_al / crossed

| seed | bias_cue wal | cue× | nocue wal | nocue× | planted wal |
|------|-------------:|:----:|----------:|:------:|------------:|
| 851 | **+0.30** | yes | +0.06 | no | +0.38 |
| 852 | −0.12 | no | +0.02 | no | +0.17 |
| 853 | +0.04 | no | +0.01 | no | +0.17 |
| 854 | +0.07 | yes | +0.03 | yes | **+0.98** |

### cue − nocue contrasts

| seed | Δ(gA−gB) | ΔpoiB | Δwithin_al | cue crossed | nocue crossed |
|------|----------|-------|------------|-------------|---------------|
| 851 | +0.05 | −0.00 | **+0.23** | yes | no |
| 852 | −0.25 | +0.03 | −0.13 | no | no |
| 853 | −0.37 | +0.04 | +0.04 | no | no |
| 854 | −0.41 | −0.16 | +0.04 | yes | yes |
| **mean** | **−0.25** (1/4 +) | −0.02 | **+0.045** (3/4 +) | **2/4** | 1/4 |

### vs 18f / 18d

| | bias_cue wal | planted wal | cue−nocue Δ(gA−gB) |
|--|-------------:|------------:|-------------------:|
| 18d full (841–844, 20k) | +0.26 | +0.68 | **+0.33** (4/4 +) |
| 18f pilot (841–842, 12k) | **+0.45** | +0.51 | — (no nocue) |
| **18g full (851–854, 20k)** | **+0.07** | +0.42 | **−0.25** (1/4 +) |

## Readout

1. **18f did not replicate on new seeds.** bias_cue within_al drops from +0.45 (841–842) to +0.07 (851–854). Seed 851 alone (+0.30) carries most of the cue signal; 852 is anti-aligned.
2. **Planted remains the robust within-body control** (+0.42, 100% crossed), still with large replacement_al (+0.52).
3. **Cue vs nocue on within_al is weak** (+0.045, 3/4 +) — directionally consistent with a small cue benefit but far below 18f’s apparent effect size.
4. **Cue vs nocue on Δ(gA−gB) flips sign vs 18d** (−0.25 here vs +0.33 in 18d). Season-matched end biases are not reliably cue-selected on these founders/seeds.
5. A→B within_al for bias_cue is near zero/negative; B→A carries the positive mean — same asymmetry as 18f but with much smaller magnitude.

**Verdict:** Within-body seasonal switching for evolved (non-planted) bias_cue is **seed-dependent and not confirmed** as a stable cue-selected phenotype at 20k. Planted works. Do not treat 18f’s near-planted within_al as general.

## Next

- Optional: re-run 841–844 at 20k with nocue (same genomes as 18d/18f) to separate seed effect from horizon.
- Or accept planted as the within-body demonstration and treat bias_cue’s within_al as unreliable without planted init.
- No further knob chasing (mb/hom/lead already failed in 18e/18f).

Logs: `results/eco18g/b18g_{0..11}.pkl`; summary `full18g_summary.json`.
