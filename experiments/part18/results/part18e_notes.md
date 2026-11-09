# Part 18e — within-body switching pilot

**Status:** pilot done (`b18ep`, ~$4.67). Full run not recommended yet.
**Parent:** Part 18d full (`9dcc2d3`): bias_cue crossed 4/4 via season-matched biases, but mean within +0.05 vs planted +0.29 (selection mostly replacement).

## Goal

Raise **within-body** Δg at season switches for `bias_cue`, cutting the payoff of replacement.

## Knobs

| knob | value | why |
|------|-------|-----|
| `dec` (existing) | 1e-4 → **2.5e-5** (`bias_cue_lowdec`) | Lower mass decay → fewer deaths → less replacement. |
| `cue_lead` (**new**, default 0) | **400** (`bias_cue_lead`) | Brain sees σ(t+lead); food/poison stay on σ(t). Lets cue-gated pathways prepare before the flip. Legacy when 0. |
| baseline | 18d world, `mut_bias=0.05` | Control. |
| planted | 18d planted | Positive within-body reference. |

No long CPU probes. Unit test: `tests/test_cue_lead.py`.

## Pilot design

2 seeds (841, 842) × 4 arms × 8k steps, H100 ecolist, prefix `b18ep`. One food-season switch (A→B at t≈4200). Cost ≈ **$4.67** (1.18 GPU-h).

## Results

Raw `within` is signed with Δ(g_A−g_B). On a single A→B switch, adaptive within is **negative**. Report **within_aligned = within × sign(σ_after)** (+ = toward new season).

| arm | within_al | repl_al | matched | raw within | crossed | poiB | mass | gA / gB |
|-----|----------:|--------:|--------:|-----------:|--------:|-----:|-----:|---------|
| bias_cue | **+0.135** | +0.233 | 0.50 | −0.135 | 0% | 0.43 | 14.6 | −0.37 / −0.03 |
| bias_cue_lowdec | −0.123 | +0.040 | 0.56 | +0.123 | 0% | 0.45 | 23.1 | −0.32 / −0.00 |
| bias_cue_lead | −0.015 | +0.001 | **0.71** | +0.015 | 50% | 0.48 | 13.8 | −0.11 / −0.36 |
| planted | +0.073 | +0.727 | 0.29 | −0.073 | 0% | 0.46 | 16.4 | −0.02 / +0.35 |

18d full (for scale): bias_cue within_al ≈ +0.26 (n=16 switches); planted ≈ +0.68.

### Readout

- **lowdec:** replacement shrinks as intended (repl_al +0.04 vs +0.23), mass rises (~23), but **within_aligned goes negative** — bodies persist without rewriting biases toward the new season. Not a win for within-body switching.
- **cue_lead:** matched mass up (0.71), replacement ~0, within_aligned ≈ 0. No clear gain vs baseline on this short A→B window. One seed crossed (gA+/gB−) by end.
- **planted** under-shoots its 18d within_al (short horizon, low match 0.29); still mostly replacement_al.
- Baseline bias_cue already shows moderate within_al (+0.14) plus replacement.

**Verdict:** neither variant beat baseline on within_aligned. **Do not go full** on these settings.

## Next

1. Longer seasons / more switches (12–20k, Ts≥8000) so within-body rewrite has time and both A→B and B→A are visible.
2. Probe `cue_lead` on planted genomes (sanity that the knob moves within_al).
3. Consider higher `mut_bias` or lower `hom` for faster within-body bias RW; avoid lowdec alone.
4. Irregular season lengths (new schedule) only if longer+lead still fails.

Logs: `results/eco18e/b18ep_{0..7}.pkl`; summary `pilot18e_summary.json`.
