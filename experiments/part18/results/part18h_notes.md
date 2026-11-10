# Part 18h — competition pilot (planted vs bias_cue)

**Status:** pilot done (`b18hp`, ~$5.24).
**Parent:** 18g (`d42e39a`): within_al for evolved bias_cue was seed-dependent. Question: is within-body plasticity selected against, or just hard to find?

## Design

A heritable **lineage tag** `p[iL]` with `K=KB+1=44`. Unused by the brain or feeding; not mutated. Default `K=KB` is legacy (unit test: `tests/test_lineage_tag.py`).

| | |
|--|--|
| Seeds | **861, 862** → founders 621, 622 |
| Arms | `mixed`: 6 planted (tag +1) + 6 bias_cue (tag −1), interleaved; `control`: 12 bias_cue, 6/6 tags (neutrality) |
| Steps | 20000 (4 switches, 18g schedule), cue=1, mut_bias=0.05, krep=90 |
| Cost | **~$5.24** (1.33 GPU-h) |

## Results

| arm / seed | frac+ init | frac+ end | state | wal | planted/plus own wal | cue/minus own wal | crossed | mass |
|------------|----------:|----------:|-------|----:|---------------------:|------------------:|:-------:|-----:|
| mixed 861 | 0.50 | **0.56** | poly | +0.54 | **+1.11** | −0.02 | yes | 16.2 |
| mixed 862 | 0.50 | **0.76** | poly | +0.37 | **+1.77** | −0.10 | no | 15.4 |
| control 861 | 0.50 | 0.53 | poly | +0.10 | +0.20 | +0.40 | yes | 13.0 |
| control 862 | 0.50 | 0.55 | poly | −0.01 | −0.45 | −0.05 | no | 11.1 |

Mean end frac+: mixed (planted) **0.66**, control **0.54**. No fixation/extinction by 20k (`t_fix=None`).

## Readout

- **Planted is not displaced.** Tag+ (planted) mass fraction rises or holds in both mixed seeds (0.50→0.56, 0.50→0.76). The control tag stays near 0.5 (0.53, 0.55), so the mixed rise is not a tag artifact.
- Planted bodies keep a large **within_own_aligned** (+1.1, +1.8); the cue half does not. Plasticity remains visible in the planted lineage when it coexists.
- 20k is not long enough for fixation. Direction is “planted holds/gains,” not “selected against.”

**Verdict:** Within-body switching looks **hard to find, not selected against** — once planted switchers are present, they at least persist and tend to gain mass share.

## Next

A longer run (or more seeds) could test fixation. No need to change ecology knobs for this question.

Logs: `results/eco18h/b18hp_{0..3}.pkl`; summary `pilot18h_summary.json`.
