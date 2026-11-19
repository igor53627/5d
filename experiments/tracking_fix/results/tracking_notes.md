# Lineage tracking fix: Price terms, heritability and lineages of Parts 10-17

**Status:** done on the published logs (no new simulation, no GPU cost). Issues #7 and #11.
**Scripts:** `recompute_tracking.py`, `validate_tracking.py` (this directory);
`src/lineage_tracking.py`, `src/price_equation.py`.
**Tables:** [tracking_recompute.md](tracking_recompute.md), [tracking_validation.md](tracking_validation.md).

## What was wrong

Two defects in the analysis, not in the simulations.

1. **Wrong ancestors.** `analyze_eco.parents` takes the nearest object in space, one snapshot
   (200 steps) earlier, as the ancestor. In that time a body moves 12-15 cells, and its
   nearest neighbour is about 19 cells away. The nearest object is then often another body.
2. **Mixed weights.** `analyze_eco.selection_diff` is an unweighted covariance over objects.
   The analysis scripts subtract it from the change of the mass-weighted mean and call the
   rest "transmission". The two terms do not add up to a Price identity.

Defect 1 is the large one. Defect 2 alone changes little (columns "S published" and
"S nearest" of the table are close).

## The fix

- `lineage_tracking.assign_ancestors` picks, among the objects within a gate radius, the one
  whose heritable traits are closest to the descendant's. Traits change little in 200 steps
  (median distance 0.01 for p0, p1; 0.1 for the 24 or 31 brain weights) and differ much more between
  bodies (median 1-3.5 for brain weights).
- `price_equation.price_terms` is mass-weighted. `selection + transmission` equals the change
  of the mass-weighted mean exactly; the code checks the identity to 1e-9 on every window.
- The gate is 20 cells without brain taxis and 30 cells with it. The 99th percentile of the
  ancestor distance is 16-17 cells without a brain and 26-29 with one.

## How the tracker was checked

- **Split-half** (`tracking_validation.md`). The brain weights are split at random into two
  disjoint halves, and each half picks the ancestor alone. In 4D the halves agree for
  95.1-98.3% of descendants in the evolving and frozen arms, and for 99.9-100% in the no-brain and blind arms. The nearest object in space agrees with the full fingerprint for only 24-31% of
  descendants in the evolving 4D arms.
- **Independent fingerprint** (column "S check"). The full fingerprint contains the analysed
  trait, which could bias transmission toward 0. The check identifies ancestors by traits
  that do not enter the analysed trait: the neutral p0, p1 for brain outputs, p1 alone when
  p0 is analysed. The check rule uses fewer traits than the full rule (1 against 2, or 2 against 26
  or 33), so it identifies a body less exactly: its division counts are about 2-4 times
  those of the full rule. A conclusion below
  is stated only where both rules agree in sign.
- **Part 18 pairs.** `metrics18c.match_bodies` (radius 8, exclusive pairs) agrees with the
  trait rule for 99.7-99.8% of its pairs in 18g and 18i. The within_al metric is not affected.

## Results

Lag 1000 steps, mean over seeds. "published" is `selection_diff`; "tracked" and "check" are
mass-weighted with the two trait rules. p is a one-sample t-test of the tracked term over seeds.

| part | arm, trait, phase | seeds | S published | S tracked | S check | T tracked | p (S tracked) |
|------|-------------------|------:|------------:|----------:|--------:|----------:|--------------:|
| 10 | sel, p0, growth | 3 | +0.055 | +0.062 | +0.062 | +0.000 | 0.018 |
| 10 | sel, p0, capacity | 3 | -0.015 | +0.006 | +0.004 | -0.001 | 0.021 |
| 12 | sel 96^4, p0, plateau | 5 | -0.028 | +0.004 | +0.002 | -0.001 | 0.0015 |
| 12 | neu 96^4, p0, plateau | 5 | +0.003 | -0.000 | +0.001 | -0.000 | 0.97 |
| 13 | sel 3D / 4D / 5D, p0, growth | 8 | +0.057 / +0.020 / -0.017 | +0.072 / +0.060 / +0.064 | +0.075 / +0.056 / +0.056 | +0.001 / -0.001 / -0.007 | all < 0.001 |
| 13 | sel 3D / 4D / 5D, p0, plateau | 8 | -0.033 / -0.025 / +0.009 | +0.008 / +0.006 / +0.007 | +0.005 / -0.001 / +0.003 | -0.002 / -0.001 / -0.003 | all < 0.001 |
| 15 | evo 4D, g_F, growth | 6 | -0.183 | +0.092 | +0.054 | -0.036 | 0.005 |
| 15 | frozen 4D, g_F, growth | 6 | -0.185 | +0.073 | +0.049 | -0.016 | 0.0009 |
| 15 | nobrain 4D, g_F, growth | 6 | +0.013 | -0.000 | -0.002 | -0.001 | 0.94 |
| 17 | evo, g_F, whole run | 4 | -0.101 | +0.034 | +0.032 | -0.006 | 0.010 |
| 17 | evo, g_P, whole run | 4 | +0.017 | -0.033 | -0.030 | +0.003 | 0.001 |
| 17 | nobrain, g_P, whole run | 4 | +0.005 | +0.003 | +0.001 | +0.001 | 0.091 |

Tracking numbers (mean per run):

| part | arm | nearest differs from tracked | divisions: nearest / tracked / check | parent-child r: nearest / check | effective lineages: nearest / tracked / check |
|------|-----|-----------------------------:|-------------------------------------:|--------------------------------:|----------------------------------------------:|
| 10 | sel / neu | 39.5% / 10.2% | 8909 / 1181 / 3714; 2066 / 371 / 804 | 0.44 / 0.72; 0.54 / 0.79 | 4.8 / 3.6 / 4.3; 9.8 / 11.9 / 11.4 |
| 13 | neu 3D / 4D / 5D | 12.3% / 25.3% / 28.7% | 560 / 37 / 99; 979 / 52 / 205; 966 / 66 / 260 | 0.56 / 0.72; 0.50 / 0.66; 0.47 / 0.74 | 5.3 / 11.2 / 9.3; 2.3 / 11.2 / 8.1; 1.7 / 7.7 / 5.3 |
| 13 | sel 3D / 4D / 5D | 14.3% / 48.8% / 47.1% | 632 / 73 / 181; 1708 / 150 / 561; 1449 / 268 / 724 | 0.47 / 0.77; 0.43 / 0.72; 0.46 / 0.62 | 3.6 / 3.4 / 3.7; 2.3 / 3.7 / 3.5; 1.5 / 2.3 / 2.1 |
| 15 | evo / nobrain 4D | 68.3% / 20.7% | 1279 / 314 / 640; 674 / 36 / 117 | 0.42 / 0.63; 0.51 / 0.93 | 2.3 / 3.1 / 2.5; 3.2 / 10.9 / 10.6 |
| 17 | evo / nobrain | 65.6% / 8.5% | 722 / 204 / 399; 86 / 1 / 2 | 0.47 / 0.78; 0.54 / 0.48 | 1.4 / 2.2 / 2.0; 2.5 / 10.7 / 9.7 |

## What changes in the published statements

1. **Parts 15 and 17: the selection term changes sign.** The notes read a negative S for g_F
   (and a positive S for g_P) as "selection acts inside bodies, the between-body term is small
   or opposite". With correct ancestors, g_F rises and g_P falls through selection **between**
   bodies, in every 4D seed, under both trait rules. The transmission term is small and has
   the opposite sign (-0.036 for g_F in Part 15, p = 0.048). The data give no support for
   within-body selection as the driver in 4D.
2. **Part 15 "evolving = frozen" stands and gets a mechanism.** The frozen arm has the same
   positive selection term (+0.073), so the response is sorting of founder variation.
3. **Parts 12 and 13: the negative plateau S is an artefact.** The published value is -0.03.
   The full rule gives +0.004 to +0.008 in every arm. The independent rule gives +0.002 in
   Part 12 (4 of 5 seeds positive), +0.005 in 3D, +0.003 in 5D and -0.001 in 4D 56^4. Both
   terms are small. Weak selection on the plateau has support in 3D and 5D; in 4D the sign
   depends on the rule, so the mechanism of the slow rise of mean p0 is not resolved there.
4. **Part 13: selection in growth does not depend on dimension.** S is +0.06 to +0.07 in 3D,
   4D and 5D. The published values (+0.057, +0.020, -0.017) fell with dimension only because
   tracking errors rise with dimension.
5. **Part 13: lineage loss with dimension is mostly an artefact.** Neutral runs keep 11.2 of
   12 founder lineages (effective number) in 3D and in 4D, and 7.7 in 5D (9.3, 8.1 and 5.3
   with the independent rule). Published: 5.3, 2.3 and 1.7.
6. **Division counts were inflated 2 to 19 times** in 3D-5D (the factor depends on the trait
   rule), because a wrong ancestor with two
   descendants counts as a division. The order 3D < 4D < 5D stays. The published merge rates
   use the same rule and are inflated too. "Merges about equal divisions on the plateau" and
   "response per generation falls with dimension" (Part 13) are not supported by these logs.
7. **Parent-child correlation is higher than published:** 0.62-0.86 with the independent rule
   in Parts 10-13, against 0.41-0.56. The value 0.96-0.98 from the full fingerprint is circular for p0 (the
   fingerprint contains p0) and must not be quoted.
8. **Unchanged:** the growth-phase selection term of Part 10 (+0.055 published, +0.062), all
   neutral and no-brain controls (S near 0), and every result that does not use ancestors
   (population size, mean traits, diet, response curves, poison share).

## What is not resolved

- **5D runs with a brain (Parts 15b and 16).** After fusion the bodies are one genotype, and
  the two halves of the weights agree for only 66-77% of descendants. The two trait rules give
  opposite signs (g_F: +0.125 and -0.081 in 15b; g_R: -0.082 and +0.132 in Part 16). The
  published statement "between-individual selection favours repulsion, transmission drags g_R
  down" has no support, and neither has its opposite.
- **No-transport runs (Part 10 `notr`).** The heritable field does not move with the mass, so
  there is no fingerprint; these rows still use the nearest object.
- **Fusion.** Each descendant has one ancestor. A fusion credits the fused body to one
  partner and appears as selection for that partner with an opposite transmission term.
- **Not recomputed:** Parts 9 and 14 (heritability, lineages), Parts 18 and 18b (Price S),
  merge rates, and the Part 13 response per generation. Part 18g/18i logs are sparse: the
  nearest object differs from the trait rule for 3% of descendants there.
- **Sensitivity, one-off checks not in the repository:** for Part 17 `evo`, gates of 22, 30
  and 45 cells give S(g_F) = +0.022, +0.034, +0.036 and S(g_P) = -0.025, -0.033, -0.034. Lags
  of 1, 2, 5 and 10 snapshots give S(g_F) = +0.008, +0.014, +0.034, +0.062 (same sign, the
  size grows with the lag).
- The simulation defects in issue #10 (exchange of p between near bodies by homogenization,
  MUSCL positivity) are separate. They act in the engine and are not corrected here.

## How to rerun

```bash
huggingface-cli download igor53627/5d --repo-type dataset --local-dir DATA \
  --include "part10/*" "part12/*" "part13/*" "part15/*" "part16/*" "part17/*" "part18/*"
python experiments/tracking_fix/recompute_tracking.py DATA     # about 6 minutes on one core
python experiments/tracking_fix/validate_tracking.py DATA
```
