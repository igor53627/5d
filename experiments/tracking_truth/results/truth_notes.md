# Ancestor rules against measured ancestors

**Status:** six CPU runs of the Part 13 world with tracer fields, done 2026-11-20. Issue #13.
**Scripts:** `run_truth.py`, `analyze_truth.py` (the directory above); `src/lineage_tracer.py`.
**Tables:** [truth_summary.md](truth_summary.md), `truth_summary.json`.
**Raw files:** six `truth_*.json` files, 1.4 MB. They are not in this repository. Their place is
the folder `tracking_truth/` of the Hugging Face dataset `igor53627/5d`; `DATA.md` lists the
folder from the commit that follows the upload.

## Why

The published logs hold no true ancestor. The tracking fix
([tracking_notes.md](../../tracking_fix/results/tracking_notes.md)) could only compare ancestor
rules with each other, so its rates are rates of disagreement. These runs measure the ancestor.

## Method

- Every object of a snapshot gets its own tracer fraction. The fractions move with the mass
  flux of `fl_eco3` and nothing reads them. The simulated state is bitwise the state of
  `fl_eco3.run_chunk` (`tests/test_lineage_tracer.py`).
- One snapshot (200 steps) later, the tracer content of an object is the share of its mass
  that comes from each earlier object. The "main ancestor" is the object with the largest
  share.
- World: the Part 13 runs `neu_*_s201` and `sel_*_s201` (soliton, food, mutation, 12 founder
  placements), with a smaller box and new seeds.

| runs | box | steps | seeds | objects per snapshot |
|------|-----|------:|-------|---------------------:|
| 3D neutral, 3D selection | 64^3 | 20,000 | 301, 302 | 19-22 |
| 4D neutral, 4D selection | 40^4 | 4,000 | 301 | 14 |

The 4D runs cover the growth phase only, with one seed and 281 and 292 descendants. A step
with tracers costs about 9 times a plain step in 3D (16 tracer channels; not measured in
4D); the six runs took 2.2 to
3.5 hours each on shared CPU cores.

## Results

"nearest" is the nearest object in space (the published analysis). "tracked" is the object
with the closest (p0, p1) inside a 20-cell gate. "check" is the same with p1 alone.

| run | seed | descendants | nearest right | tracked right | check right |
|-----|-----:|------------:|--------------:|--------------:|------------:|
| 3D neutral | 301 | 2110 | 61.2% | 99.9% | 97.1% |
| 3D neutral | 302 | 2214 | 61.0% | 99.6% | 97.3% |
| 3D selection | 301 | 2057 | 73.3% | 99.7% | 96.0% |
| 3D selection | 302 | 1913 | 79.8% | 99.7% | 98.4% |
| 4D neutral | 301 | 281 | 69.0% | 97.5% | 92.5% |
| 4D selection | 301 | 292 | 70.2% | 98.6% | 93.2% |

1. **The (p0, p1) rule picks the main ancestor** for 99.6-99.9% of descendants in 3D and for
   97.5% and 98.6% in the two 4D runs.
2. **Where the nearest object and the (p0, p1) rule differ, the trait rule is right** for
   98.9-99.8% of those descendants in 3D and for 94.3% and 96.6% in 4D. The nearest object is
   right for 0.0-0.9% in 3D and for 2.3% and 3.4% in 4D. In these runs the rate of
   disagreement (20-39%) is thus close to the error rate of the nearest rule (20-39% in 3D,
   30-31% in 4D).
3. **One ancestor per body is adequate at 200 steps.** At least 99.6% of descendants get 90%
   or more of their mass from one earlier object. Mass from outside every earlier object is
   1.4-1.8% of an object.
4. **Price terms of p0** (lag 1000 steps, means over the first and the second half of a run;
   the measured term multiplies the measured 200-step shares over five snapshots, see
   "Limits of the chained numbers"):
   - The (p0, p1) rule is within 0.013 of the measured selection term in all 12 half-runs,
     with the same sign wherever the measured term is above 0.003 in size.
   - The nearest rule differs by up to 0.079. In the neutral runs, where the measured term is
     within 0.003 of zero, it gives terms of 0.008 to 0.049 in size. It has the opposite sign
     in both halves of 3D selection seed 302 (-0.030 and -0.011 against +0.037 and +0.043)
     and in the second half of seed 301 (-0.035 against +0.030).
   - The p1 rule differs by up to 0.068 (4D neutral, second half: -0.070 against -0.002).
   - In the second half of the two 3D selection runs the (p0, p1) rule gives +0.021 and
     +0.030 against a measured +0.030 and +0.043. Two runs do not show whether this is a
     bias.
5. **Divisions per run:** measured 8-13; (p0, p1) rule 10-19; p1 rule 24-80; nearest rule
   46-266.
6. **Effective founder lineages at the end** (the measured number multiplies the measured
   200-step shares over the whole run): measured 10.3 and 8.4 (3D neutral), 3.6 and 2.9
   (3D selection), 7.2 and 6.3 (4D, after 4000 steps). The (p0, p1) rule gives 10.3, 8.0,
   3.9, 3.7, 7.4 and 7.0. The nearest rule gives 1.0 to 2.4 in every run.

## What this supports, and what it does not

- For the world of Parts 10-13 (no brain, two heritable traits), the direction of the
  corrections in the tracking fix has direct support: fewer divisions, more surviving founder
  lineages, a selection term near zero in neutral runs, and a positive selection term in
  selection runs.
- Of the two trait rules, the (p0, p1) rule is the closer one to the measured ancestors here.
  The p1 rule gives 2 to 6 times the measured divisions and selection terms of either sign
  in neutral runs. The tracking notes state a conclusion only where both rules agree in sign;
  that condition is strict.
- Not covered:
  - Runs with a brain (Parts 15-18b). The tracer works with `fl_eco3` only, so the sign
    change of the selection term in Parts 15 and 17 still rests on the trait rule alone.
  - 5D and 6D, and the 4D plateau (the 4D runs stop at 4000 steps).
  - The published box sizes. The 3D runs here hold 7.3-8.4 objects per 100,000 cells against
    4.3-5.5 in the published 3D runs, and the two rules differ for 20-39% of descendants
    against 12-14% there. The 4D density (0.54-0.56) is inside the published range
    (0.42-0.58). The densities are a one-off calculation from `truth_summary.json` and
    `tracking_recompute.json`. The error rate of the nearest rule in the published 3D runs is therefore
    not measured; it is probably lower than here.
- Each condition has one or two seeds. The numbers are measurements of these runs, not
  estimates with an interval.

## Limits of the chained numbers

- `run_truth.py` gives every body a new tracer at each snapshot. A share is thus measured over
  200 steps only.
- The Price terms at 1000 steps (result 4) and the effective founder lineages (result 6)
  multiply these 200-step shares. The rules are chained in the same way.
- The product is exact only when two conditions hold. First, the mass that a body passes on
  has the same mix of origins as the whole body. Second, no mass passes from one body to
  another through background (see below).
- The first condition fails when a body holds mass of two origins and one part grows more or
  splits off. Example: a body that is a 50/50 merge stays 50/50 in the product, also when only
  one side grows.
- For at least 99.6% of descendants, the mass from outside the main ancestor is at most 10%
  in one 200-step interval (result 3). This does not show that merges are rare. It does not
  bound the number of bodies that hold mass of two origins after many intervals, because a
  merged body and its descendants keep the mix. It does not bound the error of the product.
- Mass that is outside every detected body at a snapshot is background, and background is
  credited to no ancestor. Ancestry that leaves every body at one snapshot and enters a body
  later is thus lost, also when every body passes on a uniform mix. In one interval 1.4-1.8%
  of the mass of an object comes from background, as a mean over objects (result 3). As an
  illustration: if every object lost that share in every interval, the product would credit
  about 91-93% of the mass after five intervals and about 16-24% after 100 intervals. The
  mean alone does not fix these numbers, and the credited mass is not measured. The Price
  terms and the lineage number use the credited mass only.
- The size of the two errors is not measured: that needs tracers that stay for the whole span.
- Results 1, 2, 3 and 5 use single 200-step shares and do not depend on this.

## Records

- The `git_revision` field of the six files says `6d71d28`. The script read the revision at
  the end of a run, and the branch moved while the runs were in progress. A later rebase of
  the branch replaced that hash; it was the commit "Recompute Parts 14, 18 and 18b with
  ancestors identified by traits". The runs used the
  working tree of 2026-11-19 21:59; the later changes do not touch the simulation path
  (`run_truth.py`: one argument check; `price_equation.py`: used by the analysis only).
  `run_truth.py` now reads the revision at the start.
- `analyze_truth.py` takes about 1 second for the six files.
- The six runs used `lineage_tracer.py` before one correction. The copy that carries the
  tracers then had the decay of the world, and the engine clears the traits of a cell whose
  mass falls to zero. A cell that died in the copy and lived in the simulation lost its
  fractions. The copy now runs without decay. With the decay of these worlds (0.0003 per
  step before the crowding factor) the case needs a cell at the mass limit of 1e-6 or
  crowding of 5 times the reference. One comparison shows no effect: `sel_3D96_s201`, box
  48, 4000 steps, seed 301, 240 descendants, with the old and the new copy gives equal
  shares, background and fraction sum error to five decimals (a one-off run, not in the
  repository). The six runs are not repeated, and the comparison does not cover their
  plateau or 4D.

## How to rerun

```bash
cd experiments/tracking_truth
python analyze_truth.py DATA                      # DATA: a local copy of the dataset
python run_truth.py neu_3D96_s201 64 20000 301 DATA/tracking_truth   # about 2.2 hours on CPU
python run_truth.py sel_4D56_s201 40 4000 301 DATA/tracking_truth    # about 3.5 hours on CPU
```
