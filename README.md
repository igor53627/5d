# Hyperlife

Evolving artificial life in 4D/5D Flow-Lenia.

Hyperlife is a continuous cellular automaton (Flow-Lenia) in four and five spatial dimensions. The transport step conserves mass; feeding adds mass and decay removes it. Each cell is a chaotic Rössler oscillator, so its local dynamics sit on the period-doubling cascade (the Feigenbaum route). Localized gliders and organisms self-organize, and open-ended evolution acts on heritable parameters and small neural micro-brains that travel with the mass. The experiments probe foraging, poison avoidance, and seasons, and they measure change with the Price equation, heritability, and founder lineages.

Python modules keep their historical names (`feigenbaum_lenia.py` and the `fl_*` engines). The installable distribution name is `hyperlife`.

## Why 5D?

Classic Lenia is a two-dimensional world. Here the lattice is four- or five-dimensional. Early moving-structure runs used five-dimensional grids, including 5D 24⁵ (the `5D24` sweeps and the Part 6 controls). The main evolution experiments are in 4D, because a 5D lattice large enough for a population is much more expensive to step. The question is how life-like self-organization and evolution behave above three dimensions. The repository is named `5d` for that reason.

## Results so far

Numbers below are copied from the part notes. There is no Part 11 notes file; that folder is an engine-timing study.

- **Part 5** ([notes](experiments/part05/results/part5_notes.md)). Drift-Lenia speed tracks kernel asymmetry and falls below $5\times10^{-4}$ as soon as asymmetry is quenched to 0, with no hysteresis. Flow-Lenia plus a phase drive $\chi$ still moves at asymmetry 0 (speed about 0.003–0.012); $\chi=0$ stops it. A short evolution strategy (three jobs) reached fitness 0.42, about 0.007 cells/step.
- **Part 6** ([notes](experiments/part06/results/part6_notes.md)). CMA-ES on 5D $20^5$ found a rigid self-moving soliton (S1) at asymmetry 0. On 5D $32^5$ it travels 216 cells in 3400 steps ($v\approx0.0635$, straightness 0.99–1.0); on 4D $64^4$, $v\approx0.047$–0.049. Controls on 5D $24^5$: $\chi=0$ still moves at $v\approx0.061$, $\beta=0$ at 0.056–0.060, and pure flow nearly stops ($v\approx0.0004$). Directions sit near lattice diagonals.
- **Part 7** ([notes](experiments/part07/results/part7_notes.md)). `flow_iso_muscl` was kept for isotropy: 5D S1 Kolmogorov–Smirnov 0.84 → 0.25, speed 0.0639 → 0.0771. On 4D $64^4$, collisions reclassify as 10 capture-then-re-emission, 11 merge, 2 bounce with mass exchange, and 1 capture with exchange. On 4D $80^4$, two survivors are common. Verified S1iso on 5D $32^5$: 259 cells in 3400 steps, $v\approx0.076$, straightness 0.994–0.998.
- **Part 8** ([notes](experiments/part08/results/part8_notes.md)). A search for two solitons colliding into three never met a strict criterion of three objects each at least 0.6 of the unit mass (mass conservation). The best candidate, R2g5, made three objects of at least 0.49 in 8 of 19 four-dimensional configurations, and all 24 daughters survived alone and moved. The parameter $c$ inside daughters stays about 2.8–2.96 and is slaved to the mass field, so that channel carries nothing heritable. Cascades of six solitons on 4D $96^4$ did not grow the population. In 5D $40^5$, 0 of 3 configurations replicated.
- **Part 9** ([notes](experiments/part09/results/part9_notes.md)). A food field lets a soliton grow and divide. Scarce food ($F_0=0.06$) keeps mass about 3–7 unit masses with repeated divisions; abundant food fills the box with one blob; no food leaves a single soliton at mass 1. After a dilution bug in the heritable parameter $p$ was fixed, selection raised mean $p_0$ above the neutral control in 7 of 7 paired runs. Parent–child correlation of $p_0$ was 0.76 with mass transport (35 pairs) and −0.16 without it. Mean $p_0$ also rises when nothing divides, which the notes read as cell-level selection inside one object. A 5D single-start run divided twice.
- **Part 10** ([notes](experiments/part10/results/part10_notes.md)). In the tuning sweep, repulsion between bodies cuts mergers to almost none (0–3 at krep 10). On 4D $96^4$ (three seeds, 25k steps) final counts were selection 479/568/586, neutral 348/343/345, and no-transport 343/341/342. Selection shifted mean $p_0$ by +0.54/+0.60/+0.59 against a neutral shift near 0 (Welch $p=2\times10^{-5}$). The Price selection term is positive while the population grows and about 0 at capacity. Heritability of $p_0$ is 0.42–0.55 with transport and 0.01–0.03 without it. A 4D $128^4$ selection run ended at 1634 individuals (heritability 0.44). The 5D $40^5$ selection run is the first self-reproducing population in 5D in this project (6 → 61 individuals, mean $p_0$ 0.26 → 0.80); that part has no 5D neutral control, and the selection term was weak (+0.013).
- **Part 12** ([notes](experiments/part12/results/part12_notes.md)). At capacity on 4D $96^4$ (five seeds), selection's Price term on the plateau is slightly negative (−0.028, $p=0.013$ against 0) while mean $p_0$ still creeps up through transmission. Neutral diversity stays flat (standard deviation of $p_0$ about 0.48, selection term about 0). Two resources keep both specialist types and a high standard deviation (about 0.60). A spatial niche produced no local adaptation in the pilot: individuals cross a niche faster than selection acts. In 5D $40^5$ (one seed each), selection moved mean $p_0$ from 0.26 to 0.79 and the neutral run stayed at 0.26.
- **Part 13** ([notes](experiments/part13/results/part13_notes.md)). Same rules and food, eight seeds: 3D $96^3$, 4D $56^4$, and 5D $40^5$ reach a neutral plateau of about 40–45 individuals. In 2D, division is about 0.006–0.036 per individual per 1000 steps. With 12 founders that 40–45 count stays out of reach; with 48 founders the count peaks near 97 (neutral plateau mean 67.7) and then declines. The selection response after 5k steps is about 0.39 in 3D, 0.39 in 4D, and 0.36 in 5D (no trend with dimension); the neutral response is about 0. The division rate rises with dimension, and effective founder lineages fall (3D 3.6–5.3, 4D 2.3, 5D 1.5–1.7). Response per generation falls with dimension (3D 0.18, 4D 0.08, 5D 0.07).
- **Part 14** ([notes](experiments/part14/results/part14_notes.md)). A 6D soliton exists. On 6D $28^6$ (one seed, 20k steps) selection goes from 9 to about 16 individuals, the standard deviation of $p_0$ from 0.33 to 0.10, $\Delta p_0=0.17$, heritability 0.36. The neutral run stays near 10–11 individuals, heritability 0.42. Both end on one effective founder lineage. Founders overlap, so diversity is already low at the start.
- **Part 15** ([notes](experiments/part15/results/part15_notes.md)). A $3\to3\to3$ micro-brain is carried with the mass. On 4D $56^4$ (six seeds, 24k steps) evolved chemotaxis $g_F$ is 0.47 against −0.09 with the brain switched off; plateau counts are 59.5 against 47.7. Evolving and frozen weights agree on every reported metric ($p>0.1$): within 24k steps the change is sorting of the founders' standing variation. The Price term for $g_F$ during growth is −0.18 while the mean still rises, so the notes attribute the rise to within-body transmission. In one 5D $40^5$ seed the brain run fuses into a single object and later breaks up, with $g_R$ near −0.35; the no-brain control stays fragmented.
- **Part 15b** ([notes](experiments/part15/results/part15b_notes.md)). Response curves of the evolved brains are almost flat: a constant bias, "always go up the food gradient." Dependence of $g_F$ on inputs falls under evolution (4D standard deviation 0.13, founders 0.18). No single weight shifts consistently across seeds, while $g_F>0$ in all 18 brain runs. Clustering stays a continuum (BIC keeps falling through $K=8$). In the 5D run, $g_R$ turns negative before fusion and moves with $g_F$; between-individual selection on $g_R$ is positive ($S=+0.21$) and transmission is −0.28. The notes treat the 5D fusion link as a one-seed hypothesis.
- **Part 16** ([notes](experiments/part16/results/part16_notes.md)). Fusion (largest object above half the mass) occurs in 3 of 4 evolving 5D runs, in 1 of 2 runs whose repulsion is clamped at $g_R\ge0$, and in 0 of 2 runs with no brain. The clamp does not stop fusion, so a negative $g_R$ is not required. Shared chemotaxis is a hypothesized cause: the no-brain control also turns off brain-controlled repulsion, so it does not isolate chemotaxis. In three of the four fusions the giant later breaks apart. Samples are small (12k steps).
- **Part 17** ([notes](experiments/part17/results/part17_notes.md)). With separate good food and poison (4D $56^4$, four seeds, poison cost 2), avoidance is in place by 2–3k steps: $g_P$ reaches −0.66 to −0.93 in every seed. Poison is 3.2% of the diet against 18.3% with no brain ($p=7\times10^{-4}$). A blind brain that sees only the mixture collapses below the no-brain control. The response is again an almost constant bias (nonlinear fraction at most 0.01). The Price split says $g_P$ falls through within-body transmission ($S=+0.017$, transmission −0.047).
- **Part 18** ([notes](experiments/part18/results/part18_notes.md)). Seasons that swap which food is poisonous are too harsh on the original parameters: every population is extinct by 16k steps. A brain extends lifetime by about 1.4–1.8× (memory against no brain, $p=6\times10^{-4}$). Memory and a direct season cue do not separate from each other ($p\ge0.2$). What evolves is a fixed "A is good" reflex. Behaviour lags the 2000-step season by about 1200–1400 steps.
- **Part 18b** ([notes](experiments/part18/results/part18b_notes.md)). A milder world and a warm start from Part 17 brains keep all 16 populations alive through 24k steps. No condition switches inside the bad season: the inherited "toward A" reflex lasts the whole 8000-step B season, and the poison fraction of the diet in the B season rises from 0.70 to 0.80–0.81. Memory, no-memory, and cue runs match ($p\ge0.2$); weights on the new inputs only reach about 0.02. The notes read this as evolution stalled: founders start nearly identical, bodies fuse, and the Price selection term is about 0.
- **Part 18c** ([notes](experiments/part18/results/part18c_notes.md)). Setup only; no population has been run. On the four Part 18b founders, noise of standard deviation 0.5 on the new inputs does not produce a cue-conditional sign flip (0 of 48 genomes). The output bias locks $g_A-g_B$. The 18c specs keep the 18b world, use four new seeds, and compare a low-diversity control, that input-noise arm, an oracle cue on the same genomes, a planted switching control, and no brain. Repulsion at krep 10 is about a tenth of food taxis before the speed cap, so raising it is not used as a fusion fix.
- **Part 18d** ([notes](experiments/part18/results/part18d_notes.md)). Mutable output biases $g_A/g_B$ (`mut_bias=0.05`) let evolution learn cue-switching. Full assay (4 seeds × 5 arms × 20k, ~$25): `bias_cue` ends A+/B− on 4/4 seeds vs 2/4 for `bias_nocue`; cue−nocue $\Delta(g_A-g_B)=+0.33$ (all seeds positive); B-season poison lower with cue in 3/4. Inputs baseline still fails. No early fusion. Oracle cue selects for seasonal switching across seeds.
- **Part 18e** ([notes](experiments/part18/results/part18e_notes.md)). Pilot to raise within-body seasonal switching for `bias_cue` (18d within_al≈+0.26 vs planted≈+0.68; selection mostly replacement). Tried lower decay (`dec` 2.5e-5) and new `cue_lead=400` (brain sees future σ; default 0=legacy). Neither beat baseline on σ-aligned within in an 8k/1-switch pilot (~$4.7); lowdec hurt alignment. Full run not recommended yet.
- **Part 18f** ([notes](experiments/part18/results/part18f_notes.md)). Longer-horizon pilot (12k, both season directions): baseline `bias_cue` within_aligned≈+0.45 ≈ planted +0.51. Higher `mut_bias` (0.10) and weaker `hom` (0.0125) hurt; `cue_lead` on planted fails sanity (−0.02). No full on these variants; optional confirmatory full of bias_cue vs nocue/planted.
- **Part 18g** ([notes](experiments/part18/results/part18g_notes.md)). Confirmatory full (seeds 851–854 × {bias_cue, bias_nocue, planted} × 20k): 18f’s near-planted within_al for bias_cue (+0.45) did **not** replicate (+0.07); planted stays robust (+0.42). Cue−nocue Δwithin_al only +0.05 (3/4 +); Δ(gA−gB) −0.25 (unlike 18d +0.33). Within-body switching for non-planted bias_cue is seed-dependent.
- **Part 18h** ([notes](experiments/part18/results/part18h_notes.md)). Competition pilot: 50/50 planted vs ordinary `bias_cue`, distinguished by a behavior-neutral lineage tag (`K=KB+1`). 18g season schedule, 20k, seeds 861–862, plus a tag-neutrality control (both halves `bias_cue`). Planted mass fraction rose 0.50→0.66 mean (0.56, 0.76) vs control ~0.54; no fixation. Planted within_own_aligned stayed large. Within-body switching looks hard to find, not selected against. Cost ~$5.24.
- **Part 18i** ([notes](experiments/part18/results/part18i_notes.md)). Four new seeds (855–858) on the 18g assay, pooled to 8 seeds/arm. bias_cue within_al +0.20 [0.07, 0.34]; nocue +0.03 [−0.01, 0.07]; planted +0.45 [0.24, 0.68]. Cue−nocue Δwithin_al +0.17 [0.04, 0.32] (CI excludes 0); Δ(gA−gB) −0.00 [−0.23, 0.25] (ns). Cue raises within-body switching, but moderately and below planted; 18d’s strong Δ(gA−gB) does not hold. Cost ~$15.05.
- **Part 19** ([notes](experiments/part19/results/part19_notes.md)). Predator–prey smoke/pilot on the Part 15 world (4D L=48, 8 prey + 4 predators, 6k steps, ~$3.23). A second mass field consumes nearby prey. No immediate mass extinction on any of 2 seeds × 3 kpred values. `kpred=0.015–0.05` keeps prey bodies and predator mass; `kpred=0.15` wipes prey bodies by 2–3k. At most one half-cycle in 6k, not a period claim.
- **Part 19b** ([notes](experiments/part19/results/part19b_notes.md)). Confirmatory 20k run, 4 new seeds (193–196) × {kpred=0.015, 0.05, prey-only control}. Mid (0.05) loses predator bodies by ~5k on all four seeds (time-to-loss 5050 [4700, 5400]); prey usually recover. Low (0.015) holds both as bodies to 20k on 2/4 seeds (plus one loss at 19.8k); seed 195 collapses. Prey-only control booms (M_A≈26). No reliable oscillation period. Cost ~$20.27.


## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
# GPU build of JAX (optional): pip install -U "jax[cuda12]"
```

## Running

```bash
# Quick CPU demo (3D, L=8; writes animation.mp4)
python -m feigenbaum_lenia --N 3 --L 8 --steps 80 --every 10 --out demo_run

# Full simulation (4D, L=24, about 6 minutes on CPU)
python -m feigenbaum_lenia --out full_run

# GPU, after `pip install -U "jax[cuda12]"`
JAX_PLATFORMS=cuda python -m feigenbaum_lenia --out gpu_run
```

## Project Structure

```
src/                 Core library: models, kernels, dynamics
experiments/         Per-part experiment scripts and results
  part05/            Asymmetry continuation and swimmers
  part06/            Long 5D CMA-ES
  ...
  part18/            Seasonal environments
  part19/            Predator–prey
figures/             PNG figures by part
videos/              MP4 videos by part
docs/                Notes and older reports
archive/             Early searches, benchmarks, reservoir experiments
```

## Table of Contents: Parts 1–19

| Part | Topic | Notes | Key Figure |
|------|-------|-------|------------|
| 5 | Asymmetry continuation and swimmers | [part5_notes.md](experiments/part05/results/part5_notes.md) | [glider_cont_bifurcation.png](figures/glider_cont_bifurcation.png) |
| 6 | Long 5D CMA-ES | [part6_notes.md](experiments/part06/results/part6_notes.md) | [glider_es5_fitness.png](figures/glider_es5_fitness.png) |
| 7 | Collision experiments | [part7_notes.md](experiments/part07/results/part7_notes.md) | [glider_part7_coll4D.png](figures/glider_part7_coll4D.png) |
| 8 | Replication | [part8_notes.md](experiments/part08/results/part8_notes.md) | [glider_part8_replication.png](figures/glider_part8_replication.png) |
| 9 | Growth and selection | [part9_notes.md](experiments/part09/results/part9_notes.md) | [glider_part9_growth.png](figures/glider_part9_growth.png) |
| 10 | Ecology experiments | [part10_notes.md](experiments/part10/results/part10_notes.md) | [glider_part10_population.png](figures/glider_part10_population.png) |
| 12 | Population dynamics | [part12_notes.md](experiments/part12/results/part12_notes.md) | [glider_part12_population.png](figures/glider_part12_population.png) |
| 13 | Dimension comparison | [part13_notes.md](experiments/part13/results/part13_notes.md) | [glider_part13_metrics.png](figures/glider_part13_metrics.png) |
| 14 | 6D experiments | [part14_notes.md](experiments/part14/results/part14_notes.md) | [glider_part14_dims.png](figures/glider_part14_dims.png) |
| 15 | Micro-brain metrics | [part15_notes.md](experiments/part15/results/part15_notes.md) | [glider_part15_timeseries.png](figures/glider_part15_timeseries.png) |
| 15b | Brain encoding analysis | [part15b_notes.md](experiments/part15/results/part15b_notes.md) | [glider_part15b_weights.png](figures/glider_part15b_weights.png) |
| 16 | Fusion catastrophe | [part16_notes.md](experiments/part16/results/part16_notes.md) | [glider_part16_fusion.png](figures/glider_part16_fusion.png) |
| 17 | Poison food (4D) | [part17_notes.md](experiments/part17/results/part17_notes.md) | [glider_part17_dynamics.png](figures/glider_part17_dynamics.png) |
| 18 | Seasonal environments | [part18_notes.md](experiments/part18/results/part18_notes.md) | [glider_part18_dynamics.png](figures/glider_part18_dynamics.png) |
| 18b | Memory and adaptation | [part18b_notes.md](experiments/part18/results/part18b_notes.md) | [glider_part18b_dynamics.png](figures/glider_part18b_dynamics.png) |
| 18c | Seasonal assay (setup) | [part18c_notes.md](experiments/part18/results/part18c_notes.md) | — |
| 18d | Mutable output biases | [part18d_notes.md](experiments/part18/results/part18d_notes.md) | — |
| 18e | Within-body switching pilot | [part18e_notes.md](experiments/part18/results/part18e_notes.md) | — |
| 18f | Longer-horizon within_aligned | [part18f_notes.md](experiments/part18/results/part18f_notes.md) | — |
| 18g | Confirmatory within_aligned full | [part18g_notes.md](experiments/part18/results/part18g_notes.md) | — |
| 18h | Competition pilot (planted vs cue) | [part18h_notes.md](experiments/part18/results/part18h_notes.md) | — |
| 18i | More seeds, pooled 18g+18i | [part18i_notes.md](experiments/part18/results/part18i_notes.md) | — |
| 19 | Predator–prey pilot | [part19_notes.md](experiments/part19/results/part19_notes.md) | — |
| 19b | Confirmatory predator–prey (20k) | [part19b_notes.md](experiments/part19/results/part19b_notes.md) | — |

## Data

Simulation logs are published on the Hugging Face dataset [igor53627/5d](https://huggingface.co/datasets/igor53627/5d) (public, CC-BY-4.0, 511 files under `part03`–`part19`, 1,623,131,569 bytes, 1.62 GB). Part 18 also holds the 18b–18i runs; Part 19 also holds the 19b confirmatory run. Per-part sizes, the log cadence, and where `bench_b300_0.pkl` and `bench_h100_0.pkl` sit are in [DATA.md](DATA.md).

```bash
huggingface-cli download igor53627/5d --repo-type dataset
```

```python
from huggingface_hub import snapshot_download
snapshot_download("igor53627/5d", repo_type="dataset", allow_patterns="part07/*")
```

## License

Research code — see individual files for details.
