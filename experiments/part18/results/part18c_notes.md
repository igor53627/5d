# Part 18c — seasonal assay, setup only

No population has been run. Nothing here is evidence that a brain learned to switch seasons. The GPU commands below are for a later, separate launch. This change does not start Modal and does not touch the Hugging Face dataset.

Parts 18 and 18b are unchanged as experiments. Their specs, notes, and `analyze18.py` stay. Re-analyzing those logs still calls `season()` (the historical sine). Re-running them still uses `season_rule=0`.

## What was actually wrong

Checked on the four Part 18b founder genomes in `archive/specs/b18bm.json` (mapped Part 17 brains, seeds 621–624) and on the step code.

1. **The inherited policy is a sign-locked bias.** At the body-size inputs used by `analyze18.py` (`Ub/U1−1 = −0.4`, `Ufar/U1 = 0.06`), `g_A − g_B` stays positive on an A-rich patch and on a B-rich patch. For founder 621 it stays about +1.66 to +1.84 across the whole food mixture. Hidden units are not saturated; the output biases are. Effective `b2` for `g_A` is about +1.2 after the `ws=2` scale, so the readout cannot cross zero.

2. **Noise on the new channels does not unlock that sign.** `W1` columns for reward, memory, and the oracle cue, plus the `o_h` readout, at `N(0, 0.5)` (the scale guessed earlier) give **0 cue-switchers out of 48 genomes** (12 founders × 4 seeds). The same is true of the Part 18b genomes themselves: those columns are exactly 0, and `p_spread=0.05` cannot move a +1.7 bias through zero. A direct search can build a switcher, but only by also shifting the `g_A` / `g_B` biases. The smallest such edit has RMS about 0.10–0.17 and still leaves the food, body-size, and `W2` weights alone. So the earlier proposal — keep the old weights, randomize only the new inputs — does not put a switching phenotype in the founder pool. It is still run, as the `inputs` / `cue` arms, because that is the hypothesis under test. The CPU probe says those arms should not switch. A GPU run is what can show whether mutation and selection build the missing readout anyway. Part 18b's between-body weight change was about 0.02 over 24k steps, so that is not the expected outcome.

3. **Reward is a weak teacher at `cp=0.5`.** `food_gain` is `eA − cp·eB` when A is food and the swap when B is food. The sign is negative only when the poison share exceeds `1/(1+cp) = 2/3`. Part 18b's B-season poison fraction was 0.70–0.81, so the population-mean reward was already slightly negative, but only just. Until a body is mostly eating poison, reward does not say so. This is the 18b parameter, kept on purpose. It is not a sign bug: a step that eats only A gives reward about +1.0 in an A season and −0.5 in a B season (`tests/test_season18c.py`).

4. **Oracle cue wiring is intact.** With cue weights set by hand, the live step's memory state follows the season sign (+0.96 then −0.96). The null cue result in 18b is not a disconnected input. It is a zero weight on an input that the locked readout ignores.

5. **`krep=10` does not compete with food taxis.** On two blobs, coarse kernels, the Part 18b `dtA`, median `|krep ∇U_far|` is 0.012 cells/step and `|∇Ã|` is 0.111. Ratio 0.109, while the speed cap is 0.225. The far kernel is also a ring, so once bodies are inside it the repulsion does not push them apart. Raising `krep` into the tens was not adopted as a fusion fix. `min_gap=4` only spaces the 0.02 contours at placement. It does not stop bodies that later share one food lobe from meeting. Whether that meeting erases a planted cue weight by homogenization (`hom=0.05`) is an outcome to read off `n(t)`, max mass fraction, and the between-body std of the new weights. It is not assumed.

6. **Season endpoints.** `season()` is still `sin(2πt/Ts) >= 0`, so a re-analysis of Parts 18 and 18b does not move. In float32, `sin(kπ) ≈ −1e-7`, so the live step at a half-period is one step early relative to float64. `season_rule=1` (18c only) uses `(t/Ts mod 1) <= 1/2`, which matches in both dtypes, and includes the endpoint in season A. Logged `D` is the last step of the chunk, and that step read the clock `t−1`. `snapshot_season` labels the row with that step. On the 200-step grid of the full run there are 4 such sign changes (frames 4200, 8200, 12200, 16200).

Memory is still written from the pre-advection output onto the post-advection field (one-step spatial lag). Drift per step is far below the blur radius, and the cue result does not depend on it. The order was left as in 18/18b.

`min_sep` is still a flag. `None` skips the test; any other value only rejects mask overlap. The number is not a distance. `min_gap` is the new distance, in Chebyshev cells.

## Arms

World is the Part 18b world: `L=48`, `cp=0.5`, `dec=1e-4`, `rG=0.002`, `smooth=1`, `lam=48`, `krep=10`, `kb/kr` as below, coarse step `m=5`. Season length is 4000 steps (`Ts=8000`), not 8000. That is the one ecology change, so that 20000 steps contain four switches. `lo` is the within-experiment small-noise control at this season length. It is not a re-run of 18b.

Seeds 731–734 inherit the mapped genomes of 18b seeds 621–624 (Part 17 `evo_s511`–`evo_s514`). Placement uses the run seed. Weights use a separate stream, so a `min_gap` retry cannot reshuffle them. `cue` copies `inputs`. `nobrain` copies `lo`. Pilot seed 731 uses the same genomes as the full run and stops at 12000 steps.

| arm | mem | cue | kb, kr | genomes |
|---|---|---|---|---|
| `lo` | 1 | 0 | 1, 10 | founder + `N(0, 0.05)` on every channel. 0/12 switchers on each seed. Noise std about 0.05. |
| `inputs` | 1 | 0 | 1, 10 | old weights identical to the founder. New channels `N(0, 0.5)`, std about 0.45–0.49. 0/12 switchers. |
| `cue` | 1 | 1 | 1, 10 | same genomes as `inputs`. The only difference from `inputs` is the oracle input. |
| `planted` | 1 | 1 | 1, 10 | 6 exact copies + 6 cue-switchers. The edit is the cue column and the `g_A`/`g_B` biases. This is an assay control: it asks whether a phenotype that already switches on the CPU probe changes the diet and is visible as a within-body sign flip. It is not a claim that the circuit evolved. |
| `nobrain` | 0 | 0 | 0, 0 | same genomes as `lo`. Diet floor. |

A switcher means, at the mixed-food probe, `g_A−g_B > 0.4` when the cue is +1 (and still `> 0.2` on an A-rich patch) and `< −0.4` when the cue is −1. Counts and RMS values are in `b18c_init_report.json` (numpy 2.5.4). The committed specs are the genomes that will run; do not regenerate them on another numpy unless the report is rebuilt and checked.

`p_spread` in the spec is 0. The genomes are `p_inits`. The placement RNG still consumes a Gaussian so positions match a `p_spread` run with the same seed.

## What a result would mean

`analyze18c.py` splits each logged sign change:

- **within**: mass-weighted change of `g_A−g_B` in bodies matched across the frame (centers within 8 cells). This is an individual change.
- **replacement**: change of the population mean minus that within term. This is mass reweighting, births, and deaths.
- Diet in a two-snapshot window on each side of the frame (poison share).
- `n`, max mass fraction, first time the largest body holds more than half the mass, division count, Price `S` on `g_A−g_B`, and the mass-weighted mean and between-body std of the new-channel weights.
- Founder `p_init` std, which the pickle already stores.

A within-body flip on `planted`, with the diet moving, would mean the assay can see a switch that was planted. It would not mean the population learned one. A within-body flip on `inputs` or `cue` would be new: those genomes do not switch at initialization. A mean shift with `within ≈ 0` and a changing cast of bodies is replacement, not a behavioral switch. `lo` and `nobrain` are there so a change in `cue` is not just season length, placement, or the food field.

## Commands

CPU smoke, no GPU charge:

```bash
python experiments/part18/run18c.py smoke
python -m pytest tests/test_season18c.py -q
```

`smoke` rebuilds the genomes, checks the switch counts, prints the taxis/repulsion ratio, and runs one 2D feeding step. Specs are rewritten only by `python experiments/part18/run18c.py write`.

Pilot, then full, from the repo root. These start H100 jobs. They are not part of this change.

```bash
modal run src/modal_app.py::spawn_h100 --kind ecolist --spec archive/specs/b18c_pilot.json --prefix b18cp
modal run src/modal_app.py::spawn_h100 --kind ecolist --spec archive/specs/b18c_full.json --prefix b18c
```

Pilot: 5 runs (seed 731 × five arms), 12000 steps, two phenotype-aligned switches. Full: 20 runs (seeds 731–734 × five arms), 20000 steps, four switches. Logs land on the Modal volume as `b18cp_*.pkl` and `b18c_*.pkl`. After they are local:

```bash
python experiments/part18/analyze18c.py 'results/eco18c/b18c_*.pkl' 48
```

## Cost

Part 18b was 16×H100, about 24 minutes each, 24000 steps, `L=48`, and the notes' billing delta was about $30.3. That is about $1.89 per run at 24000 steps, or about $4.73 per GPU-hour. Scaling only steps and run count, with the same grid and channel count:

| | runs | steps | GPU-hours | estimate |
|---|---:|---:|---:|---:|
| pilot | 5 | 12000 | 1.0 | $4.7 |
| full | 20 | 20000 | 6.7 | $31.6 |
| both | 25 |  | 7.7 | $36 |

Wall time if the jobs stay parallel is about 12 minutes (pilot) and 20 minutes (full). A queue turns that into waiting, not into a higher GPU-hour count. The historical band for one of these experiments is about $40, with a hard cap of $50. The sum above is inside the cap. It is an extrapolation from the 18b note, not a new invoice, and a price change or a retry would move it. Do not launch if the sum of pilot, full, and retries would pass $50.

## Unresolved

- No GPU trajectories, so no measured diet, fusion time, or selection differential.
- The planted arm can look like adaptation if the within/replacement split is ignored. It must not be reported as learning.
- `inputs` is predicted not to switch. If the full run agrees, that supports the CPU probe under mutation, fusion, and the weak reward. It does not test a different brain.
- One food lobe (`λ=L=48`). A-seekers share a destination. Smaller `λ` reintroduces the Part 18 overlap of A and B inside one body, so it was not changed.
- `hom=0.05` will mix genomes that touch. The logs have to show whether the six switchers are still distinguishable at the first B season.
- `D` stores one step, not the mean of the 200-step chunk. Switch metrics have that resolution.
- Four seeds, one season length, the Part 18b cost `cp=0.5`.
