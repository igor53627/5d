# Part 19 design: predator–prey

Short design for a second species in the Hyperlife ecology. The public name of
the project is Hyperlife. This part is independent of the Part 18 seasonal
assay; it builds on the Part 15 micro-brain world (one food field, heritable
weights) rather than on seasons, so food oscillations do not confound
Lotka–Volterra-like cycles.

## Goal

Ask whether two soliton species can coexist on the same 4D lattice, whether
their total masses oscillate with a predator lag, and whether prey can evolve
avoidance of predator density. A later assay can look at an arms race
(prey avoidance vs predator pursuit). The first question, and the only one
this pilot answers, is mechanical: is there a predation-rate window where
neither species goes extinct immediately?

## Why a second mass field

The ecology already carries heritable channels `Q` on a single mass field `A`.
A species bit in `Q` would be simpler to store, but homogenization and fusion
would mix predator and prey genotypes inside one body and blur the identity
away. A second mass field `AP` (predators), with its own oscillator state
`(xp, yp, zp, cp)` and its own channels `QP`, keeps the two species from
merging. Coupling is only through sensing, a soft cross-repulsion, and a
short-range feeding term.

Defaults keep the legacy prey-only world: `kpred=0`, `ka=0`, `kp=0`, `krep_x=0`,
and `AP=0`. With those gains off and the new brain weights at zero, prey mass
and food follow the Part 15 micro-brain step (no predation, no extra force).
Old `model="brain"` / `model="season"` runs are untouched; Part 19 is
`model="pred"` in `glider_eco.run_eco`.

## Mechanics

Prey `A` eat the food field `F` exactly as in Part 15 (`keat`, `rF`, `F0`,
decay `dec`). Predators do not eat `F`. After both species advect, predators
consume nearby prey:

- Let `K_eat` be the body-scale blur kernel, normalized so it sums to 1.
- `P_s = conv(AP, K_eat)`, `A_s = conv(A, K_eat)`.
- Prey loss at a cell: `loss = min(kpred * A * P_s, A)`.
- Predator gain at a cell: `gain = eff * kpred * AP * A_s`.
- For a symmetric normalized kernel, `sum(A * P_s) = sum(AP * A_s)` when the
  `min` cap is not binding, so total gain is about `eff` times total loss.

`kpred=0` (the default) makes both terms zero. `eff` is the conversion
efficiency (typical values 0.2–0.6). Predators decay at `dec_p` (set higher
than prey `dec` in the specs; default 0, i.e. off unless a spec sets it).

Cross-repulsion, default off: each species adds `-krep_x * ∇Ufar(other)` to
its velocity so they do not sit on top of each other, while still being able
to approach within the blur radius. Ordinary within-species `krep` is
unchanged.

### Prey brain (avoidance)

The Part 15 micro-brain is `3→3→3`. Part 19 adds one input and one output,
inert unless the new gains are on:

- Inputs: `[F̃−1, Ub/U1−1, Ufar/U1, P̃]` with `P̃ = conv(AP, blur) / U1`.
- Outputs: `[g_F, g_R, g_E, g_Av]`.
- Extra force: `v += dtA·10·(−ka·g_Av·∇P̃)`.

`ka=0` (default) turns avoidance off. Prey can evolve `g_Av` through the
usual weight mutations (`mutw`). A planted-avoidance control can set the
`g_Av` bias positive at init.

### Predator motion (pursuit)

Predators have the same brain layout. Their first input is prey scent
`Ã = conv(A, blur) / U1` instead of food. A fixed pursuit term,
`v += dtA·10·(kp·∇Ã)`, is the default hunting behaviour for the pilot
(`kp=0` keeps it off). Evolving pursuit (`kb` on the predator brain) is
reserved for a later arms-race arm.

## Metrics

Computed from `run_eco` logs. Prey stay in the usual `log` / `tot` fields.
Predators are stored alongside as `plog` / `ptot` (mass, object list, and
the current-step kill rate). Thresholds use the founder soliton mass `M1`.

| Metric | Definition |
|--------|------------|
| Coexistence time | Last log time at which `M_A` and `M_P` are both `> 0.15 M1` |
| Extinction | `prey` / `pred` / `both` / `neither`, using the same threshold at the final frame (and whether the time series hit the floor earlier and stayed there) |
| Mass series | `M_A(t)`, `M_P(t)`, object counts `n_A`, `n_P` |
| Oscillation | After a linear detrend: number of signed peaks, dominant period (FFT or peak spacing), and the lag that maximises `corr(M_A, M_P shifted)`. A Lotka–Volterra-like cycle has a positive period and a predator lag of about a quarter period |
| Kill rate | Spatial sum of prey loss, in units of `M1` per step, snapshot at each log frame |
| Arms-race indicators (logged now, interpreted later) | Mass-weighted mean `g_Av` of prey; mass-weighted mean pursuit output of predators; spatial overlap `sum(A·P_s) / (sum(A) sum(AP))` |

A short 6–8k-step pilot can support coexistence and a rough cycle count. It
is not long enough for a serious arms-race claim.

## Arms and controls

### Pilot (this patch)

Two seeds × three predation rates, same world otherwise. Budget about $5,
hard cap $8, on Modal H100, no long CPU tuning.

| Arm | Role |
|-----|------|
| `low` | `kpred` below the guessed coexistence window (predators may starve) |
| `mid` | Baseline rate, hoped to coexist |
| `high` | `kpred` above the window (prey may be driven out, then predators starve) |

Shared knobs: Part 15 ecology (`keat=0.01`, `dec=3e-4`, `rF=1e-4`, `F0=0.08`,
`krep=10`, `hom=0.05`, `mutw=0.004`, `kb=1`, `kr=10`) on 4D `L=48` (cheaper
than the Part 15 `56^4` grid, same `opts`: coarse / fnoise / m=5). `eff=0.4`,
`dec_p=6e-4` (twice prey decay), `kp=1`, `ka=1`, `krep_x=5`. Founders:
8 prey + 4 predators, `p_spread=0.5`, seeds 191 and 192, 6000 steps, log
every 200. Avoidance and pursuit channels are on so they can start to move,
but the pilot verdict is coexistence, not a claim that avoidance evolved.

### Later full assay (not in this patch)

Only if the pilot finds a non-extinct window:

1. **Prey only** — `n_pred=0` or `kpred=0`. Control for food-driven mass
   cycles that are not predator–prey.
2. **No pursuit** — `kp=0`. Predators hunt only by sitting still / soliton
   wander.
3. **No avoidance** — `ka=0`. Prey cannot steer from `AP`.
4. **Planted avoiders** — `g_Av` bias set positive at init, same genomes
   otherwise.
5. **Evolving pursuit** — predator `kb>0`, compare mean pursuit output
   against a frozen-pursuit control.
6. **More seeds, longer horizon** (20k) on the mid rate, for periods and
   coexistence time with confidence intervals.

A true competition / arms-race read needs (4) vs (3) and (5), plus the
overlap and `g_Av` series. That is Part 19b or the full run, not the smoke.

## Implementation map

- `src/fl_pred.py` — step, params, state. Defaults off.
- `experiments/part10/glider_eco.py` — `model="pred"`, optional `n_pred` /
  `p_inits_pred`, and `plog` / `ptot` logging. Existing models unchanged.
- `src/modal_app.py` — ship `fl_pred.py` in the image.
- `experiments/part19/` — design, runner, metrics, notes, summary JSON.
- `tests/test_pred19.py` — defaults, mass transfer, avoidance taxis, specs.
- Specs: `archive/specs/b19_pilot.json`.

## Risks

- **Extinction on every arm.** Then the mid rate is wrong, or 4D solitons
  with `krep` never get close enough to feed. Next step would be a second
  short pilot on `kpred` / `krep_x` / `eff`, still under a few dollars, not
  a full run.
- **Overlap too rare.** Raise `kpred` or lower `krep_x`, or shorten the eat
  kernel. The local `A * P` term without blur would almost certainly starve
  predators, which is why feeding uses `K_eat`.
- **Mass leak.** The `min(..., A)` cap and non-normalized kernels can break
  `gain ≈ eff * loss`. The eat kernel is DC-normalized; tests check the
  ratio on a tiny overlapping pair.
- **Compute.** Two species is roughly 1.5–2× a Part 15 step. Six runs of
  6k steps on H100 should stay near $3–6 if we batch the three arms per
  seed. The hard cap is $8; we do not start a longer run in this patch.
