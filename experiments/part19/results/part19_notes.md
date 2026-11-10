# Part 19 notes: predator–prey pilot

Design: [part19_design.md](../part19_design.md). New engine `src/fl_pred.py`: a second mass field `AP` consumes nearby prey `A`. Defaults (`kpred=ka=kp=krep_x=0`, `AP=0`) leave a prey-only world. Prey keep the Part 15 micro-brain plus an avoidance input/output. Predators have a matching brain and a fixed pursuit term. This part does not use the seasonal world.

## Pilot

2 seeds (191, 192) × 3 predation rates on the Part 15 ecology, 4D `L=48`, 8 prey + 4 predators, 6000 steps, `every=200`. Shared knobs: `eff=0.4`, `dec_p=6e-4`, `ka=1`, `kp=1`, `krep_x=5`. Arms: `low` 0.015, `mid` 0.05, `high` 0.15. Batched (3 ecos / seed) on Modal H100. Spec: `archive/specs/b19_pilot.json`, prefix `b19p`.

Cost: 0.818 H100-hours (1455 s + 1491 s), about **$3.23** at the $3.95/h rate used for 18g. Hard cap was $8. A first launch failed at log time 0 (wrong reduction axis on `D`); almost no GPU time. The rerun is the one reported here.

Raw logs: Hugging Face `part19/eco19/` (`b19p_0.pkl`, `b19p_1.pkl`, `manifest.json`, `pilot19_summary.json`).

## Results

Mass is in units of the founder soliton `M1`. “Bodies” are connected components of mass at least `0.15 M1` (the usual `run_eco` object cut). A leftover diffuse field can sit above the mass threshold with zero bodies.

| arm | kpred | seed | M_A end | M_P end | n_A | n_P | mass status | bodies | body-coexist until |
|-----|------:|-----:|--------:|--------:|----:|----:|-------------|--------|-------------------:|
| low | 0.015 | 191 | 10.19 | 1.74 | 20 | 2 | coexist | coexist | 6000 |
| low | 0.015 | 192 | 18.61 | 1.42 | 23 | 0 | coexist | prey_only | 4000 |
| mid | 0.05 | 191 | 7.57 | 1.06 | 4 | 0 | coexist | prey_only | 5800 |
| mid | 0.05 | 192 | 11.31 | 1.72 | 13 | 0 | coexist | prey_only | 4600 |
| high | 0.15 | 191 | 2.17 | 0.42 | 0 | 0 | coexist | no_bodies | 2200 |
| high | 0.15 | 192 | 6.89 | 1.52 | 0 | 0 | coexist | no_bodies | 1600 |

Start of every run: about 8 prey / 4 predators, `M_A≈8`, `M_P≈4`.

- **No immediate mass extinction.** All six runs still have both `M_A` and `M_P` above `0.15` at 6k. The coupling does not wipe the grid in a few hundred steps.
- **`low`:** prey boom (20–23 bodies, mass 10–19). Predator mass slowly shrinks from ~4 to ~1.4–1.7. Seed 191 still has two predator bodies at the end; seed 192 loses them at 4k and keeps only diffuse predator mass. This is weak predation, close to a prey-only world.
- **`mid`:** the most interesting arm. Seed 191 shows one clear half-cycle: prey mass falls 8.2→4.6 around t=3k as predator bodies peak (5), then prey recovers to 7.6 while predator mass declines 2.55→1.06. Seed 192 is smoother (prey 7.5–11.3, predator mass ~1.6–2.3). Predator *bodies* drop below the 0.15 cut by the end on both seeds, but predator mass remains.
- **`high`:** too harsh for bodies. Prey objects are gone by 2–3k and do not reform. Predator objects rise briefly (up to 8–9) then also vanish. Diffuse mass lingers, so a mass-only cutoff would still say “coexist”; that is misleading. Body coexistence lasts only 1.6–2.2k steps.

Oscillation metrics on a 6k window are weak: at most one half-cycle, FFT periods sit near half the window (~3100), and predator lags are not stable across seeds. They are logged in `pilot19_summary.json` but should not be quoted as a Lotka–Volterra period. Avoidance (`g_Av`) and pursuit were enabled but this pilot is not long enough for an arms-race claim.

## Verdict

The mechanics work, and there is a usable window around **`kpred=0.015–0.05`**. `0.15` eliminates prey bodies and is not a candidate for a full assay. Predator mass persists in the window, but predator bodies often fragment below the object cut by 6k, so a longer run is needed to see whether they regroup or slowly die.

## Recommended next step

A confirmatory mid (and low) run at 20k steps, 4 seeds, plus a prey-only control (`kpred=0` or `n_pred=0`) so food-driven mass wiggles are not mistaken for cycles. Keep body-level coexistence as the primary readout. Budget on the order of $15–20. Do not start that run without an explicit yes. An arms-race arm (planted avoiders vs evolving avoidance / pursuit) only makes sense after a longer mid run actually holds both species as bodies.
