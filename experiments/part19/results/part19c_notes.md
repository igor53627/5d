# Part 19c notes: low-side kpred scan

Builds on the [19b confirmatory](part19b_notes.md) and [design](../part19_design.md). Same Part 15 ecology as the 19b low arm (4D `L=48`, `keat=0.01`, `dec=3e-4`, `rF=1e-4`, `F0=0.08`, `krep=10`, `hom=0.05`, `mutw=0.004`, `kb=1`, `kr=10`, opts coarse/fnoise/`m=5`), same knobs (`eff=0.4`, `dec_p=6e-4`, `ka=1`, `kp=1`, `krep_x=5`), 8 prey + 4 predators, `every=200`, `proj_every=400`. `pred_pursuit_evolve` defaults to 1, as in 19/19b.

## Run

4 new seeds (197–200, disjoint from the pilot 191–192 and 19b 193–196) × 2 arms × 20k steps. Arms: `k005` `kpred=0.005` and `k010` `kpred=0.01`. Each seed is one Modal container with both kpred values batched (`n_pred=4`, batch=2), the same packing as the 19b pred batch. The 19b low arm (`kpred=0.015`, seeds 193–196) is a comparison arm in the analyzer and was not rerun. Spec: `archive/specs/b19c_full.json`, prefix `b19c`.

Cost: 3.481 GPU-hours (3284 s + 2655 s + 3275 s + 3317 s), about **$13.75** at the $3.95/h H100 rate used for 18g/19/19b. Hard cap was ~$16. One launch, 09:04–09:58 Asia/Dubai (UTC+4): https://modal.com/apps/igor53627/main/ap-45CUEEEEN4gk8xH04RT0O0 . Three containers ran on H100; seed 198 was issued an H200 and finished earlier (2655 s). A client interruption mid-run (around 09:48, with seed 198 already complete and the other three at t=16000) did not stop the jobs; no pickle was relaunched.

Raw logs: Hugging Face `part19/eco19c/` (`b19c_0.pkl`–`b19c_3.pkl`, `manifest.json`, `full19c_summary.json`).

Primary readout is body-level coexistence: both species have at least one connected component of mass ≥ `0.15 M1`. Time-to-predator-loss is the first time predator bodies drop and never recover; held-pred seeds enter as 20000, and the pooled interval is marked `right_censored` only when `n_held_pred > 0`. Prey held and predator held are reported separately. The analyzer skips a run whose checkpoint is `partial` or whose log ends before the 20k horizon. Periods are FFT dominant period and autocorrelation; lags use the smallest `|lag|` among near-ties. Pooled 95% CIs are bootstrap percentile, 10,000 resamples, seed 19.

## Results

Mass is in units of the founder soliton `M1`. Every 19c run starts with about 8 prey / 4 predators, `M_A≈8`, `M_P≈4`. The 19b comparison arm used the same founder counts on seeds 193–196.

| arm | kpred | seed | src | M_A end | M_P end | n_A | n_P | bodies | body-coexist until | pred loss |
|-----|------:|-----:|-----|--------:|--------:|----:|----:|--------|-------------------:|-----------|
| k005 | 0.005 | 197 | 19c | 19.45 | 1.02 | 26 | 1 | coexist | 20000 | held |
| k005 | 0.005 | 198 | 19c | 19.30 | 1.06 | 20 | 1 | coexist | 20000 | held |
| k005 | 0.005 | 199 | 19c | 19.31 | 1.07 | 33 | 0 | prey_only | 12000 | pred @ 12200 |
| k005 | 0.005 | 200 | 19c | 12.72 | 1.22 | 15 | 3 | coexist | 20000 | held |
| k010 | 0.01 | 197 | 19c | 15.70 | 2.17 | 21 | 6 | coexist | 20000 | held |
| k010 | 0.01 | 198 | 19c | 1.72 | 0.05 | 0 | 0 | no_bodies | 13000 | pred @ 13200 |
| k010 | 0.01 | 199 | 19c | 14.44 | 1.20 | 24 | 0 | prey_only | 9200 | pred @ 9400 |
| k010 | 0.01 | 200 | 19c | 2.59 | 0.37 | 3 | 0 | prey_only | 19200 | pred @ 19400 |
| k015 | 0.015 | 193 | 19b | 10.04 | 0.99 | 24 | 0 | prey_only | 19600 | pred @ 19800 |
| k015 | 0.015 | 194 | 19b | 18.70 | 1.53 | 1 | 3 | coexist | 20000 | held |
| k015 | 0.015 | 195 | 19b | 0.13 | 0.00 | 0 | 0 | no_bodies | 7800 | pred @ 9000 (prey first @ 8000) |
| k015 | 0.015 | 196 | 19b | 4.09 | 1.32 | 6 | 3 | coexist | 20000 | held |

Pooled (mean and 95% bootstrap CI). Time-to-predator-loss is right-censored at 20k on every arm here (`n_held_pred > 0`).

| arm | held pred | held prey | held both | coexist bodies t | time-to-pred-loss (censored at 20k if held) | frac both-bodies | n_A end | n_P end | M_A end | M_P end |
|-----|----------:|----------:|----------:|-----------------:|-------------:|-----------------:|--------:|--------:|--------:|--------:|
| k005 | 3/4 | 4/4 | 3/4 | 18000 [14000, 20000] | 18050 [14150, 20000] | 0.68 [0.43, 0.92] | 23.5 [17.5, 29.8] | 1.3 [0.3, 2.5] | 17.70 [14.37, 19.41] | 1.09 [1.03, 1.18] |
| k010 | 1/4 | 3/4 | 1/4 | 15350 [11100, 19600] | 15500 [11300, 19700] | 0.67 [0.46, 0.87] | 12.0 [1.5, 22.5] | 1.5 [0.0, 4.5] | 8.61 [2.16, 15.07] | 0.95 [0.21, 1.72] |
| k015 | 2/4 | 3/4 | 2/4 | 16850 [10850, 20000] | 17200 [11750, 20000] | 0.69 [0.48, 0.87] | 7.8 [0.5, 18.3] | 1.5 [0.0, 3.0] | 8.24 [2.11, 15.05] | 0.96 [0.33, 1.43] |

The k015 coexist-time CI matches 19b. Time-to-pred-loss (17200 [11750, 20000]) is slightly later than 19b’s first-of-either interval (16950 [11000, 20000]) because seed 195 lost prey bodies first at 8k and predator bodies at 9k.

- **`k005` (0.005):** the only arm in which both species persist as bodies on most seeds (3/4 held both; prey 4/4; pred 3/4). Seeds 197, 198, and 200 are held to 20k. Prey end large (15–26 bodies, mass 12.7–19.5) with 1–3 predator bodies and predator mass stuck near 1.0–1.2. Seed 200 has both-bodies on every logged frame (`frac_bodies=1`). Seed 199 is the miss: predator bodies are gone at 12.2k and never return, while prey boom to 33 bodies / mass 19.3; leftover predator mass is still 1.07, as a diffuse field.
- **`k010` (0.01):** worse than 0.015, not a midpoint. Only seed 197 holds both (21 prey / 6 predator bodies, the strongest predator-body count in the scan). Seed 199 loses predator bodies at 9.4k (prey recover to 24). Seed 200 almost holds, then loses predator bodies at 19.4k. Seed 198 is a collapse: predators go first at 13.2k, prey follow at 15k, leftover mass is small.
- **`k015` (0.015, 19b):** unchanged scientifically (2/4 both held, one late predator loss at 19.8k, one predation-driven collapse). See [part19b_notes.md](part19b_notes.md).

Oscillation metrics are still not a period claim. Autocorrelation finds no period on any k005 or k015 seed. The only ACF hit is k010 seed 198, `period_P_acf=9200` with r=0.29 — a weak peak on a collapsing run, not a cycle. FFT dominant periods sit at half the window (~10100) on almost every series, a trend/window artifact. Predator lags on k005 are 400–4800 (mean 2700 [1100, 4300]) with moderate-to-high r (0.74–0.86), but without a repeatable ACF period they are not evidence of a Lotka–Volterra oscillation.

## Verdict

**Lower kpred helps, but only at 0.005, and it is not yet a fully stable regime.** At `kpred=0.005`, both species persist as bodies on 3/4 seeds (prey on all four); that is the first arm in this world where coexistence is the typical 20k outcome. `kpred=0.01` does not improve on 19b’s 0.015 edge (1/4 vs 2/4 held both) and includes one collapse. Even at 0.005, one seed still sheds predator bodies at 12.2k, and held runs end with few predator objects (often one) beside a large prey population — an edge that works more often, not a tight cycle. There is still no reliable oscillation period to quote.

A next probe, if wanted, could add more seeds at 0.005 (to see whether 3/4 holds up) or a slightly lower rate, rather than revisiting 0.01. Do not start an arms-race arm, and do not start a paid follow-up, without an explicit yes.

## Engine notes (what 19c actually ran)

Same engine as 19b on main `781409e`: `coarse=True`, `kb=1`, `kp=1`, `pred_pursuit_evolve=1` (default, not set in the eco dict). All four `b19c_*.pkl` jobs have `partial=False` on every `run_eco` result, with logs through t=20000 (101 frames); none of the 8 new runs, and none of the 4 comparison runs, were incomplete.
