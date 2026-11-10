# Part 19b notes: confirmatory predator–prey

Builds on the [Part 19 pilot](part19_notes.md) and [design](../part19_design.md). Same Part 15 ecology (4D `L=48`, `keat=0.01`, `dec=3e-4`, `rF=1e-4`, `F0=0.08`, `krep=10`, `hom=0.05`, `mutw=0.004`, `kb=1`, `kr=10`, opts coarse/fnoise/`m=5`), same knobs (`eff=0.4`, `dec_p=6e-4`, `ka=1`, `kp=1`, `krep_x=5`), 8 prey + 4 predators (`n_pred=0` on the control), `every=200`, `proj_every=400`.

## Run

4 new seeds (193–196, disjoint from the pilot 191–192) × 3 arms × 20k steps. Arms: `low` `kpred=0.015`, `mid` `kpred=0.05`, and a prey-only control (`kpred=0`, `n_pred=0`). Because `n_pred` is a `run_eco`-level flag, each seed is one Modal container with two calls: low+mid batched (`n_pred=4`, batch=2), then prey-only (`n_pred=0`, batch=1). Same seed ⇒ the same 8 prey placements. Spec: `archive/specs/b19b_full.json`, prefix `b19b`.

Cost: 5.131 H100-hours (4649 s + 4612 s + 4617 s + 4593 s), about **$20.27** at the $3.95/h rate used for 18g. Hard cap was $28. One launch, 11:58–13:16 Asia/Dubai (UTC+4): https://modal.com/apps/igor53627/main/ap-JBhs1xMKCsV6GXToX3hsEM

Raw logs: Hugging Face `part19/eco19b/` (`b19b_0.pkl`–`b19b_3.pkl`, `manifest.json`, `full19b_summary.json`).

Primary readout is body-level coexistence: both species have at least one connected component of mass ≥ `0.15 M1` (the usual `run_eco` object cut). Time-to-loss is the first time a species drops all bodies and never recovers; held-to-end seeds are right-censored at 20k in the pooled CI. Mass can linger as a diffuse field with zero bodies, so mass-only “coexist” is secondary. Periods are FFT dominant period and autocorrelation; lags use the smallest `|lag|` among near-ties. Pooled 95% CIs are bootstrap percentile, 10,000 resamples, seed 19.

## Results

Mass is in units of the founder soliton `M1`. Start of every pred-arm run: about 8 prey / 4 predators, `M_A≈8`, `M_P≈4`. The control starts with the same 8 prey and no predators.

| arm | kpred | seed | M_A end | M_P end | n_A | n_P | mass status | bodies | body-coexist until | loss (who @ t) |
|-----|------:|-----:|--------:|--------:|----:|----:|-------------|--------|-------------------:|----------------|
| low | 0.015 | 193 | 10.04 | 0.99 | 24 | 0 | coexist | prey_only | 19600 | pred @ 19800 |
| low | 0.015 | 194 | 18.70 | 1.53 | 1 | 3 | coexist | coexist | 20000 | held |
| low | 0.015 | 195 | 0.13 | 0.00 | 0 | 0 | both_extinct | no_bodies | 7800 | prey @ 8000 |
| low | 0.015 | 196 | 4.09 | 1.32 | 6 | 3 | coexist | coexist | 20000 | held |
| mid | 0.05 | 193 | 9.90 | 1.29 | 4 | 0 | coexist | prey_only | 4400 | pred @ 4800 |
| mid | 0.05 | 194 | 13.20 | 1.70 | 2 | 0 | coexist | prey_only | 5400 | pred @ 5600 |
| mid | 0.05 | 195 | 3.52 | 0.66 | 0 | 0 | coexist | no_bodies | 4000 | pred @ 4600 |
| mid | 0.05 | 196 | 6.08 | 1.28 | 1 | 0 | coexist | prey_only | 5000 | pred @ 5200 |
| preyonly | 0 | 193 | 25.67 | 0 | 7 | 0 | prey_only | prey_only | — | (no predators) |
| preyonly | 0 | 194 | 27.37 | 0 | 30 | 0 | prey_only | prey_only | — | (no predators) |
| preyonly | 0 | 195 | 27.84 | 0 | 35 | 0 | prey_only | prey_only | — | (no predators) |
| preyonly | 0 | 196 | 24.51 | 0 | 38 | 0 | prey_only | prey_only | — | (no predators) |

Pooled (mean and 95% bootstrap CI):

| arm | held bodies | coexist bodies t | time-to-loss | frac both-bodies | n_A end | n_P end | M_A end | M_P end |
|-----|------------:|-----------------:|-------------:|-----------------:|--------:|--------:|--------:|--------:|
| low | 2/4 | 16850 [10850, 20000] | 16950 [11000, 20000] | 0.69 [0.48, 0.87] | 7.8 [0.5, 18.3] | 1.5 [0.0, 3.0] | 8.24 [2.11, 15.05] | 0.96 [0.33, 1.43] |
| mid | 0/4 | 4700 [4200, 5200] | 5050 [4700, 5400] | 0.21 [0.19, 0.23] | 1.8 [0.5, 3.3] | 0.0 [0.0, 0.0] | 8.18 [4.80, 11.55] | 1.23 [0.82, 1.59] |
| preyonly | — | — | — | — | 27.5 [14.0, 36.5] | 0 | 26.35 [25.09, 27.61] | 0 |

On the control, body-coexistence and time-to-loss are not meaningful (no predator bodies from t=0). Prey persist on all four seeds. Mean prey-body count over the run is 20.8 [12.0, 27.4] versus 6.5 [3.0, 11.1] on low and 2.3 [1.7, 2.6] on mid.

- **`low` (0.015):** the only arm that can hold both species as bodies to 20k, but not reliably (2/4 held). Seed 196 shows a clear half-cycle: prey peak at t=4k (11 bodies, mass 9.2), crash to 1 body / mass 2.7 at t=8k as predator bodies peak (5, mass 2.0), then prey recover to 6 and predators end at 3. Seed 194 is held at the end (1 prey body of mass 18.7, 3 predator bodies), but predator bodies briefly hit zero around t=12k and later reformed — so “held” allows a temporary gap as long as the species returns. Seed 193 almost holds: predator bodies are gone by t=16k (loss recorded at 19.8k) while prey boom to 24 bodies. Seed 195 is a collapse: prey bodies vanish first at 8k and never return; leftover mass is near zero. The same seed in the prey-only control ends with 35 prey bodies and mass 27.8, so the crash is predation-driven, not a bad world seed. The pooled coexist-time CI is wide because of 195.
- **`mid` (0.05):** consistently too harsh for predator *bodies*. All four seeds lose them first between 4.6k and 5.6k (time-to-loss 5050 [4700, 5400]), matching the 6k pilot where mid predator bodies were already gone or going by 4.6–5.8k. Prey usually survive: they crash around t=4k (often to 0–2 bodies) then recover once predator objects are gone (3/4 end prey_only). Seed 195 never reforms bodies, but diffuse mass of both fields lingers (`M_A=3.52`, `M_P=0.66`). Predator mass remains on every mid seed (~0.7–1.7) without regrouping into objects.
- **Prey-only control:** prey boom on every seed (`M_A` end 24.5–27.8). Body counts vary (7 on 193, with a few large objects; 30–38 on the others). Mid/low prey suppression is therefore predation, not food exhaustion.

Oscillation metrics over 20k are still not a period claim. Autocorrelation never finds a period (no arm, no seed). FFT dominant periods sit at half the window (~10100) almost everywhere — a trend/window artifact, not a Lotka–Volterra cycle. Predator lags on mid are 0 or 2000 with high correlation (r≈0.80–0.92), but that tracks a one-way crash more than a delayed cycle. Low lags are 0–2600 with only moderate r (0.50–0.85). Seed 196 low is the closest thing to one half-cycle; it is not repeated.

## Verdict

The usable window is narrower than the 6k pilot suggested. **`kpred=0.05` does not sustain predator bodies over 20k** — the loss at ~5k is tight across four new seeds. Predator mass persists as a diffuse field; prey typically recover after the objects are gone. **`kpred=0.015` can hold both as bodies to 20k, but only on 2/4 seeds**, with a third loss at 19.8k and one full collapse. That is an edge, not a stable coexistence regime. The prey-only control confirms that prey carrying capacity is high without predators. There is still no reliable oscillation period to quote.

## Recommended next step

Do not start an arms-race arm (planted avoiders vs evolving avoidance / pursuit) until body-level coexistence is more reliable than 2/4 at 20k. A cheaper next probe, if wanted, is a short low-side scan around `kpred=0.015` (slightly lower kpred, or milder predator decay / higher `eff` / more founder predators) rather than another 20k mid run. Do not start a paid follow-up without an explicit yes.

## Engine notes (what 19b actually ran)

Same engine as the 19 pilot: `coarse=True`, `kb=1`, `kp=1`. See [part19_notes.md](part19_notes.md) for the coarse-gain bound and the `pred_pursuit_evolve` knob. In short: predator gain was not logged, so the coarse-path `gain/(eff·loss)` ratio cannot be read from `ptot`; operator checks put the mismatch well below 1% on 4D fields like these runs. Evolving pursuit (`kb·g_F·∇Ã`) was on, on top of fixed `kp`. All four `b19b_*.pkl` jobs have `partial=False` on every `run_eco` result, with logs through t=20000 (101 frames); none of the 12 counted runs were incomplete. The body-coexistence table and the mid-vs-low verdict are unchanged: a sub-percent conversion error and already-on evolving pursuit do not explain why mid loses predator bodies at ~5k while low only holds 2/4 seeds to 20k.

