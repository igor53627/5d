# Part 19d notes: kpred=0.005 replicate, 8 new seeds

Builds on the [19c low-side scan](part19c_notes.md). Same Part 15 ecology as the 19c `k005` arm (4D `L=48`, `keat=0.01`, `dec=3e-4`, `rF=1e-4`, `F0=0.08`, `krep=10`, `hom=0.05`, `mutw=0.004`, `kb=1`, `kr=10`, opts coarse/fnoise/`m=5`), same knobs (`eff=0.4`, `dec_p=6e-4`, `ka=1`, `kp=1`, `krep_x=5`, `kpred=0.005`), 8 prey + 4 predators, `every=200`, `proj_every=400`. `pred_pursuit_evolve` defaults to 1, as in 19/19b/19c.

## Run

8 new seeds (201–208, disjoint from the pilot 191–192, 19b 193–196, and 19c 197–200) × one arm × 20k steps, `kpred=0.005`. Each seed is one Modal H100 container with a single eco (`batch=1`, `n_pred=4`). The 19c `k005` seeds (197–200) are pooled in the analyzer and were not rerun. Spec: `archive/specs/b19d_full.json`, prefix `b19d`.

A `batch=1` timing probe (seed 201, timeout 3600 s) finished in 1496 s on an H100, so the remaining seven fitted under the $15 hard cap. They were launched in one wave with a cap-safe timeout of 1730 s (234 s of margin vs the probe; worst-case spend $14.93 if every job hit the timeout). A client interruption while five of those seven were still at t=16000 did not stop the jobs; no pickle was relaunched.

Cost: 3.036 GPU-hours (1496 + 1406 + 1430 + 1064 + 1455 + 1498 + 1081 + 1500 s), about **$11.99** at the $3.95/h H100 rate used for 18g/19/19b/19c. Under the $15 cap. Probe 18:46–19:09 Asia/Dubai (UTC+4): https://modal.com/apps/igor53627/main/ap-Z2bTgwdjaXwlBV2lq96E36 . Remaining seven 19:12–19:35 Asia/Dubai (UTC+4): https://modal.com/apps/igor53627/main/ap-17DjzJXUZkZG5sSRQZa20Q . Six containers ran on H100 (1406–1500 s); seeds 204 and 207 were issued H200s and finished earlier (1064 s, 1081 s).

Raw logs: Hugging Face `part19/eco19d/` (`b19d_0.pkl`–`b19d_7.pkl`, `manifest.json`, `full19d_summary.json`).

Primary readout is body-level coexistence (both species have at least one connected component of mass ≥ `0.15 M1`). Time-to-predator-loss is the first time predator bodies drop and never recover; held-pred seeds enter as 20000, and the pooled interval is `right_censored` only when `n_held_pred > 0`. Prey held and predator held are reported separately. Incomplete or short-of-horizon logs are skipped. Periods are FFT dominant period and autocorrelation. Pooled 95% CIs for means are bootstrap percentile, 10,000 resamples, seed 19. The 12-seed “both held at 20k” fraction uses a Clopper–Pearson binomial interval.

## Results

Mass is in units of the founder soliton `M1`. Every run starts with about 8 prey / 4 predators, `M_A≈8`, `M_P≈4`.

| arm | kpred | seed | src | M_A end | M_P end | n_A | n_P | bodies | body-coexist until | pred loss |
|-----|------:|-----:|-----|--------:|--------:|----:|----:|--------|-------------------:|-----------|
| k005 | 0.005 | 201 | 19d | 8.83 | 1.29 | 10 | 3 | coexist | 20000 | held |
| k005 | 0.005 | 202 | 19d | 12.41 | 1.59 | 17 | 2 | coexist | 20000 | held |
| k005 | 0.005 | 203 | 19d | 22.79 | 0.57 | 32 | 0 | prey_only | 8400 | pred @ 8600 |
| k005 | 0.005 | 204 | 19d | 18.52 | 0.83 | 29 | 0 | prey_only | 16000 | pred @ 16200 |
| k005 | 0.005 | 205 | 19d | 19.10 | 0.78 | 30 | 1 | coexist | 20000 | held |
| k005 | 0.005 | 206 | 19d | 15.74 | 0.95 | 27 | 0 | prey_only | 16600 | pred @ 16800 |
| k005 | 0.005 | 207 | 19d | 27.00 | 0.33 | 31 | 0 | prey_only | 19200 | pred @ 19400 |
| k005 | 0.005 | 208 | 19d | 13.65 | 0.82 | 21 | 1 | coexist | 20000 | held |
| k005 | 0.005 | 197 | 19c | 19.45 | 1.02 | 26 | 1 | coexist | 20000 | held |
| k005 | 0.005 | 198 | 19c | 19.30 | 1.06 | 20 | 1 | coexist | 20000 | held |
| k005 | 0.005 | 199 | 19c | 19.31 | 1.07 | 33 | 0 | prey_only | 12000 | pred @ 12200 |
| k005 | 0.005 | 200 | 19c | 12.72 | 1.22 | 15 | 3 | coexist | 20000 | held |

Pooled (mean and 95% bootstrap CI). Time-to-predator-loss is right-censored at 20k in every block below (`n_held_pred > 0`).

| set | held pred | held prey | held both | coexist bodies t | time-to-pred-loss | frac both-bodies | n_A end | n_P end | M_A end | M_P end |
|-----|----------:|----------:|----------:|-----------------:|------------------:|-----------------:|--------:|--------:|--------:|--------:|
| 19d (n=8) | 4/8 | 8/8 | 4/8 | 17525 [14625, 19575] | 17625 [14775, 19600] | 0.55 [0.39, 0.72] | 24.6 [19.1, 29.2] | 0.88 [0.25, 1.62] | 17.3 [13.5, 21.1] | 0.89 [0.65, 1.16] |
| 19c (n=4) | 3/4 | 4/4 | 3/4 | 18000 [14000, 20000] | 18050 [14150, 20000] | 0.68 [0.43, 0.92] | 23.5 [17.5, 29.8] | 1.25 [0.25, 2.50] | 17.7 [14.4, 19.4] | 1.09 [1.03, 1.18] |
| pooled (n=12) | 7/12 | 12/12 | 7/12 | 17683 [15433, 19517] | 17767 [15550, 19583] | 0.59 [0.46, 0.73] | 24.2 [20.1, 28.2] | 1.00 [0.42, 1.67] | 17.4 [14.7, 20.1] | 0.96 [0.78, 1.14] |

The 12-seed both-held fraction is **7/12 = 0.583**, Clopper–Pearson 95% CI **[0.277, 0.848]**.

- **19d (new seeds):** prey bodies persist on all eight seeds. Predator bodies persist on only 4/8 (201, 202, 205, 208). The four losses are at 8.6k, 16.2k, 16.8k, and 19.4k — not a tight cluster. Held runs still end with few predator objects (1–3) beside a large prey population (10–30 bodies). Seed 201 (the probe) is the cleanest: both-bodies on every logged frame, ending 10 prey / 3 predators. Seed 203 is the earliest miss (pred bodies gone at 8.6k, prey boom to 32). Seed 207 almost holds, then loses predators at 19.4k.
- **19c k005 (unchanged):** 3/4 both held; seed 199 loses predators at 12.2k. See [part19c_notes.md](part19c_notes.md).
- **Oscillation:** autocorrelation finds no period on any of the 12 seeds. FFT dominant periods again sit at half the window (~10100) on almost every series, a trend/window artifact. Predator lags on the new seeds are long (3200–7800; 19d mean 5675 [4750, 6600]) with high r (0.72–0.96), but without a repeatable ACF period they are not a Lotka–Volterra cycle.

## Verdict

**The 19c 3/4 result at `kpred=0.005` does not hold up as a “most seeds” coexistence regime.** The eight new seeds are 4/8 both held. Pooled over 12 seeds, both persist as bodies on **7/12** (prey on all 12); the binomial interval [0.28, 0.85] is wide and consistent with both a weak minority and a strong majority. Predator-body loss is still the failure mode, and its timing is seed-dependent rather than a tight horizon. Held runs remain an edge (few predator objects, large prey). There is still no reliable oscillation period. `kpred=0.005` is better than 19b’s 0.015 (2/4) and 19c’s 0.01 (1/4), but it is not a stable 20k coexistence window.

Do not start an arms-race arm, and do not start a paid follow-up, without an explicit yes.

## Engine notes (what 19d actually ran)

Same engine as 19c on main `c472f22`: `coarse=True`, `kb=1`, `kp=1`, `pred_pursuit_evolve=1` (default, not set in the eco dict), `batch=1`. All eight `b19d_*.pkl` jobs have `partial=False` on every `run_eco` result, with logs through t=20000 (101 frames); none of the 8 new runs, and none of the 4 pooled 19c k005 runs, were incomplete.
