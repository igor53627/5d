# Part 14 — 6D exploratory (budget $30)

**Correction (2026-11-19, lineage tracking fix).** h, the division and merge rates and lin_eff below, and the S columns of [part14_table.md](part14_table.md), use nearest-in-space ancestors. These differ from the ancestor identified by the heritable traits (p0, p1) for 50% of descendants in the selection run and 32% in the neutral run. There is one run per condition, so no test over seeds is possible. With ancestors identified by p1 alone, the parent–child correlation of p0 is 0.43 (selection) and 0.51 (neutral), against h .36 and .42; the values from the rule that contains p0 (0.72, 0.73) are circular. "lin_eff 1" stands for the selection run under both trait rules; for the neutral run the rules give 1.9 (p0 and p1) and 1.0 (p1 alone), so it is not resolved. Divisions per run are 201 / 241 (selection) and 57 / 77 (neutral) with the two trait rules, against 295 and 200. S in growth is +0.047 / +0.023 in the selection run (table: +0.016) and −0.021 / −0.028 in the neutral run (table: +0.032); the neutral term is as large as the selection term, and p0 alone and p1 alone pick the same ancestor for only 64.5% of descendants in the selection run, so the Price terms give no conclusion in 6D. The division and merge rates are not recomputed. Results that do not use ancestors (n, M/n, sd, dp0) stand. See [tracking_notes.md](../../tracking_fix/results/tracking_notes.md).

- 11:07 UTC: start; baseline billing = $188.06
- 11:08 UTC: launched feasibility: p14speed (perf11 speed all5, 6D L=20/24/28/32, B300) + p14feas (ecolist 6D L24 & L28, 2000 steps, 8 founders, B300)
- speed (B300, all5, 6D): L20 33 st/s 13GB; L24 11.9 st/s 41GB; L28 4.7 st/s 99GB; L32 OOM (needs ~190GB). Soliton 6D exists: M1=140.6k (×5.9 vs 5D), v0=.071.
- feas L24 (8 founders, 2000 steps, wall ~430 s incl. setup): 1–5 detected objects, masses 0.6–9 M1 → solitons touching/merging into blobs; box too cramped. L28 test running.
- feas L28 (8 founders): 6–7 objects, masses grow to 1.4–4.5 M1 by 2000 steps; setup (soliton+placement on CPU) ~645 s, then ~4.9 steps/s incl. logging. Part 14 spent so far ≈ $4.4.
- Decision: 6D L=28 (482M cells, coarse 14^6), 12 founders, 20k steps, seed 301, sel+neu on B300. Estimate ≈ $10/run (~80 min).
- 11:36 UTC: launched eco14 (2 × B300)
- 11:57 UTC: resumed; eco14 still running (2 tasks), Part 14 spend ≈ $9.9
- 12:12 UTC: sel at 8k, neu at 4k steps (~4.6 st/s). Projection: sel done ~12:55, neu ~13:10 UTC; Part 14 total ≈ $26. Guard: if spend > $28.5, stop app (partials saved every ~4k steps).
- 12:22 UTC: analyze14.py written and tested on partials (sel 8k, neu 4k): n≈10, lin_eff=1, sd low from start (overlapping founders).
- 12:47 UTC: resumed; eco14 still running (2 containers); Part 14 spend ≈ $22.9
- 12:55 UTC: sel_6D28 finished 20k (wall 4658 s); neu at 16k, ETA ~13:08 UTC
- 13:05 UTC: both 6D runs done (20k), 0 apps. Part 14 spend ≈ $26.1 (214.19−188.06; lag may add <$1).
  sel: n 9→~16, M/n 2.9 M1, sd .33→.10 (already .15 at 2k), dp0_5k .13, dp0 .17, h .36, div 1.17, merges 1.20 /ind/1k, lin_eff 1.
  neu: n ~10–11, M/n 2.8, sd .33→.20 (by 2k, then flat), h .42, div .97, merges 1.06, lin_eff 1.
