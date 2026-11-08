# Part 14 — 6D exploratory (budget $30)
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
