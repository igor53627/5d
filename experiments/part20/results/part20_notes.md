# Part 20 notes: oscillator ablation (setup only)

**Status: no population has been run.** This file records the question, the CPU probes, the spec and the launch conditions. Issue: #9.

## Question

Every earlier part runs the Rössler oscillators inside the bodies. Do Parts 10–13 need them? The oscillators act on the mass field through two parameters of the soliton:

- `beta`: growth feedback from local synchrony;
- `chi`: advection along the phase gradient.

With `beta = chi = 0` the mass field, the heritable field, the food and `c` do not depend on the oscillator state. `tests/test_oscillator_ablation.py` checks this on the `fl_eco3` step: two different oscillator states give equal fields, and with either term alone at its published value they give different mass fields, so the ablation needs both zeros.

## CPU probes

`probe20.py` prepares one soliton with the parameters of each arm (600 steps from a random blob, as `run_eco` does) and runs it alone for 3000 steps in 4D `40^4`, with no food and no ecology. The soliton parameters are those of the Part 13 4D run `sel_4D56_s201`. Speeds are in cells per step. "In preparation" is the speed that `glider_coll.soliton` returns at the end of the 600 steps. "After" is the net displacement of the centre of mass over the last 1500 steps. Output: `probe20.json`.

| Seed | Arm        | Speed in preparation | Speed after | Mass ratio at end | Bodies at end (unit mass) |
|-----:|------------|---------------------:|------------:|------------------:|---------------------------|
|    3 | baseline   |               0.0452 |      0.0491 |             1.000 | 0.988                     |
|    3 | chi=0      |               0.0475 |      0.0479 |             1.000 | 0.988                     |
|    3 | beta=0     |               0.0117 |      0.0457 |             1.000 | 0.988                     |
|    3 | beta=chi=0 |               0.0055 |      0.0459 |             1.000 | 0.989                     |
|  201 | baseline   |               0.0520 |      0.0499 |             1.000 | 0.988                     |
|  201 | chi=0      |               0.0525 |      0.0484 |             1.000 | 0.988                     |
|  201 | beta=0     |               0.0101 |      0.0529 |             1.000 | 0.552, 0.434              |
|  201 | beta=chi=0 |               0.0049 |      0.0475 |             1.000 | 0.989                     |

Readout:

1. **The soliton forms and moves without the feedback.** With `beta = chi = 0` it is one body of 0.989 unit masses on both seeds, the total mass is conserved, and the speed is 0.046–0.047 against 0.049–0.050 with the feedback. Movement of the body does not need the oscillators.
2. **Without `beta` the body starts slowly.** At the end of the 600-step preparation it moves at 0.005–0.012, against 0.045–0.052 with `beta`. In an eco run the founders without feedback are thus nearly at rest at step 0. This is a difference between the twins that is not ecology, and the analysis must not read the first steps as an effect of the oscillators on selection.
3. **One arm split without food.** With `beta = 0` and the published `chi`, the seed 201 body ended as two bodies of 0.55 and 0.43 unit masses. Its "speed after" is the speed of the centre of mass of both, not of a body. This arm is not in the spec. It is one seed of two, at one lattice size, so it shows only that a split without food is possible in that arm; no rate follows from it.

Two seeds and one lattice size are a check that the experiment can run. They are not a result about populations.

## Spec

`spec20.py` writes `archive/specs/b20_pilot.json`: the Part 13 4D `56^4` world (12 founders, 30,000 steps, `every=200`), selection (`pk=1`) and neutral (`pk=0`) arms, 4 new seeds (221–224), each run with a twin that differs only in `beta = chi = 0`. That is 8 calls and 16 runs. Both twins of a pair run in one container, so they share the GPU type. `run_eco` prepares the soliton with the parameters of the run, so the twin without feedback also forms without feedback.

Readouts, fixed before the run, each as a paired difference (with feedback minus without) over the 4 seeds:

- plateau count of individuals, neutral arm;
- selection response: mean `p0` of the selection arm minus the neutral arm after 5k steps;
- divisions per run and parent–child correlation of `p0`, with ancestors identified by heritable traits (`src/lineage_tracking.py`);
- Price selection term in growth (`src/price_equation.py`).

Four pairs can show only a large difference. With 4 pairs the smallest two-sided exact Wilcoxon p-value is 0.125, so the pilot cannot give p < 0.05 with that test; it decides whether a larger run is worth its cost.

## Launch (paid, not started)

    modal run src/modal_app.py::spawn_h100 --kind ecolist --spec archive/specs/b20_pilot.json --prefix b20p --timeout-seconds 1200 --cap-dollars 15

- Estimate: Part 13 4D `56^4` containers with 2 runs took 360–400 s on an H100, so 8 containers are about 0.85 GPU-hours, about $3.4 at $3.95/h. This is an estimate from the Part 13 logs, not a measurement of this spec.
- Spend cap (issue #14): `spawn_h100` enforces it. Before it starts a container it prints the number of calls, the timeout, the rate and the worst-case spend, which is calls × timeout × $3.95/h. It starts nothing when the worst case exceeds `--cap-dollars` (default $15, the cap of Part 19d). `--timeout-seconds` has no usable default: the command without it is refused, so the 4-hour timeout of `gpu_job_h100` (8 × 4 h × $3.95 = $126.40 for this spec) cannot apply by accident. The arithmetic is in `src/spend_cap.py`.
- Timeout: 1200 s per container. That is 3 times the upper estimate of 400 s, a margin of 800 s. The worst case for 8 calls is 8 × 1200 s = 2.67 GPU-hours, $10.53. The largest timeout that passes a $15 cap with 8 calls is 1708 s ($14.99). 1200 s leaves $4.47 for the charges that the check does not count: CPU and memory (about $0.70 per container-hour in `archive/results/perf_notes.md`, thus about $1.9 at the worst case), the start of a container and its idle seconds.
- A container that hits the timeout: Modal stops it, it returns nothing, and the launcher stops with an error at that call. `run_eco` saves to the Volume every 20 snapshots (4000 steps) and `run_eco_list` saves after each run. `b20p_<i>.pkl` then holds the finished twin, if there is one, and the other twin up to its last save, with `partial: True`. The `osc` twin runs first, so a timeout cuts the `noosc` twin. A partial run is not a result for the readouts above: the pair must run again with a longer timeout, and that is a new paid launch. No run of this code has hit a timeout, so this is read from the code, not observed.
- Start one container first as a timing probe, as in Part 19d: a spec with one of the 8 calls (no such file is committed) and `--timeout-seconds 1200` has a worst case of $1.32. Start the other 7 only if the probe time leaves room under the timeout; their worst case at 1200 s is $9.22. The check covers one launch, not the sum of launches: give the second launch `--cap-dollars` of 15 minus the probe spend. Give it also a prefix that differs from the probe's (for example `b20p1` for the probe and `b20p` for the other 7): `spawn_h100` names each result `<prefix>_<i>.pkl` with `i` counted from 0 in each launch, so two launches with one prefix overwrite each other's files.
- The launch was not run and the new options were not exercised against Modal. `Function.with_options(timeout=...)` needs Modal 1.4.3 or later; the lock file has 1.6.4 for Python 3.10 and later.
- A launch needs an explicit yes from the owner.
