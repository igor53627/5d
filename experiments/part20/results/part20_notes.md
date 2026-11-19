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

    modal run src/modal_app.py::spawn_h100 --kind ecolist --spec archive/specs/b20_pilot.json --prefix b20p

- Estimate: Part 13 4D `56^4` containers with 2 runs took 360–400 s on an H100, so 8 containers are about 0.85 GPU-hours, about $3.4 at $3.95/h. This is an estimate from the Part 13 logs, not a measurement of this spec.
- `gpu_job_h100` has a timeout of 4 hours per container. No code enforces the $15 spend cap of Part 19, so the worst case of this command is 8 × 4 h × $3.95 = $126. Use the Part 19d procedure: start one container as a timing probe, then start the rest only if the probe time leaves room under the cap.
- A launch needs an explicit yes from the owner.
