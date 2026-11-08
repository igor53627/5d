# Feigenbaum-Lenia

N-dimensional cellular automata living on the edge of chaos: Lenia-style smooth kernels meet Rössler oscillators with self-adjusting bifurcation parameters.

## Overview

Each cell in an N-dimensional lattice (default 4D, 24⁴ = 331,776 cells) is a strange attractor: a Rössler oscillator $(x, y, z)$ with a local "genome" parameter $c$ that determines its position on the Feigenbaum period-doubling diagram. Cells measure their **local Lyapunov exponent** and **local synchronization** with neighbors, then adjust their genome: too chaotic → move toward order; neighbors too synchronous → move toward chaos. The automaton self-organizes to the edge of chaos.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
# GPU build of JAX (optional): pip install -U "jax[cuda12]"
```

## Running

```bash
# Quick CPU demo (3D, L=8; writes animation.mp4)
python -m feigenbaum_lenia --N 3 --L 8 --steps 80 --every 10 --out demo_run

# Full simulation (4D, L=24, about 6 minutes on CPU)
python -m feigenbaum_lenia --out full_run

# GPU, after `pip install -U "jax[cuda12]"`
JAX_PLATFORMS=cuda python -m feigenbaum_lenia --out gpu_run
```

## Project Structure

```
src/                 Core library: models, kernels, dynamics
experiments/         Per-part experiment scripts and results
  part05/            ES-based glider search
  part06/            Continuation analysis
  ...
  part18/            Seasonal environments
figures/             PNG figures by part
videos/              MP4 videos by part
docs/                Notes and older reports
archive/             Early searches, benchmarks, reservoir experiments
```

## Table of Contents: Parts 1–18b

| Part | Topic | Notes | Key Figure |
|------|-------|-------|------------|
| 5 | ES-based glider search | [part5_notes.md](experiments/part05/results/part5_notes.md) | [glider_es5_fitness.png](figures/glider_es5_fitness.png) |
| 6 | Continuation analysis | [part6_notes.md](experiments/part06/results/part6_notes.md) | [glider_cont_bifurcation.png](figures/glider_cont_bifurcation.png) |
| 7 | Collision experiments | [part7_notes.md](experiments/part07/results/part7_notes.md) | [glider_part7_coll4D.png](figures/glider_part7_coll4D.png) |
| 8 | Replication | [part8_notes.md](experiments/part08/results/part8_notes.md) | [glider_part8_replication.png](figures/glider_part8_replication.png) |
| 9 | Growth and selection | [part9_notes.md](experiments/part09/results/part9_notes.md) | [glider_part9_growth.png](figures/glider_part9_growth.png) |
| 10 | Ecology experiments | [part10_notes.md](experiments/part10/results/part10_notes.md) | [glider_part10_population.png](figures/glider_part10_population.png) |
| 12 | Population dynamics | [part12_notes.md](experiments/part12/results/part12_notes.md) | [glider_part12_population.png](figures/glider_part12_population.png) |
| 13 | Dimension comparison | [part13_notes.md](experiments/part13/results/part13_notes.md) | [glider_part13_metrics.png](figures/glider_part13_metrics.png) |
| 14 | 6D experiments | [part14_notes.md](experiments/part14/results/part14_notes.md) | [glider_part14_dims.png](figures/glider_part14_dims.png) |
| 15 | Micro-brain metrics | [part15_notes.md](experiments/part15/results/part15_notes.md) | [glider_part15_timeseries.png](figures/glider_part15_timeseries.png) |
| 15b | Brain encoding analysis | [part15b_notes.md](experiments/part15/results/part15b_notes.md) | [glider_part15b_weights.png](figures/glider_part15b_weights.png) |
| 16 | Fusion catastrophe | [part16_notes.md](experiments/part16/results/part16_notes.md) | [glider_part16_fusion.png](figures/glider_part16_fusion.png) |
| 17 | Poison food (4D) | [part17_notes.md](experiments/part17/results/part17_notes.md) | [glider_part17_dynamics.png](figures/glider_part17_dynamics.png) |
| 18 | Seasonal environments | [part18_notes.md](experiments/part18/results/part18_notes.md) | [glider_part18_dynamics.png](figures/glider_part18_dynamics.png) |
| 18b | Memory and adaptation | [part18b_notes.md](experiments/part18/results/part18b_notes.md) | [glider_part18b_dynamics.png](figures/glider_part18b_dynamics.png) |

## Data

Heavy arrays (simulation logs, snapshots, trained models) live in a **Modal Volume** and are planned for migration to **Hugging Face Datasets**. See [DATA.md](DATA.md) for the manifest and access instructions.

## License

Research code — see individual files for details.
