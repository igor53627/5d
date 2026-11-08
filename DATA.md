# Data Manifest

This document describes the heavy data files (simulation logs, snapshots, arrays) used in the Hyperlife experiments. The project was previously called Feigenbaum-Lenia; Python module names are unchanged.

## Current Storage

Data is currently stored in a **Modal Volume** attached to experiment runs.

## Planned publication

Heavy arrays will be published on the Hugging Face dataset [igor53627/5d](https://huggingface.co/datasets/igor53627/5d) (coming soon).

## Data Types

| Type | Format | Description |
|------|--------|-------------|
| Simulation logs | `.pkl` | Pickled dictionaries with time series, object tracking, metrics |
| Snapshots | `.npy`, `.npz` | NumPy arrays: lattice states, genome fields |
| Optimizer checkpoints | `.pkl` | CMA-ES state (mean, covariance, step-size) + fitness history; **not** restartable lattice state |
| Ecology logs | `.pkl` | Object tracking and 2D projections from `run_eco` (`experiments/part10/glider_eco.py`). Object lists (mass, center, phenotype) are written every `every` steps (**default 100**). A projection is stored only on an object-log step whose time is a multiple of `proj_every` (**default 500**), because the check runs only at those steps: `(k * every) % proj_every == 0`. The effective projection interval is therefore lcm(`every`, `proj_every`): 500 steps at the defaults, and 1000 steps when `every=200` and `proj_every=500`. `partial=True` on in-progress checkpoints; `run_eco` sets `partial=False` when a run finishes, and Modal saves that result. **Not** full N-D lattice snapshots. |
| Figures | `.png` | Rendered plots (in `figures/`) |
| Videos | `.mp4` | Rendered animations (in `videos/`) |

## Per-Part Data

| Part | Primary Data | Description | Size (approx) |
|------|--------------|-------------|---------------|
| 5 | `glider_es5.run_cma` | CMA-ES optimizer checkpoints: population mean, covariance matrix, step-size, and fitness history per generation. Not lattice state. | ~50 MB |
| 6 | Continuation sweeps | Bifurcation diagrams and parameter scans | ~100 MB |
| 7 | Collision outcomes | Pairwise collision result matrices | ~200 MB |
| 8–9 | Replication/growth logs | Lineage tracking and division events | ~500 MB |
| 10–14 | `glider_eco.run_eco` | Ecology time series: object lists every `every` steps (default 100), and the spatial mean of the food field `F` on those same steps. 2D projections are kept only when that object-log time is a multiple of `proj_every` (default 500), so the effective interval is lcm(`every`, `proj_every`): 500 steps at the defaults, and 1000 steps for `every=200`, `proj_every=500`. `partial=True` while running, `partial=False` when complete. Not full N-D fields. | ~2 GB |
| 15–16 | Brain evolution logs | Micro-brain weight evolution and intake correlations | ~1 GB |
| 17–18 | Poison/seasonal logs | Seasonal adaptation metrics and memory tests | ~1 GB |

## File Naming Convention

- `results/eco{part}/b{part}m_*.pkl` — main experiment batches (example pattern, not exhaustive)
- `results/eco{part}/main{part}.pkl` — aggregated summary tables
- Part 10 batch logs are `eco10m_*.pkl`. Part 12 batch logs are `eco12m_*.pkl`; `plot_eco12.py` writes the aggregate `main_metrics.pkl` next to them.
- `archive/specs/*.json` — experiment configuration files

## Access

Until the Hugging Face dataset [igor53627/5d](https://huggingface.co/datasets/igor53627/5d) is published (coming soon), contact the author for data access.

---

*This manifest will be updated as data is published.*
