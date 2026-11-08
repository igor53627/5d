# Data Manifest

This document describes the heavy data files (simulation logs, snapshots, arrays) used in the Feigenbaum-Lenia experiments.

## Current Storage

Data is currently stored in a **Modal Volume** attached to experiment runs.

## Planned Migration

Heavy arrays are planned for migration to **Hugging Face Datasets** for easier access and reproducibility.

## Data Types

| Type | Format | Description |
|------|--------|-------------|
| Simulation logs | `.pkl` | Pickled dictionaries with time series, object tracking, metrics |
| Snapshots | `.npy`, `.npz` | NumPy arrays: lattice states, genome fields |
| Optimizer checkpoints | `.pkl` | CMA-ES state (mean, covariance, step-size) + fitness history; **not** restartable lattice state |
| Ecology logs | `.pkl` | Object tracking and 2D projections (partial=True); **not** full N-D lattice snapshots |
| Figures | `.png` | Rendered plots (in `figures/`) |
| Videos | `.mp4` | Rendered animations (in `videos/`) |

## Per-Part Data

| Part | Primary Data | Description | Size (approx) |
|------|--------------|-------------|---------------|
| 5 | `glider_es5.run_cma` | CMA-ES optimizer checkpoints: population mean, covariance matrix, step-size, and fitness history per generation. Not lattice state. | ~50 MB |
| 6 | Continuation sweeps | Bifurcation diagrams and parameter scans | ~100 MB |
| 7 | Collision outcomes | Pairwise collision result matrices | ~200 MB |
| 8–9 | Replication/growth logs | Lineage tracking and division events | ~500 MB |
| 10–14 | `glider_eco.run_eco` | Ecology time series: object lists (mass, center, phenotype), 2D max-projections, and food totals per snapshot. Partial logs, not full N-D fields. | ~2 GB |
| 15–16 | Brain evolution logs | Micro-brain weight evolution and intake correlations | ~1 GB |
| 17–18 | Poison/seasonal logs | Seasonal adaptation metrics and memory tests | ~1 GB |

## File Naming Convention

- `results/eco{part}/b{part}m_*.pkl` — main experiment batches
- `results/eco{part}/main{part}.pkl` — aggregated summary tables
- `specs/*.json` — experiment configuration files

## Access

Until Hugging Face migration is complete, contact the author for data access.

---

*This manifest will be updated as data is migrated.*
