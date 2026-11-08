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
| Checkpoints | `.pkl` | Full simulation state for restart |
| Figures | `.png` | Rendered plots (in `figures/`) |
| Videos | `.mp4` | Rendered animations (in `videos/`) |

## Per-Part Data

| Part | Primary Data | Size (approx) |
|------|--------------|---------------|
| 5 | ES optimization logs | ~50 MB |
| 6 | Continuation sweeps | ~100 MB |
| 7 | Collision outcomes | ~200 MB |
| 8–9 | Replication/growth logs | ~500 MB |
| 10–14 | Ecology time series | ~2 GB |
| 15–16 | Brain evolution logs | ~1 GB |
| 17–18 | Poison/seasonal logs | ~1 GB |

## Access

Until Hugging Face migration is complete, contact the author for data access.

---

*This manifest will be updated as data is migrated.*
