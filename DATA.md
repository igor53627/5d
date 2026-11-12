# Data Manifest

Simulation logs for the Hyperlife experiments are published on the Hugging Face dataset [igor53627/5d](https://huggingface.co/datasets/igor53627/5d).

- Public, license CC-BY-4.0.
- 515 files under `part03`–`part19`, 1,630,191,638 bytes (1.63 GB). The dataset card and `.gitattributes` are extra.
- Layout is `partXX/`. `part18/` includes the part 18 runs (`eco18/`), the part 18b runs (`eco18b/`, plus `warm18b.pkl`), the part 18c runs (`eco18c/`, plus `manifest.json` and summaries), the part 18d runs (`eco18d/`: pilots `b18dp_*`/`b18dp2_*`, full `b18d_*`, `manifest.json`, summaries), the part 18e pilot (`eco18e/`: `b18ep_*`, `manifest.json`, summary), the part 18f pilot (`eco18f/`: `b18fp_*`, `manifest.json`, summary), the part 18g confirmatory full (`eco18g/`: `b18g_*`, `manifest.json`, summary), the part 18h competition pilot (`eco18h/`: `b18hp_*`, `manifest.json`, summary), and the part 18i more-seeds full (`eco18i/`: `b18i_*`, `manifest.json`, summary; pool with 18g). `part19/` holds the part 19 predator–prey pilot (`eco19/`: `b19p_*`, `manifest.json`, summary), the part 19b confirmatory full (`eco19b/`: `b19b_*`, `manifest.json`, summary), and the part 19c low-side kpred scan (`eco19c/`: `b19c_*`, `manifest.json`, summary).

## Download

```bash
huggingface-cli download igor53627/5d --repo-type dataset
```

```python
from huggingface_hub import snapshot_download
snapshot_download("igor53627/5d", repo_type="dataset", allow_patterns="part07/*")
```

`huggingface_hub` is only needed to download the dataset. It is not a dependency of the simulator.

## Per-part sizes

Sizes are the sum of file bytes on the dataset, in decimal MB (1 MB = 10^6 bytes).

| Part | What is stored | Files | Size |
|------|----------------|------:|-----:|
| 03 | Conditional-Lyapunov and reservoir tuning logs (`alife_L48.pkl`, `final_L*`, `tune2_L8_*`) | 13 | 147.45 MB |
| 04 | Glider search batches (`glider/glider_s*`, `glider_big*`) | 36 | 366.98 MB |
| 05 | Asymmetry continuation, short ES, and swimmer checks (`cont/`, `es/`, `glider/swim_*`, `ver/`) | 18 | 33.30 MB |
| 06 | Long 5D CMA-ES checkpoints and verification (`es5/`, `ver5/`). Optimizer state and fitness history, not full lattice snapshots. | 8 | 4.77 MB |
| 07 | Collision and isotropy logs | 19 | 20.91 MB |
| 08 | Replication logs | 24 | 22.50 MB |
| 09 | Growth and selection logs | 22 | 19.16 MB |
| 10 | Ecology logs | 15 | 39.71 MB |
| 11 | Performance profiles, validation, and the bench pickles `part11/bench_b300_0.pkl`, `part11/bench_h100_0.pkl` | 9 | 4.53 MB |
| 12 | Carrying-capacity ecology logs | 39 | 89.79 MB |
| 13 | Dimension comparison | 47 | 672.63 MB |
| 14 | 6D exploratory run | 5 | 0.32 MB |
| 15 | Micro-brain logs | 41 | 62.43 MB |
| 16 | Fusion test | 7 | 2.84 MB |
| 17 | Poison-food logs | 20 | 19.06 MB |
| 18 | Seasons, including 18b–18i (`eco18c/`–`eco18g/` as before; `eco18h/` competition pilot `b18hp_*`; `eco18i/` more seeds `b18i_*` 855–858 × cue/nocue/planted × 20k; manifests, summaries) | 176 | 107.41 MB |
| 19 | Predator–prey: pilot (`eco19/`: `b19p_*` seeds 191–192 × low/mid/high kpred × 6k, `manifest.json`, summary), confirmatory 19b (`eco19b/`: `b19b_*` seeds 193–196 × low/mid/prey-only × 20k, `manifest.json`, summary), and low-side scan 19c (`eco19c/`: `b19c_*` seeds 197–200 × kpred 0.005/0.01 × 20k, `manifest.json`, summary) | 16 | 16.39 MB |

## Ecology log cadence

`run_eco` (`experiments/part10/glider_eco.py`) writes object lists (mass, center, phenotype) every `every` steps (default 100), and the spatial mean of the food field `F` on those same steps. A 2D projection is stored only on an object-log step whose time is a multiple of `proj_every` (default 500), because the check is `(k * every) % proj_every == 0`. The effective projection interval is lcm(`every`, `proj_every`): 500 steps at the defaults, and 1000 steps when `every=200` and `proj_every=500`. Checkpoints use `partial=True` while a run is in progress and `partial=False` when it finishes. These pickles are not full N-D lattice snapshots. For `model="pred"` (Part 19), predator objects and masses are stored as `plog` / `ptot` beside the usual prey `log` / `tot`.

## Other files in the repo

- Figures (`.png`) are in `figures/`. Videos (`.mp4`) are in `videos/`.
- Experiment configuration files shipped with the code are `archive/specs/*.json`.
- `results/eco{part}/b{part}m_*.pkl` and `results/eco{part}/main{part}.pkl` are example names inside the dataset trees, not a second copy in this git repo.
