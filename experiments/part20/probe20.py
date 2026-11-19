"""Part 20 CPU probe: does the eco soliton form and move without the oscillator feedback?

Usage: python probe20.py [OUTPUT_JSON]

Each arm prepares the soliton with its own parameters, as ``run_eco`` does (600 steps from a
random blob), and then runs it alone for 3000 steps on a 4D lattice of 40^4 cells. The arms
set ``beta`` (growth feedback from local synchrony), ``chi`` (advection along the phase
gradient), or both to 0. There is no food and no ecology here: the probe only shows whether
the body survives and moves, which a paired eco run needs.

The probe uses the CPU only and takes about 20 minutes.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

# The probe is documented as CPU only, so the platform is set before the first import of jax.
os.environ["JAX_PLATFORMS"] = "cpu"

# The import of _path puts src/ and experiments/part*/ on sys.path, so it must come first.
# isort: off
import _path  # noqa: F401
import fl_glider
import glider_coll
import spec20
# isort: on

DIMENSION_COUNT: int = 4
LATTICE_SIZE: int = 40
PREPARE_STEPS: int = 600
RUN_STEPS: int = 3000
CENTRE_EVERY_STEPS: int = 100
SEEDS: tuple[int, ...] = (3, 201)
BODY_MASS_MIN: float = 0.15
ARMS: dict[str, dict[str, float]] = {
    "baseline": {},
    "chi=0": {"chi": 0.0},
    "beta=0": {"beta": 0.0},
    "beta=chi=0": {"beta": 0.0, "chi": 0.0},
}
DEFAULT_OUTPUT: Path = Path(__file__).resolve().parent / "results" / "probe20.json"


def net_speed(centres: Sequence[np.ndarray]) -> float:
    """Net displacement per step over the second half of the centre track, in cells."""
    track: np.ndarray = np.array(centres[len(centres) // 2 :])
    steps: np.ndarray = (np.diff(track, axis=0) + LATTICE_SIZE / 2) % LATTICE_SIZE
    steps = steps - LATTICE_SIZE / 2
    return float(np.linalg.norm(steps.sum(0)) / (len(steps) * CENTRE_EVERY_STEPS))


def probe_arm(soliton: dict, seed: int) -> dict:
    """Prepare one soliton, run it alone and report its mass, bodies and speed."""
    state, velocity = glider_coll.soliton(
        soliton, DIMENSION_COUNT, LATTICE_SIZE, seed, pre=PREPARE_STEPS
    )
    founder_mass: float = float(state["A"].sum())
    state, centres = glider_coll._sim(
        soliton, state, RUN_STEPS, DIMENSION_COUNT, LATTICE_SIZE, every=CENTRE_EVERY_STEPS
    )
    bodies, _ = fl_glider.components(state["A"], 0.05, min_mass=BODY_MASS_MIN * founder_mass)
    return {
        "founder_mass": founder_mass,
        "speed_in_preparation": float(np.linalg.norm(velocity)),
        "speed_after": net_speed(centres),
        "mass_ratio_end": float(state["A"].sum()) / founder_mass,
        "mass_max_end": float(state["A"].max()),
        "body_masses_end": [round(body["mass"] / founder_mass, 3) for body in bodies],
    }


def main(arguments: Sequence[str]) -> int:
    """Run every arm on every seed and write the results as JSON."""
    if len(arguments) > 1:
        print(__doc__, file=sys.stderr)
        return 2
    output: Path = Path(arguments[0]) if arguments else DEFAULT_OUTPUT
    published: dict = spec20.source_runs()["sel"]["c"]
    results: dict[str, dict] = {}
    for seed in SEEDS:
        for arm, overrides in ARMS.items():
            results[f"{arm}/s{seed}"] = probe_arm({**published, **overrides}, seed)
            print(f"{arm}/s{seed}", results[f"{arm}/s{seed}"], flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
