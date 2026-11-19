"""Run one Part 13 world with tracer fields that record the true ancestor of every body.

Usage: python run_truth.py TAG BOX_SIZE STEPS SEED [OUTPUT_DIRECTORY]

TAG names a run of ``archive/specs/eco13m.json``, for example ``neu_4D56_s201``. The run uses
that world (soliton, food, mutation, 12 founders), with the box size, step count and seed
given here. The box size is at most 64 cells: ``run_eco`` embeds the soliton in a larger
field for larger boxes, and this script does not copy that code.

Every 200 steps the script writes the objects, as ``run_eco`` does, and gives each object its
own tracer (``src/lineage_tracer.py``). At the next snapshot the tracer content of an object
is the share of its mass that comes from each earlier object. The simulated state is the
state that ``fl_eco3.run_chunk`` produces: the tracers are passive.

The output is ``truth_<TAG>_L<BOX_SIZE>_s<SEED>.json`` with the log in the format of the
published logs and one share matrix for each pair of snapshots. The run uses the CPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np

# The import of _path puts src/ and experiments/part*/ on sys.path, so it must come first.
# isort: off
import _path  # noqa: F401
import fl_eco3
import fl_glider
import glider_coll
import glider_eco
import lineage_tracer
from helpers import invariant
# isort: on

import jax.numpy as jnp

ROOT: Path = Path(__file__).resolve().parents[2]
SOURCE_SPEC: Path = ROOT / "archive" / "specs" / "eco13m.json"
DEFAULT_OUTPUT: Path = Path(__file__).resolve().parent / "results"
BOX_SIZE_MAX: int = 64
PREPARE_STEPS: int = 600
TRAIT_COUNT: int = 2
POOL: int = 2
OBJECT_THRESHOLD: float = 0.05
OBJECT_MASS_MIN: float = 0.15
CHANNEL_BLOCK: int = 8
SHARE_MIN: float = 1e-4
RUN_CODE_PATHS: tuple[str, ...] = (
    "src",
    "archive/specs/eco13m.json",
    "experiments/part06",
    "experiments/part10",
    "experiments/tracking_truth/run_truth.py",
)


def source_run(tag: str) -> dict:
    """The run of the Part 13 spec with this tag."""
    runs = [run for call in json.loads(SOURCE_SPEC.read_text()) for run in call["runs"]]
    matches = [run for run in runs if run["tag"] == tag]
    if len(matches) != 1:
        raise ValueError(f"{SOURCE_SPEC} has {len(matches)} runs tagged {tag}")
    return matches[0]


def initial_world(run: dict, box_size: int, seed: int) -> tuple[dict, tuple, float]:
    """State, engine arguments and unit mass, built as ``glider_eco.run_eco`` builds them."""
    dimension: int = run["N"]
    soliton, options = run["c"], run["opts"]
    founder, _ = glider_coll.soliton(soliton, dimension, box_size, seed, pre=PREPARE_STEPS)
    unit_mass: float = float(founder["A"].sum())
    crowding: float = fl_eco3.U1_of(founder["A"], box_size, dimension, **options)
    world: dict = {**soliton, **run["ecos"][0], "U1": crowding}
    rng = np.random.default_rng(seed)
    base, traits, _ = glider_eco.init_multi(
        founder,
        rng,
        run["n_init"],
        box_size,
        dimension,
        TRAIT_COUNT,
        (0.0, 0.0),
        run["p_spread"],
        run["min_sep"],
    )
    fields: dict = {name: base[name] for name in ("A", "x", "y", "z", "c")}
    state: dict = fl_eco3.make_state(fields, world, K=TRAIT_COUNT, seed=seed * 100)
    state["Q"] = traits.copy()
    peaks: tuple = tuple(soliton["peaks"])
    engine: tuple = (
        fl_eco3.params(world),
        jnp.asarray(fl_glider.ring_kernel_np(box_size, dimension, soliton["R"], peaks)),
        jnp.asarray(fl_glider.ring_kernel_np(box_size, dimension, 3.0)),
        fl_eco3.kernels(box_size, dimension, **options),
    )
    return {name: jnp.asarray(value) for name, value in state.items()}, engine, unit_mass


def labelled_objects(state: dict, unit_mass: float) -> tuple[list, np.ndarray]:
    """Objects as ``glider_eco._objects`` lists them, and their labels on the full lattice.

    The label field holds 0 outside every object and ``i + 1`` in the cells of object ``i``.
    """
    from scipy import ndimage

    mass: np.ndarray = np.asarray(state["A"])
    dimension, box_size = mass.ndim, mass.shape[0]
    pooled_mass: np.ndarray = glider_eco._pool(mass, dimension, box_size, POOL)
    pooled_traits: np.ndarray = glider_eco._pool(np.asarray(state["Q"]), dimension, box_size, POOL)
    objects: list = glider_eco._objects(pooled_mass, pooled_traits, unit_mass, POOL, dimension)
    _, components = fl_glider.components(pooled_mass, OBJECT_THRESHOLD, min_mass=0.0)
    identifiers: np.ndarray = np.unique(components)
    identifiers = identifiers[identifiers > 0]
    labels: np.ndarray = np.zeros(components.shape, np.int64)
    if len(identifiers):
        masses = np.asarray(ndimage.sum(pooled_mass, components, identifiers))
        kept = identifiers[masses * POOL**dimension >= OBJECT_MASS_MIN * unit_mass]
        for index, identifier in enumerate(kept):
            labels[components == identifier] = index + 1
    invariant(int(labels.max()) == len(objects), "labels do not match glider_eco._objects")
    for axis in range(dimension):
        labels = np.repeat(labels, POOL, axis=axis)
    return objects, labels


def sparse_shares(shares: np.ndarray) -> list[list]:
    """Entries of the share matrix above ``SHARE_MIN`` as ``[descendant, ancestor, share]``."""
    rows, columns = np.nonzero(shares > SHARE_MIN)
    return [[int(j), int(i), round(float(shares[j, i]), 5)] for j, i in zip(rows, columns)]


def git_revision() -> str:
    """Commit of the code, with ``-dirty`` when a file that the run uses has changes."""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *RUN_CODE_PATHS],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return revision + ("-dirty" if status else "")


def run(tag: str, box_size: int, step_count: int, seed: int) -> dict:
    """Run the world and return the log and the true shares for every pair of snapshots."""
    source: dict = source_run(tag)
    every: int = source["every"]
    invariant(box_size <= BOX_SIZE_MAX, f"box size {box_size} is above {BOX_SIZE_MAX}")
    invariant(step_count >= every, f"the step count must be at least {every}")
    invariant(step_count % every == 0, "the step count must be a multiple of the snapshot step")
    state, engine, unit_mass = initial_world(source, box_size, seed)
    log: list = []
    truth: list = []
    fractions: np.ndarray | None = None
    started: float = time.time()
    for snapshot in range(step_count // every + 1):
        if snapshot:
            state, moved = lineage_tracer.run_chunk_with_tracers(
                state, jnp.asarray(fractions), *engine, every, **source["opts"]
            )
            fractions = np.asarray(moved)
        mass: np.ndarray = np.asarray(state["A"])
        invariant(bool(np.isfinite(mass).all()), "the mass field is not finite")
        objects, labels = labelled_objects(state, unit_mass)
        if snapshot:
            shares, background = lineage_tracer.object_shares(fractions, mass, labels, len(objects))
            truth.append(
                {
                    "shares": sparse_shares(shares[:, : len(log[-1][1])]),
                    "background": [round(float(value), 5) for value in background],
                    "fraction_sum_error": lineage_tracer.fraction_sum_error(fractions, mass),
                }
            )
        log.append((snapshot * every, objects))
        channel_count: int = CHANNEL_BLOCK * (len(objects) // CHANNEL_BLOCK + 1)
        fractions = lineage_tracer.fractions_from_labels(labels, channel_count)
        elapsed: float = time.time() - started
        print(f"t={snapshot * every} objects={len(objects)} seconds={elapsed:.0f}", flush=True)
    return {
        "tag": tag,
        "box_size": box_size,
        "dimension": source["N"],
        "seed": seed,
        "steps": step_count,
        "every": every,
        "eco": source["ecos"][0],
        "options": source["opts"],
        "unit_mass": unit_mass,
        "git_revision": git_revision(),
        "wall_seconds": time.time() - started,
        "log": log,
        "truth": truth,
    }


def main(arguments: Sequence[str]) -> int:
    """Run one world and write its JSON file."""
    if not 4 <= len(arguments) <= 5:
        print(__doc__, file=sys.stderr)
        return 2
    tag, box_size, step_count, seed = arguments[0], *(int(value) for value in arguments[1:4])
    output_directory: Path = Path(arguments[4]) if len(arguments) == 5 else DEFAULT_OUTPUT
    output_directory.mkdir(parents=True, exist_ok=True)
    result: dict = run(tag, box_size, step_count, seed)
    output: Path = output_directory / f"truth_{tag}_L{box_size}_s{seed}.json"
    output.write_text(json.dumps(result) + "\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
