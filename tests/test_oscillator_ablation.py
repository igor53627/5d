"""Issue #9: with beta = chi = 0 the mass field does not depend on the Rossler oscillators."""

import importlib.util
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import fl_eco3
import fl_glider

ROOT = Path(__file__).resolve().parents[1]
LATTICE_SIZE = 24
STEP_COUNT = 5
MASS_SIDE_FIELDS = ("A", "Q", "F", "c")


def _spec20():
    location = ROOT / "experiments" / "part20" / "spec20.py"
    module_spec = importlib.util.spec_from_file_location("spec20", location)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _world(**overrides):
    run = _spec20().source_runs()["sel"]
    return {**run["c"], **run["ecos"][0], "U1": 0.1, **overrides}


def _run(world, oscillator_seed):
    """Run a few eco3 steps of one 2D body whose oscillators start from the given seed."""
    rng = np.random.default_rng(oscillator_seed)
    shape = (LATTICE_SIZE, LATTICE_SIZE)
    offset = np.arange(LATTICE_SIZE) - LATTICE_SIZE / 2
    mass = 0.8 * np.exp(-(offset[:, None] ** 2 + offset[None, :] ** 2) / 18.0)
    base = {
        "A": mass,
        "x": rng.uniform(-6, 6, shape),
        "y": rng.uniform(-6, 6, shape),
        "z": rng.uniform(0, 0.5, shape),
        "c": np.full(shape, 2.6),
    }
    state = fl_eco3.make_state(base, world, K=2, p0=np.array([0.3, -0.2]), seed=0)
    state = {name: jnp.asarray(value) for name, value in state.items()}
    params = fl_eco3.params(world)
    peaks = tuple(world["peaks"])
    mass_kernel = jnp.asarray(fl_glider.ring_kernel_np(LATTICE_SIZE, 2, float(world["R"]), peaks))
    oscillator_kernel = jnp.asarray(fl_glider.ring_kernel_np(LATTICE_SIZE, 2, 3.0))
    wide_kernels = fl_eco3.kernels(LATTICE_SIZE, 2)
    for _ in range(STEP_COUNT):
        state = fl_eco3.step(state, params, mass_kernel, oscillator_kernel, wide_kernels)
    return {name: np.asarray(value) for name, value in state.items()}


def test_mass_field_ignores_oscillators_without_feedback():
    """Goal: beta = chi = 0 removes the coupling. Method: two oscillator states, equal mass."""
    world = _world(beta=0.0, chi=0.0)
    first, second = _run(world, oscillator_seed=1), _run(world, oscillator_seed=2)
    assert not np.array_equal(first["x"], second["x"])
    for name in MASS_SIDE_FIELDS:
        assert np.array_equal(first[name], second[name]), name


def test_each_feedback_term_alone_couples_the_mass_field():
    """Goal: the ablation needs both zeros. Method: the published value of one term at a time."""
    published = _world()
    assert published["beta"] > 0 and published["chi"] > 0
    for remaining in ("beta", "chi"):
        removed = "chi" if remaining == "beta" else "beta"
        world = _world(**{removed: 0.0})
        first, second = _run(world, oscillator_seed=1), _run(world, oscillator_seed=2)
        assert not np.array_equal(first["A"], second["A"]), remaining


def test_spec_pairs_differ_only_in_the_feedback():
    """Goal: the committed spec is the paired design. Method: rebuild it and compare twins."""
    module = _spec20()
    committed = json.loads(module.OUTPUT_SPEC.read_text())
    assert committed == module.build_spec()
    assert len(committed) == 8
    runs = {run["tag"]: run for call in committed for run in call["runs"]}
    assert len(runs) == 16
    for tag, run in runs.items():
        if "_noosc_" not in tag:
            continue
        twin = runs[tag.replace("_noosc_", "_osc_")]
        assert run["c"]["beta"] == 0.0 and run["c"]["chi"] == 0.0
        assert twin["c"]["beta"] > 0 and twin["c"]["chi"] > 0
        same_soliton = {k: v for k, v in run["c"].items() if k not in module.FEEDBACK_KEYS}
        assert same_soliton == {k: v for k, v in twin["c"].items() if k not in module.FEEDBACK_KEYS}
        assert {k: v for k, v in run.items() if k not in ("c", "tag")} == {
            k: v for k, v in twin.items() if k not in ("c", "tag")
        }
