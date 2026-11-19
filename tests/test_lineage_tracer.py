"""Issue #13: tracer fractions record the true origin of mass and do not touch the simulation."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fl_eco3
import fl_glider
import lineage_tracer as LT

ROOT = Path(__file__).resolve().parents[1]
LATTICE_SIZE = 32
MUTATION_PERIOD = 5
OPTIONS = {"coarse": True, "fnoise": True, "m": MUTATION_PERIOD}
BODY_CENTRES = ((8.0, 8.0), (22.0, 20.0))
BODY_TRAITS = ((0.6, -0.3), (-0.4, 0.5))


def _world():
    calls = json.loads((ROOT / "archive" / "specs" / "eco13m.json").read_text())
    runs = {run["tag"]: run for call in calls for run in call["runs"]}
    run = runs["sel_4D56_s201"]
    return {**run["c"], **run["ecos"][0], "U1": 0.1}


def _two_bodies():
    """A 2D eco3 state with two separate bodies, its kernels, and the label field of the bodies."""
    world = _world()
    rng = np.random.default_rng(0)
    shape = (LATTICE_SIZE, LATTICE_SIZE)
    grid = np.arange(LATTICE_SIZE)
    mass = np.zeros(shape)
    traits = np.zeros((2, *shape))
    labels = np.zeros(shape, int)
    for index, (centre, body_traits) in enumerate(zip(BODY_CENTRES, BODY_TRAITS)):
        squared = (grid[:, None] - centre[0]) ** 2 + (grid[None, :] - centre[1]) ** 2
        body = 0.8 * np.exp(-squared / 8.0)
        mass += body
        traits += np.array(body_traits).reshape(2, 1, 1) * body[None]
        labels[body > 0.05] = index + 1
    base = {
        "A": mass,
        "x": rng.uniform(-6, 6, shape),
        "y": rng.uniform(-6, 6, shape),
        "z": rng.uniform(0, 0.5, shape),
        "c": np.full(shape, 2.6),
    }
    state = fl_eco3.make_state(base, world, K=2, seed=0)
    state["Q"] = traits.astype(np.float32)
    state = {name: jnp.asarray(value) for name, value in state.items()}
    peaks = tuple(world["peaks"])
    kernels = (
        fl_eco3.params(world),
        jnp.asarray(fl_glider.ring_kernel_np(LATTICE_SIZE, 2, float(world["R"]), peaks)),
        jnp.asarray(fl_glider.ring_kernel_np(LATTICE_SIZE, 2, 3.0)),
        fl_eco3.kernels(LATTICE_SIZE, 2, coarse=True),
    )
    return state, kernels, labels


def _plain_run(state, kernels, step_count):
    """The engine alone: ``fl_eco3.run_chunk`` on a batch of one run."""
    parameters, mass_kernel, oscillator_kernel, wide_kernels = kernels
    batch = jax.tree.map(lambda value: value[None], state)
    batch_parameters = jax.tree.map(lambda value: value[None], parameters)
    final = fl_eco3.run_chunk(
        batch,
        batch_parameters,
        mass_kernel[None],
        oscillator_kernel,
        wide_kernels,
        step_count,
        **OPTIONS,
    )
    return jax.tree.map(lambda value: value[0], final)


def test_tracers_do_not_change_the_simulation():
    """Goal: the tracer is passive. Method: the run against the engine alone, bit for bit."""
    state, kernels, labels = _two_bodies()
    step_count = 2 * MUTATION_PERIOD
    fractions = jnp.asarray(LT.fractions_from_labels(labels, 3))
    traced, _ = LT.run_chunk_with_tracers(state, fractions, *kernels, step_count, **OPTIONS)
    plain = _plain_run(state, kernels, step_count)
    assert set(traced) == set(plain)
    assert not np.array_equal(np.asarray(plain["A"]), np.asarray(state["A"]))
    for name in plain:
        assert np.array_equal(np.asarray(traced[name]), np.asarray(plain[name])), name


def test_each_body_keeps_its_own_tracer():
    """Goal: a body is its own ancestor. Method: two separate bodies, 20 steps, then shares."""
    state, kernels, labels = _two_bodies()
    fractions = jnp.asarray(LT.fractions_from_labels(labels, 3))
    state, fractions = LT.run_chunk_with_tracers(state, fractions, *kernels, 20, **OPTIONS)
    mass = np.asarray(state["A"])
    _, new_labels = fl_glider.components(mass, 0.05, min_mass=0.0)
    identifiers = [int(new_labels[int(x), int(y)]) for x, y in BODY_CENTRES]
    assert identifiers[0] != identifiers[1] and min(identifiers) > 0
    relabelled = np.zeros_like(labels)
    for index, identifier in enumerate(identifiers):
        relabelled[new_labels == identifier] = index + 1
    shares, background = LT.object_shares(np.asarray(fractions), mass, relabelled, 2)
    assert shares[0, 0] > 0.999 and shares[1, 1] > 0.999
    assert shares[0, 1] < 0.001 and shares[1, 0] < 0.001
    assert background.max() < 0.05
    assert LT.fraction_sum_error(np.asarray(fractions), mass) < 0.05


def test_object_shares_are_weighted_by_mass():
    """Goal: shares follow mass, not cell counts. Method: a hand-made mixture and a background."""
    mass = np.array([[1.0, 3.0, 2.0, 0.0]])
    labels = np.array([[1, 1, 2, 0]])
    fractions = np.zeros((3, 1, 4))
    fractions[1, 0, 0] = 1.0
    fractions[2, 0, 1] = 1.0
    fractions[0, 0, 2] = 0.5
    fractions[1, 0, 2] = 0.5
    shares, background = LT.object_shares(fractions, mass, labels, 2)
    assert shares[0] == pytest.approx([0.25, 0.75])
    assert shares[1] == pytest.approx([1.0, 0.0])
    assert background == pytest.approx([0.0, 0.5])
    assert LT.fraction_sum_error(fractions, mass) == pytest.approx(0.0)
    fractions[2, 0, 1] = 0.9
    assert LT.fraction_sum_error(fractions, mass) == pytest.approx(0.3 / 6.0)


def test_an_object_without_traced_mass_has_no_shares():
    """Goal: no division by zero. Method: an object that lies fully in the background."""
    mass = np.array([[2.0, 1.0]])
    labels = np.array([[1, 2]])
    fractions = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
    shares, background = LT.object_shares(fractions, mass, labels, 2)
    assert shares[0].tolist() == [0.0]
    assert shares[1].tolist() == [1.0]
    assert background.tolist() == [1.0, 0.0]


def test_labels_must_fit_the_channels():
    """Goal: a label is never dropped in silence. Method: one channel too few."""
    labels = np.array([[0, 1, 2]])
    assert LT.fractions_from_labels(labels, 3).shape == (3, 1, 3)
    with pytest.raises(AssertionError):
        LT.fractions_from_labels(labels, 2)
