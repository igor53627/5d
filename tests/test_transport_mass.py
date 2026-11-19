"""Mass conservation and positivity of the base transport in fl_glider."""

from typing import Callable

import jax.numpy as jnp
import numpy as np
import pytest

import fl_glider

# The engines scale every velocity vector to at most this magnitude, in cells per step.
SPEED_CAP_NUMERATOR = 0.45
MASS_RELATIVE_ERROR_MAX = 1e-5
LATTICE_SIZE_BY_DIMENSION = {2: 16, 3: 10, 4: 8}
SEEDS = (0, 1, 2)
ADVECTION_SCHEMES = (fl_glider._advect, fl_glider._advect_muscl)
KNOWN_DEFECT = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="_advect_muscl is not positivity-preserving for N >= 3 at the speed cap, issue #10",
)
DIMENSIONS_WITH_DEFECT = (
    2,
    pytest.param(3, marks=KNOWN_DEFECT),
    pytest.param(4, marks=KNOWN_DEFECT),
)


def speed_cap(dimension: int) -> float:
    """Return the engine's cap on the velocity magnitude, 0.45/sqrt(N) cells per step."""
    return SPEED_CAP_NUMERATOR / float(np.sqrt(dimension))


def total(field: object) -> float:
    """Sum a float32 field in float64, so the sum adds no rounding error of its own."""
    return float(np.asarray(field, dtype=np.float64).sum())


def random_field(rng: np.random.Generator, dimension: int) -> np.ndarray:
    """Return a non-negative float32 field in which about a third of the cells are empty."""
    shape = (LATTICE_SIZE_BY_DIMENSION[dimension],) * dimension
    occupied = rng.random(shape) > 0.33
    return (rng.random(shape) * occupied).astype(np.float32)


def random_velocities(rng: np.random.Generator, dimension: int) -> list:
    """Return N velocity fields, capped by the same formula as the engine step."""
    shape = (LATTICE_SIZE_BY_DIMENSION[dimension],) * dimension
    raw = [rng.normal(0.0, 0.3, shape).astype(np.float32) for _ in range(dimension)]
    magnitude = np.sqrt(sum(component * component for component in raw)) + 1e-12
    scale = np.minimum(1.0, speed_cap(dimension) / magnitude).astype(np.float32)
    velocities = [component * scale for component in raw]
    magnitude_capped = np.sqrt(sum(component.astype(np.float64) ** 2 for component in velocities))
    assert magnitude_capped.max() <= speed_cap(dimension) * (1.0 + 1e-6)
    # At least a tenth of the cells must sit at the cap, or the test would not reach it.
    assert (magnitude_capped >= speed_cap(dimension) * (1.0 - 1e-6)).mean() > 0.1
    return [jnp.asarray(component) for component in velocities]


def sloped_worst_case(dimension: int) -> tuple:
    """Build the deterministic worst case for MUSCL: a unit slope with outflow on all faces.

    The centre cell holds 1, each lower neighbour 0 and each upper neighbour 2; all other
    cells are empty, so the total mass is 1 + 2N. The centre velocity is diagonal and each
    neighbour's velocity points away along its axis, all with the magnitude of the cap.
    The centre then loses 0.45*(sqrt(N) + 0.5) of its mass: it keeps 0.139 for N=2 and
    ends at -0.0044 for N=3 and at -0.125 for N=4.
    """
    lattice_size, centre_index = 6, 2
    shape = (lattice_size,) * dimension
    centre = (centre_index,) * dimension
    cap = speed_cap(dimension)
    field = np.zeros(shape, np.float32)
    field[centre] = 1.0
    velocities = [np.zeros(shape, np.float32) for _ in range(dimension)]
    for axis in range(dimension):
        velocities[axis][centre] = cap / np.sqrt(dimension)
        for direction in (-1, 1):
            neighbour = list(centre)
            neighbour[axis] += direction
            velocities[axis][tuple(neighbour)] = direction * cap
            field[tuple(neighbour)] = 1.0 + direction
    return field, [jnp.asarray(component) for component in velocities], centre


@pytest.mark.parametrize("scheme", ADVECTION_SCHEMES)
@pytest.mark.parametrize("dimension", (2, 3, 4))
def test_advection_conserves_mass(dimension: int, scheme: Callable) -> None:
    """Goal: both schemes conserve total mass. Method: random fields at the speed cap."""
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        field = random_field(rng, dimension)
        velocities = random_velocities(rng, dimension)
        advected = scheme(jnp.asarray(field), velocities, 1.0)
        assert advected.dtype == jnp.float32
        error = abs(total(advected) - total(field))
        assert error <= MASS_RELATIVE_ERROR_MAX * total(field)


@pytest.mark.parametrize("dimension", (2, 3, 4))
def test_heritable_field_sum_is_conserved(dimension: int) -> None:
    """Goal: transport conserves sum(Q) for Q = p*A. Method: advect Q like the engines do.

    Q is signed, so the error is measured against sum(|Q|), not against sum(Q).
    """
    for seed in SEEDS:
        rng = np.random.default_rng(100 + seed)
        field = random_field(rng, dimension)
        velocities = random_velocities(rng, dimension)
        parameter = rng.uniform(-1.0, 1.0, field.shape).astype(np.float32)
        heritable = parameter * field
        advected = fl_glider._advect_muscl(jnp.asarray(heritable), velocities, 1.0)
        error = abs(total(advected) - total(heritable))
        assert error <= MASS_RELATIVE_ERROR_MAX * total(np.abs(heritable))


def test_muscl_keeps_random_fields_positive_in_two_dimensions() -> None:
    """Goal: MUSCL is positivity-preserving for N=2. Method: random fields at the speed cap.

    With minmod a face value is at most 1.5 times the cell value, so a cell loses at most
    0.45*(sqrt(2) + 0.5) = 0.86 of its mass per step.
    """
    for seed in SEEDS:
        rng = np.random.default_rng(200 + seed)
        field = random_field(rng, 2)
        velocities = random_velocities(rng, 2)
        advected = fl_glider._advect_muscl(jnp.asarray(field), velocities, 1.0)
        assert float(advected.min()) >= 0.0


@pytest.mark.parametrize("dimension", (2, 3, 4))
def test_sloped_worst_case_centre_matches_the_minmod_bound(dimension: int) -> None:
    """Goal: pin the size of the outflow. Method: compare the centre with the closed form.

    The centre keeps 1 - 0.45*(sqrt(N) + 0.5): 0.139 for N=2, -0.0044 for N=3, -0.125 for N=4.
    """
    field, velocities, centre = sloped_worst_case(dimension)
    advected = np.asarray(fl_glider._advect_muscl(jnp.asarray(field), velocities, 1.0))
    expected_centre = 1.0 - SPEED_CAP_NUMERATOR * (np.sqrt(dimension) + 0.5)
    assert advected[centre] == pytest.approx(expected_centre, abs=1e-5)


@pytest.mark.parametrize("dimension", DIMENSIONS_WITH_DEFECT)
def test_muscl_keeps_sloped_worst_case_positive(dimension: int) -> None:
    """Goal: no cell goes negative. Method: the sloped worst case at the speed cap.

    Passes for N=2. Known defect for N=3 and N=4: the strict xfail turns into a failure
    when the scheme becomes positivity-preserving, so that the marker is removed.
    """
    field, velocities, _ = sloped_worst_case(dimension)
    advected = np.asarray(fl_glider._advect_muscl(jnp.asarray(field), velocities, 1.0))
    assert advected.min() >= 0.0


@pytest.mark.parametrize("dimension", DIMENSIONS_WITH_DEFECT)
def test_clamp_after_muscl_creates_no_mass(dimension: int) -> None:
    """Goal: the engines' clamp keeps mass. Method: clamp the sloped worst case at zero.

    The engine step applies jnp.maximum(An, 0.0) to the MUSCL result. Where MUSCL went
    negative, the clamp adds that mass back from nothing: +0.0044 for N=3, +0.125 for N=4.
    """
    field, velocities, _ = sloped_worst_case(dimension)
    advected = fl_glider._advect_muscl(jnp.asarray(field), velocities, 1.0)
    assert abs(total(advected) - total(field)) <= MASS_RELATIVE_ERROR_MAX * total(field)
    clamped = jnp.maximum(advected, 0.0)
    assert abs(total(clamped) - total(field)) <= MASS_RELATIVE_ERROR_MAX * total(field)
