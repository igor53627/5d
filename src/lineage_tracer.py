"""Passive tracers that record where the mass of every body comes from in an ``fl_eco3`` run.

The eco logs hold no true ancestor: ``lineage_tracking.assign_ancestors`` infers one from
positions or from heritable traits. This module measures it. At a snapshot every object gets
its own tracer channel, a fraction field that is 1 inside the object and 0 elsewhere; channel 0
holds the mass outside every object. The fractions move with the mass flux of the engine and
do nothing else: they are not mutated, not homogenized, and no field of the simulation reads
them. At the next snapshot the tracer content of an object is the share of its mass that
comes from each earlier object.

The engine is not changed. Every step runs ``fl_eco3.step`` twice from the same state: once as
the simulation, and once without mutation, homogenization and decay on a copy whose trait
field has the tracers appended. The engine moves every channel of the trait field with the same
velocity, which it computes from the mass field and the first two traits, so the copy gives
the transported fractions. The simulated state never sees a tracer.

The copy has no decay because the engine clears the traits of a cell whose mass falls to zero
in the reaction. Food uptake depends on the first trait, which homogenization changes, so the
reaction of the copy can differ from the reaction of the simulation. Without decay the mass of
the copy is never below the transported mass, and a cell keeps its fractions whatever the
reaction does.

Mass that grows from food inherits the tracer content of its cell, so growth is credited to
the lineage that grows. The engine limits the flux of each channel on its own, so the
fractions of one cell do not sum to exactly 1; ``fraction_sum_error`` reports the difference.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import fl_eco3
from helpers import invariant

MASS_EPSILON: float = 1e-6
BACKGROUND_CHANNEL: int = 0
DECAY_PARAMETER: str = "dec"

State = dict


def fractions_from_labels(labels: np.ndarray, channel_count: int) -> np.ndarray:
    """One-hot fractions of shape ``(channel_count, *labels.shape)``.

    ``labels`` holds 0 for cells outside every object and ``i + 1`` for cells of object ``i``.
    Channel 0 is the background, channel ``i + 1`` is object ``i``. ``channel_count`` can
    exceed the number of objects; the extra channels are empty.
    """
    invariant(labels.min() >= 0, "labels must not be negative")
    invariant(labels.max() < channel_count, "channel_count is too small for the labels")
    channels: np.ndarray = np.arange(channel_count).reshape((-1,) + (1,) * labels.ndim)
    return (labels[None] == channels).astype(np.float32)


def _carried_fractions(
    state: State, fractions: jnp.ndarray, engine_step: Callable[[State], State]
) -> jnp.ndarray:
    """Fractions after one step of transport from ``state``."""
    trait_count: int = state["Q"].shape[0]
    tracer_mass: jnp.ndarray = fractions * state["A"][None]
    copy: State = {**state, "Q": jnp.concatenate([state["Q"], tracer_mass])}
    moved: State = engine_step(copy)
    mass: jnp.ndarray = moved["A"][None]
    carried: jnp.ndarray = moved["Q"][trait_count:] / jnp.maximum(mass, MASS_EPSILON)
    return jnp.where(mass > MASS_EPSILON, jnp.clip(carried, 0.0, 1.0), 0.0)


@partial(jax.jit, static_argnames=("step_count", "coarse", "fnoise", "m"))
def run_chunk_with_tracers(
    state: State,
    fractions: jnp.ndarray,
    params: dict,
    mass_kernel: jnp.ndarray,
    oscillator_kernel: jnp.ndarray,
    wide_kernels: dict,
    step_count: int,
    coarse: bool = False,
    fnoise: bool = False,
    m: int = 1,
) -> tuple[State, jnp.ndarray]:
    """Advance one unbatched ``fl_eco3`` state and its tracer fractions by ``step_count`` steps.

    The state follows the schedule of ``fl_eco3.run_chunk``: mutation and homogenization act
    on every ``m``-th step. ``step_count`` must be a multiple of ``m``.
    """
    invariant(step_count % m == 0, "step_count must be a multiple of m")
    options: dict = {"transport": True, "coarse": coarse, "fnoise": fnoise, "m": m}

    def engine_step(current: State, step_params: dict, with_mutation: bool) -> State:
        return fl_eco3.step(
            current,
            step_params,
            mass_kernel,
            oscillator_kernel,
            wide_kernels,
            do_mut=with_mutation,
            do_hom=with_mutation,
            **options,
        )

    transport_only = partial(
        engine_step, step_params={**params, DECAY_PARAMETER: 0.0}, with_mutation=False
    )

    def advance(carry: tuple[State, jnp.ndarray], with_mutation: bool) -> tuple[State, jnp.ndarray]:
        current, current_fractions = carry
        carried: jnp.ndarray = _carried_fractions(current, current_fractions, transport_only)
        return engine_step(current, params, with_mutation), carried

    def block(_: int, carry: tuple[State, jnp.ndarray]) -> tuple[State, jnp.ndarray]:
        carry = jax.lax.fori_loop(0, m - 1, lambda _, inner: advance(inner, False), carry)
        return advance(carry, True)

    return jax.lax.fori_loop(0, step_count // m, block, (state, fractions))


def object_shares(
    fractions: np.ndarray, mass: np.ndarray, labels: np.ndarray, object_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Tracer content of the objects of one snapshot.

    Returns ``(shares, background)``. ``shares[j, i]`` is the share of the traced mass of
    object ``j`` that comes from earlier object ``i``; a row sums to 1, or to 0 when the
    object holds no traced mass. ``background[j]`` is the share of the mass of object ``j``
    that was outside every earlier object. ``labels`` uses the convention of
    ``fractions_from_labels`` for the objects of the current snapshot.
    """
    content: np.ndarray = np.zeros((object_count, fractions.shape[0]))
    for channel in range(fractions.shape[0]):
        weights: np.ndarray = (fractions[channel] * mass).ravel()
        totals: np.ndarray = np.bincount(labels.ravel(), weights, minlength=object_count + 1)
        content[:, channel] = totals[1 : object_count + 1]
    total: np.ndarray = content.sum(axis=1)
    traced: np.ndarray = total - content[:, BACKGROUND_CHANNEL]
    background: np.ndarray = np.divide(
        content[:, BACKGROUND_CHANNEL], total, out=np.zeros(object_count), where=total > 0
    )
    from_objects: np.ndarray = content[:, BACKGROUND_CHANNEL + 1 :]
    shares: np.ndarray = np.divide(
        from_objects, traced[:, None], out=np.zeros_like(from_objects), where=traced[:, None] > 0
    )
    return shares, background


def fraction_sum_error(fractions: np.ndarray, mass: np.ndarray) -> float:
    """Mass-weighted mean of ``|sum of fractions - 1|`` over the cells that hold mass."""
    occupied: np.ndarray = mass > MASS_EPSILON
    if not occupied.any():
        return 0.0
    deviation: np.ndarray = np.abs(fractions.sum(axis=0) - 1.0)
    return float((deviation * mass)[occupied].sum() / mass[occupied].sum())
