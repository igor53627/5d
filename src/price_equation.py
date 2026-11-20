"""Mass-weighted Price decomposition for eco logs.

A log is a list of snapshots ``(time, objects)``. An object is ``(mass, centre, traits)``.
The population mean of a trait is weighted by mass, so the Price terms must use the same
weights. For ancestors ``i`` at one snapshot and descendants ``j`` a fixed number of
snapshots later, let ``W_i`` be the descendant mass credited to ancestor ``i`` and ``M'``
the total descendant mass. Then::

    selection    = sum_i (W_i / M') * p_i  -  mean
    transmission = mean'  -  sum_i (W_i / M') * p_i
    selection + transmission = mean' - mean        (exact)

``selection`` is the mass-weighted covariance of relative fitness and trait. ``transmission``
holds every change inside a lineage: mutation, selection among the cells of one body, exchange
between bodies, and errors of the ancestor rule. It is not a measurement of any single one of these.

Both terms depend on the ancestor of each descendant. Pass ``fingerprint_indices`` so that
``lineage_tracking.assign_ancestors`` identifies ancestors by heritable traits. The
nearest-in-space rule picks a different object for most descendants in dense populations.

``analyze_eco.selection_diff`` is the older estimator. It uses an unweighted covariance, so
the remainder ``delta - S`` that older analysis scripts call "transmission" is not a
transmission term. It stays in the repository so that published numbers can be reproduced.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from lineage_tracking import GATE_RADIUS_DEFAULT, EcoObject, Snapshot, assign_ancestors

IDENTITY_TOLERANCE: float = 1e-9


@dataclass(frozen=True)
class PriceTerms:
    """Price terms for every usable window of one log; all arrays have one entry per window."""

    times: np.ndarray
    selection: np.ndarray
    transmission: np.ndarray
    delta: np.ndarray

    @property
    def window_count(self) -> int:
        """Number of windows that entered the decomposition."""
        return len(self.times)

    def mean(self, term: str) -> float:
        """Mean of one term (``selection``, ``transmission`` or ``delta``) over the windows."""
        values: np.ndarray = getattr(self, term)
        return float(values.mean()) if len(values) else float("nan")


def step_shares(ancestor_of: np.ndarray, ancestor_count: int) -> np.ndarray:
    """One-hot matrix: ``shares[j, i]`` is 1 when descendant ``j`` is credited to ancestor ``i``."""
    shares: np.ndarray = np.zeros((len(ancestor_of), ancestor_count))
    shares[np.arange(len(ancestor_of)), ancestor_of] = 1.0
    return shares


def window_terms(
    ancestors: Sequence[EcoObject],
    descendants: Sequence[EcoObject],
    shares: np.ndarray,
    trait_index: int = 0,
) -> tuple[float, float, float]:
    """Return ``(selection, transmission, delta)`` for one ancestor-descendant window.

    ``shares[j, i]`` is the fraction of descendant ``j``'s mass credited to ancestor ``i``.
    """
    ancestor_mass: np.ndarray = np.array([o[0] for o in ancestors], float)
    ancestor_trait: np.ndarray = np.array([o[2][trait_index] for o in ancestors], float)
    descendant_mass: np.ndarray = np.array([o[0] for o in descendants], float)
    descendant_trait: np.ndarray = np.array([o[2][trait_index] for o in descendants], float)
    if shares.shape != (len(descendants), len(ancestors)):
        raise ValueError(f"shares has shape {shares.shape}, which does not match the objects")
    if ancestor_mass.sum() <= 0 or descendant_mass.sum() <= 0:
        raise ValueError("window_terms needs positive total mass on both sides")
    credited_mass: np.ndarray = descendant_mass @ shares
    mean_before: float = float(ancestor_mass @ ancestor_trait / ancestor_mass.sum())
    mean_after: float = float(descendant_mass @ descendant_trait / descendant_mass.sum())
    ancestral_mean: float = float(credited_mass @ ancestor_trait / credited_mass.sum())
    selection: float = ancestral_mean - mean_before
    transmission: float = mean_after - ancestral_mean
    delta: float = mean_after - mean_before
    if abs(selection + transmission - delta) > IDENTITY_TOLERANCE:
        raise AssertionError("Price identity broken: selection + transmission != delta")
    return selection, transmission, delta


def _chained_shares(
    step_matrices: Sequence[np.ndarray | None], start: int, lag: int
) -> np.ndarray | None:
    """Shares between snapshot ``start`` and ``start + lag``; None if a snapshot is empty."""
    chained: np.ndarray | None = None
    for step in range(start + lag, start, -1):
        matrix: np.ndarray | None = step_matrices[step]
        if matrix is None:
            return None
        chained = matrix if chained is None else chained @ matrix
    return chained


def _credits_no_mass(
    ancestors: Sequence[EcoObject], descendants: Sequence[EcoObject], shares: np.ndarray
) -> bool:
    """True when the shares credit no descendant mass to an ancestor.

    A share matrix of the wrong shape is left to ``window_terms``, which reports it.
    """
    if shares.shape != (len(descendants), len(ancestors)):
        return False
    descendant_mass: np.ndarray = np.array([o[0] for o in descendants], float)
    return float(descendant_mass @ shares.sum(axis=1)) <= 0


def step_share_matrices(
    log: Sequence[Snapshot],
    box_size: float,
    fingerprint_indices: Sequence[int] | None = None,
    gate_radius: float = GATE_RADIUS_DEFAULT,
) -> list[np.ndarray | None]:
    """Share matrix of every snapshot against the one before it, from ``assign_ancestors``.

    Entry ``k`` belongs to the snapshots ``k - 1`` and ``k``. It is None for ``k = 0`` and
    when one of the two snapshots has no objects. An empty log gives an empty list.
    """
    if not log:
        return []
    step_matrices: list[np.ndarray | None] = [None]
    for (_, ancestors), (_, descendants) in zip(log[:-1], log[1:]):
        if not ancestors or not descendants:
            step_matrices.append(None)
            continue
        ancestor_of: np.ndarray = assign_ancestors(
            ancestors, descendants, box_size, fingerprint_indices, gate_radius
        )
        step_matrices.append(step_shares(ancestor_of, len(ancestors)))
    return step_matrices


def price_terms_from_shares(
    log: Sequence[Snapshot],
    step_matrices: Sequence[np.ndarray | None],
    lag: int = 5,
    trait_index: int = 0,
    time_min: float = 0.0,
) -> PriceTerms:
    """Decompose the change of the mass-weighted mean trait, given the shares of every step.

    ``step_matrices`` has the layout of ``step_share_matrices``. The shares can be fractions,
    for example measured tracer content. ``shares[j, i]`` is then the share of the whole mass
    of descendant ``j`` that comes from ancestor ``i``, and a row sums to less than 1 when a
    part of the mass comes from no object. That part is credited to no ancestor, also along a
    chain of steps. A window is skipped when it starts before ``time_min``, when it has fewer
    than two ancestors, when any snapshot inside it has no objects, or when no descendant mass
    is credited to an ancestor.
    """
    if lag < 1:
        raise ValueError(f"lag must be at least 1, got {lag}")
    if len(step_matrices) != len(log):
        raise ValueError(f"{len(step_matrices)} share matrices for {len(log)} snapshots")
    rows: list[tuple[float, float, float, float]] = []
    for start in range(len(log) - lag):
        time, ancestors = log[start]
        if time < time_min or len(ancestors) < 2:
            continue
        descendants: Sequence[EcoObject] = log[start + lag][1]
        shares: np.ndarray | None = _chained_shares(step_matrices, start, lag)
        if shares is None or _credits_no_mass(ancestors, descendants, shares):
            continue
        terms: tuple[float, float, float] = window_terms(
            ancestors, descendants, shares, trait_index
        )
        rows.append((float(time), *terms))
    table: np.ndarray = np.array(rows, float).reshape(-1, 4)
    return PriceTerms(
        times=table[:, 0], selection=table[:, 1], transmission=table[:, 2], delta=table[:, 3]
    )


def price_terms(
    log: Sequence[Snapshot],
    box_size: float,
    lag: int = 5,
    trait_index: int = 0,
    time_min: float = 0.0,
    fingerprint_indices: Sequence[int] | None = None,
    gate_radius: float = GATE_RADIUS_DEFAULT,
) -> PriceTerms:
    """Decompose the change of the mass-weighted mean trait over every window of ``lag``.

    A window is skipped when it starts before ``time_min``, when it has fewer than two
    ancestors, or when any snapshot inside it has no objects. Lineages are followed one
    snapshot at a time with ``assign_ancestors``.

    Each descendant has one ancestor. A fusion therefore credits the whole fused body to one
    partner: it appears as selection for that partner and as an opposite transmission term.
    """
    if lag < 1:
        raise ValueError(f"lag must be at least 1, got {lag}")
    step_matrices = step_share_matrices(log, box_size, fingerprint_indices, gate_radius)
    return price_terms_from_shares(log, step_matrices, lag, trait_index, time_min)
