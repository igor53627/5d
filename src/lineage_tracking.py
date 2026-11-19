"""Ancestor assignment between two snapshots of an eco log.

A log is a list of snapshots ``(time, objects)``. An object is ``(mass, centre, traits)``.

``analyze_eco.parents`` takes the nearest object in space as the ancestor. Snapshots are
200 steps apart, and in that time a body moves 12 to 15 cells while its neighbours are about
19 cells away. The nearest object is then often a different body. Heritable traits change
little in 200 steps and differ much more between bodies, so they identify the ancestor.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

EcoObject = tuple  # (mass, centre, traits)
Snapshot = tuple  # (time, list of EcoObject)

GATE_RADIUS_DEFAULT: float = 30.0
DIVISION_MASS_MIN: float = 0.3


def periodic_distances(
    centres_from: np.ndarray, centres_to: np.ndarray, box_size: float
) -> np.ndarray:
    """Distances on a periodic box, shape ``(len(centres_from), len(centres_to))``."""
    offset: np.ndarray = centres_from[:, None, :] - centres_to[None, :, :]
    wrapped: np.ndarray = (offset + box_size / 2) % box_size - box_size / 2
    return np.sqrt((wrapped * wrapped).sum(-1))


def assign_ancestors(
    ancestors: Sequence[EcoObject],
    descendants: Sequence[EcoObject],
    box_size: float,
    fingerprint_indices: Sequence[int] | None = None,
    gate_radius: float = GATE_RADIUS_DEFAULT,
) -> np.ndarray:
    """Index of the ancestor of each descendant, one snapshot apart.

    Without ``fingerprint_indices`` the ancestor is the nearest object in space, which is the
    rule of ``analyze_eco.parents``. With ``fingerprint_indices``, the ancestor is the object
    within ``gate_radius`` whose heritable traits at those indices are closest to the
    descendant's. A descendant with no object inside the gate falls back to the nearest
    object in space.
    """
    if not ancestors or not descendants:
        raise ValueError("assign_ancestors needs at least one ancestor and one descendant")
    ancestor_centres: np.ndarray = np.array([o[1] for o in ancestors], float)
    descendant_centres: np.ndarray = np.array([o[1] for o in descendants], float)
    distances: np.ndarray = periodic_distances(descendant_centres, ancestor_centres, box_size)
    nearest_in_space: np.ndarray = distances.argmin(axis=1)
    if fingerprint_indices is None:
        return nearest_in_space
    indices: list[int] = list(fingerprint_indices)
    ancestor_traits: np.ndarray = np.array([o[2] for o in ancestors], float)[:, indices]
    descendant_traits: np.ndarray = np.array([o[2] for o in descendants], float)[:, indices]
    trait_offset: np.ndarray = descendant_traits[:, None, :] - ancestor_traits[None, :, :]
    trait_distances: np.ndarray = np.sqrt((trait_offset * trait_offset).sum(-1))
    gated: np.ndarray = np.where(distances <= gate_radius, trait_distances, np.inf)
    nearest_in_traits: np.ndarray = gated.argmin(axis=1)
    has_candidate: np.ndarray = np.isfinite(gated.min(axis=1))
    return np.where(has_candidate, nearest_in_traits, nearest_in_space)


def tracking_disagreement(
    log: Sequence[Snapshot],
    box_size: float,
    fingerprint_indices: Sequence[int],
    gate_radius: float = GATE_RADIUS_DEFAULT,
    time_min: float = 0.0,
) -> tuple[float, int]:
    """Fraction of descendants whose nearest object in space is not their fingerprint ancestor.

    Returns ``(fraction, descendant_count)``. The fraction is NaN when no pair of snapshots
    after ``time_min`` has objects on both sides.
    """
    differing: int = 0
    total: int = 0
    for (time, ancestors), (_, descendants) in zip(log[:-1], log[1:]):
        if time < time_min or not ancestors or not descendants:
            continue
        by_space: np.ndarray = assign_ancestors(ancestors, descendants, box_size)
        by_traits: np.ndarray = assign_ancestors(
            ancestors, descendants, box_size, fingerprint_indices, gate_radius
        )
        differing += int((by_space != by_traits).sum())
        total += len(descendants)
    return (differing / total if total else float("nan")), total


def division_pairs(
    log: Sequence[Snapshot],
    box_size: float,
    trait_index: int,
    fingerprint_indices: Sequence[int] | None = None,
    gate_radius: float = GATE_RADIUS_DEFAULT,
) -> tuple[int, list[tuple[float, float]]]:
    """Count divisions and collect ``(parent trait, child trait)`` pairs.

    A division is an ancestor of mass at least ``DIVISION_MASS_MIN`` with two or more
    descendants of at least that mass, as in ``analyze_eco.lineage``.
    """
    division_count: int = 0
    pairs: list[tuple[float, float]] = []
    for (_, ancestors), (_, descendants) in zip(log[:-1], log[1:]):
        if not ancestors or not descendants:
            continue
        ancestor_of: np.ndarray = assign_ancestors(
            ancestors, descendants, box_size, fingerprint_indices, gate_radius
        )
        for ancestor_index, ancestor in enumerate(ancestors):
            children: list[EcoObject] = [
                descendants[j]
                for j in np.flatnonzero(ancestor_of == ancestor_index)
                if descendants[j][0] >= DIVISION_MASS_MIN
            ]
            if len(children) < 2 or ancestor[0] < DIVISION_MASS_MIN:
                continue
            division_count += 1
            pairs += [(float(ancestor[2][trait_index]), float(c[2][trait_index])) for c in children]
    return division_count, pairs


def parent_child_correlation(pairs: Sequence[tuple[float, float]]) -> float:
    """Pearson correlation of parent and child traits; NaN below 3 pairs or without variance."""
    if len(pairs) < 3:
        return float("nan")
    values: np.ndarray = np.array(pairs, float)
    if values[:, 0].std() < 1e-9 or values[:, 1].std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(values[:, 0], values[:, 1])[0, 1])


def founder_mass_shares(
    log: Sequence[Snapshot],
    box_size: float,
    fingerprint_indices: Sequence[int] | None = None,
    gate_radius: float = GATE_RADIUS_DEFAULT,
) -> np.ndarray:
    """Share of the final mass that descends from each object of the first snapshot.

    The array has one entry per founder and sums to 1. It is empty when the log is empty,
    when any snapshot has no objects (the lineages cannot be followed across it), or when
    the final snapshot has no mass.
    """
    if not log or any(not objects for _, objects in log):
        return np.zeros(0)
    founder_of: np.ndarray = np.arange(len(log[0][1]))
    for (_, ancestors), (_, descendants) in zip(log[:-1], log[1:]):
        ancestor_of: np.ndarray = assign_ancestors(
            ancestors, descendants, box_size, fingerprint_indices, gate_radius
        )
        founder_of = founder_of[ancestor_of]
    final_mass: np.ndarray = np.array([o[0] for o in log[-1][1]], float)
    founder_mass: np.ndarray = np.bincount(founder_of, weights=final_mass, minlength=len(log[0][1]))
    if founder_mass.sum() <= 0:
        return np.zeros(0)
    return founder_mass / founder_mass.sum()


def effective_lineage_count(shares: np.ndarray) -> float:
    """Exponential of the Shannon entropy of the founder shares; NaN for no shares."""
    positive: np.ndarray = shares[shares > 0]
    if not len(positive):
        return float("nan")
    return float(np.exp(-(positive * np.log(positive)).sum()))
