"""Lineage tracking and the mass-weighted Price decomposition, on logs with a known answer."""

import numpy as np
import pytest

import lineage_tracking as LT
import price_equation as PE

BOX_SIZE = 64.0


def _body(mass, x, traits):
    return (float(mass), [float(x), 10.0], [float(v) for v in traits])


def _moving_pair_log():
    """Two bodies move 12 cells per snapshot and start 10 cells apart; only body A grows.

    Traits are (fingerprint, analysed trait). After one snapshot body A sits 2 cells from
    where body B was, so the nearest-in-space rule takes B as the ancestor of both.
    """
    snapshots = []
    for index, mass_a in enumerate((1.0, 2.0, 4.0)):
        shift = 12.0 * index
        snapshots.append(
            (
                200 * index,
                [_body(mass_a, 10 + shift, (0.5, 1.0)), _body(1.0, 20 + shift, (-0.5, 0.0))],
            )
        )
    return snapshots


def test_fingerprint_finds_the_ancestor_that_nearest_in_space_misses():
    """Goal: the tracker follows a body that moves past its neighbour. Method: two movers."""
    (_, ancestors), (_, descendants) = _moving_pair_log()[:2]
    by_space = LT.assign_ancestors(ancestors, descendants, BOX_SIZE)
    by_traits = LT.assign_ancestors(ancestors, descendants, BOX_SIZE, fingerprint_indices=[0])
    assert by_space.tolist() == [1, 1]
    assert by_traits.tolist() == [0, 1]
    fraction, count = LT.tracking_disagreement(_moving_pair_log(), BOX_SIZE, [0])
    assert count == 4
    assert fraction == pytest.approx(0.5)


def test_gate_excludes_far_objects_and_falls_back_to_space():
    """Goal: both sides of the gate. Method: the trait twin sits inside, then outside it."""
    ancestors = [_body(1, 10, (0.5,)), _body(1, 30, (-0.5,))]
    descendants = [_body(1, 28, (0.5,))]
    inside = LT.assign_ancestors(ancestors, descendants, BOX_SIZE, [0], gate_radius=18.0)
    outside = LT.assign_ancestors(ancestors, descendants, BOX_SIZE, [0], gate_radius=17.9)
    nothing_in_gate = LT.assign_ancestors(ancestors, descendants, BOX_SIZE, [0], gate_radius=1.0)
    assert inside.tolist() == [0]
    assert outside.tolist() == [1]
    assert nothing_in_gate.tolist() == [1]


def test_selection_sign_depends_on_tracking():
    """Goal: show the published artefact. Method: only the high-trait body grows.

    Mean before: (1*1 + 1*0)/2 = 0.5. Credited masses with correct ancestors are (2, 1), so
    the ancestral mean is 2/3 and selection is +1/6. Nearest-in-space credits all mass to
    the low-trait body: selection is -1/2 and the rest appears as transmission.
    """
    log = _moving_pair_log()[:2]
    tracked = PE.price_terms(log, BOX_SIZE, lag=1, trait_index=1, fingerprint_indices=[0])
    nearest = PE.price_terms(log, BOX_SIZE, lag=1, trait_index=1)
    assert tracked.window_count == 1
    assert tracked.selection[0] == pytest.approx(1 / 6)
    assert tracked.transmission[0] == pytest.approx(0.0, abs=1e-12)
    assert nearest.selection[0] == pytest.approx(-0.5)
    assert nearest.transmission[0] == pytest.approx(0.5 + 1 / 6)
    assert tracked.delta[0] == pytest.approx(nearest.delta[0])


def test_selection_is_mass_weighted():
    """Goal: weights follow mass. Method: static bodies of mass 1, 1, 10 with exact inheritance.

    Mean: (0.9 - 1 + 10)/12 = 0.825. Growth 2, 0.6, 0.5 gives credited masses 2, 0.6, 5 and
    an ancestral mean of 6.2/7.6. An unweighted covariance gives +0.248 for the same data.
    """
    masses, traits, growth = (1.0, 1.0, 10.0), (0.9, -1.0, 1.0), (2.0, 0.6, 0.5)
    before = [_body(m, 10 + 20 * i, (p,)) for i, (m, p) in enumerate(zip(masses, traits))]
    after = [
        _body(m * g, 10 + 20 * i, (p,)) for i, (m, p, g) in enumerate(zip(masses, traits, growth))
    ]
    terms = PE.price_terms([(0, before), (200, after)], BOX_SIZE, lag=1)
    assert terms.selection[0] == pytest.approx(6.2 / 7.6 - 0.825)
    assert terms.transmission[0] == pytest.approx(0.0, abs=1e-12)


def test_change_inside_lineages_is_transmission():
    """Goal: no false selection. Method: masses stay fixed and every trait rises by 0.1."""
    before = [_body(1 + i, 10 + 20 * i, (0.2 * i,)) for i in range(3)]
    after = [_body(1 + i, 10 + 20 * i, (0.2 * i + 0.1,)) for i in range(3)]
    terms = PE.price_terms([(0, before), (200, after)], BOX_SIZE, lag=1)
    assert terms.selection[0] == pytest.approx(0.0, abs=1e-12)
    assert terms.transmission[0] == pytest.approx(0.1)


def test_terms_sum_to_the_change_of_the_mean_on_random_logs():
    """Goal: the identity holds for any tracking. Method: random bodies, mean computed apart."""
    rng = np.random.default_rng(7)
    log = []
    for index in range(8):
        count = int(rng.integers(2, 9))
        bodies = [
            (
                float(rng.uniform(0.2, 3)),
                rng.uniform(0, BOX_SIZE, 2).tolist(),
                rng.normal(size=3).tolist(),
            )
            for _ in range(count)
        ]
        log.append((200 * index, bodies))
    for fingerprint in (None, [0, 1]):
        terms = PE.price_terms(log, BOX_SIZE, lag=3, trait_index=2, fingerprint_indices=fingerprint)
        assert terms.window_count == 5
        for window, time in enumerate(terms.times):
            start = int(time // 200)
            means = []
            for _, bodies in (log[start], log[start + 3]):
                mass = np.array([b[0] for b in bodies])
                trait = np.array([b[2][2] for b in bodies])
                means.append(float((mass * trait).sum() / mass.sum()))
            total = terms.selection[window] + terms.transmission[window]
            assert total == pytest.approx(means[1] - means[0], abs=1e-12)


def test_windows_with_an_empty_snapshot_or_one_ancestor_are_skipped():
    """Goal: the limits of a usable window. Method: one body, then no body, then two bodies."""
    pair = [_body(1, 10, (0.0,)), _body(1, 40, (1.0,))]
    log = [(0, pair[:1]), (200, pair), (400, []), (600, pair), (800, pair)]
    terms = PE.price_terms(log, BOX_SIZE, lag=1)
    assert terms.times.tolist() == [600.0]
    assert np.isnan(PE.price_terms(log[:2], BOX_SIZE, lag=1).mean("selection"))


def test_invalid_input_is_refused():
    """Goal: errors are reported at the boundary. Method: one bad argument per call."""
    pair = [_body(1, 10, (0.0,)), _body(1, 40, (1.0,))]
    with pytest.raises(ValueError, match="lag"):
        PE.price_terms([(0, pair), (200, pair)], BOX_SIZE, lag=0)
    with pytest.raises(ValueError, match="at least one"):
        LT.assign_ancestors([], pair, BOX_SIZE)
    with pytest.raises(ValueError, match="shape"):
        PE.window_terms(pair, pair, np.ones((3, 2)))
    with pytest.raises(ValueError, match="positive"):
        PE.window_terms([_body(0, 10, (0.0,))], pair, np.ones((2, 1)))


def test_divisions_are_counted_once_and_not_invented_by_bad_tracking():
    """Goal: a real split counts, a passing neighbour does not. Method: both cases."""
    parent = [_body(2, 10, (0.5, 0.5)), _body(1, 40, (-0.5, -0.5))]
    children = [_body(1, 8, (0.5, 0.4)), _body(1, 12, (0.5, 0.6)), _body(1, 40, (-0.5, -0.5))]
    count, pairs = LT.division_pairs([(0, parent), (200, children)], BOX_SIZE, 1, [0])
    assert count == 1
    assert sorted(pairs) == [(0.5, 0.4), (0.5, 0.6)]
    moving = _moving_pair_log()[:2]
    assert LT.division_pairs(moving, BOX_SIZE, 1)[0] == 1
    assert LT.division_pairs(moving, BOX_SIZE, 1, [0])[0] == 0
    assert np.isnan(LT.parent_child_correlation(pairs))


def test_founder_shares_follow_mass_and_drop_dead_lineages():
    """Goal: lineage diversity is weighted by mass. Method: three founders, one dies.

    Final masses 3 and 1 give shares 0.75 and 0.25; exp(entropy) is 4 / 3**0.75.
    """
    founders = [_body(1, 10, (0.5,)), _body(1, 30, (-0.5,)), _body(1, 50, (0.0,))]
    survivors = [_body(3, 10, (0.5,)), _body(1, 30, (-0.5,))]
    shares = LT.founder_mass_shares([(0, founders), (200, survivors)], BOX_SIZE, [0])
    assert shares.tolist() == [0.75, 0.25, 0.0]
    assert LT.effective_lineage_count(shares) == pytest.approx(4 / 3**0.75)
    assert LT.founder_mass_shares([(0, founders), (200, [])], BOX_SIZE).size == 0
    massless = [_body(0, 10, (0.5,))]
    assert LT.founder_mass_shares([(0, founders), (200, massless)], BOX_SIZE).size == 0
    assert np.isnan(LT.effective_lineage_count(np.zeros(0)))
