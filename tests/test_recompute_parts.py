"""Goal: the Part 18 score and phases of recompute_tracking.py follow the season sign.

Method: synthetic logs with a known season; no published data is read.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

import fl_season

ROOT = Path(__file__).resolve().parents[1]
SEASON_PERIOD = 4000.0
SNAPSHOT_STEPS = 200
TRAIT_COUNT = fl_season.KB + fl_season.ND


def _recompute_tracking() -> ModuleType:
    """Load the script as a module; its dataclasses need the entry in ``sys.modules``."""
    name = "recompute_tracking"
    if name not in sys.modules:
        location = ROOT / "experiments" / "tracking_fix" / "recompute_tracking.py"
        module_spec = importlib.util.spec_from_file_location(name, location)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[name] = module
        module_spec.loader.exec_module(module)
    return sys.modules[name]


def _body(gain_a: float, gain_b: float) -> tuple:
    """One object whose realised gains are ``gain_a`` and ``gain_b``; other traits are 0."""
    traits = [0.0] * TRAIT_COUNT
    traits[fl_season.KB + 3] = gain_a
    traits[fl_season.KB + 4] = gain_b
    return (1.0, [0.0, 0.0, 0.0, 0.0], traits)


def test_adaptive_score_uses_the_season_of_the_snapshot_time():
    recompute = _recompute_tracking()
    objects = [_body(0.75, 0.25), _body(-1.0, 0.5)]
    eco = {"Ts": SEASON_PERIOD}
    in_season_a = recompute.trait_adaptive_score(objects, eco, 200)
    in_season_b = recompute.trait_adaptive_score(objects, eco, 2200)
    np.testing.assert_allclose(in_season_a, [0.5, -1.5])
    np.testing.assert_allclose(in_season_b, [-0.5, 1.5])
    default_period = recompute.trait_adaptive_score(objects, {}, 3200)
    np.testing.assert_allclose(default_period, [-0.5, 1.5])


def test_with_trait_appends_the_score_at_each_snapshot_time():
    recompute = _recompute_tracking()
    log = [(time, [_body(0.75, 0.25)]) for time in (1800, 2000, 2200)]
    fingerprint_count = recompute.SEASON_TRAIT_COUNT
    extended = recompute.with_trait(
        log, {"Ts": SEASON_PERIOD}, fingerprint_count, recompute.trait_adaptive_score
    )
    scores = [objects[0][2][fingerprint_count] for _, objects in extended]
    assert scores == [0.5, 0.5, -0.5]
    assert all(len(objects[0][2]) == fingerprint_count + 1 for _, objects in extended)


def test_season_windows_do_not_span_a_switch():
    recompute = _recompute_tracking()
    times = range(0, 8000 + SNAPSHOT_STEPS, SNAPSHOT_STEPS)
    log = [(time, [_body(0.75, 0.25)]) for time in times]
    eco = {"Ts": SEASON_PERIOD}
    run = recompute.Run("18", "mem_s1", "mem", 56, log, eco, True)
    phases = recompute.whole_run_and_seasons(run)
    assert phases["all"] == [(0.0, 8000.0)]
    assert phases["seasons"] == [
        (0.0, 2000.0),
        (2200.0, 4000.0),
        (4200.0, 6000.0),
        (6200.0, 8000.0),
    ]
    for time_low, time_high in phases["seasons"]:
        signs = {
            recompute.season_sign(time, eco) for time in times if time_low <= time <= time_high
        }
        assert len(signs) == 1
