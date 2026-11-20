"""Goal: analyze_truth.py scores the ancestor rules against measured shares correctly.

Method: hand-made share matrices with a known answer; no simulation is run.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _analyze_truth() -> ModuleType:
    """Load the script as a module; it imports ``_path`` from its own directory."""
    name = "analyze_truth"
    if name not in sys.modules:
        directory = ROOT / "experiments" / "tracking_truth"
        sys.path.insert(0, str(directory))
        try:
            module_spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[name] = module
            module_spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(directory))
    return sys.modules[name]


def _matrices() -> dict:
    """Three descendants of two ancestors; the truth gives 0, 1, 1 as the main ancestors.

    The nearest rule is wrong for descendant 0, the trait rule is wrong for descendant 2, and
    the two rules agree (and are right) for descendant 1.
    """
    truth = np.array([[0.95, 0.0], [0.1, 0.8], [0.3, 0.6]])
    nearest = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    tracked = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    return {
        "truth": [None, truth],
        "nearest": [None, nearest],
        "tracked": [None, tracked],
        "check": [None, tracked],
    }


def _body(mass: float) -> tuple:
    """One object of the given mass; position and traits do not matter here."""
    return (mass, [0.0, 0.0], [0.0, 0.0])


def test_rules_are_scored_against_the_main_ancestor():
    """Goal: right means the object with the largest share. Method: one wrong pick per rule."""
    counts = _analyze_truth().agreement_counts(_matrices())
    assert counts["traced"] == 3
    assert counts["pure"] == 1
    assert counts["nearest_right"] == 2 and counts["tracked_right"] == 2
    assert counts["nearest_right_pure"] == 0 and counts["tracked_right_pure"] == 1


def test_disagreements_are_split_by_which_rule_is_right():
    """Goal: the split of the cases where the rules differ. Method: the same three cases."""
    counts = _analyze_truth().disagreement_counts(_matrices())
    assert counts == {
        "traced": 3,
        "differ": 2,
        "tracked_right": 1,
        "nearest_right": 1,
        "neither": 0,
    }


def test_divisions_and_lineages_follow_the_shares():
    """Goal: lineage numbers from share matrices. Method: one split, then mass 3 against 1."""
    module = _analyze_truth()
    log = [(0, [_body(2.0), _body(1.0)]), (200, [_body(1.5), _body(1.5), _body(1.0)])]
    shares = [None, np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])]
    assert module.division_count(log, shares) == 1
    expected = float(np.exp(-(0.75 * np.log(0.75) + 0.25 * np.log(0.25))))
    assert abs(module.effective_lineages(log, shares) - expected) < 1e-12
    assert module.effective_lineages(log, [None, None]) is None
    light = [(0, [_body(2.0), _body(1.0)]), (200, [_body(1.5), _body(0.2), _body(1.0)])]
    assert module.division_count(light, shares) == 0


def test_a_run_that_goes_extinct_still_gives_a_summary():
    """Goal: the summary of an extinct run is valid JSON. Method: no body after the start."""
    module = _analyze_truth()
    record = {"shares": [], "background": [], "fraction_sum_error": 0.0}
    start = [(1.0, [0.0, 0.0], [0.1, 0.2]), (1.0, [9.0, 9.0], [0.3, 0.4])]
    run = {
        "tag": "neu_test",
        "dimension": 2,
        "box_size": 32,
        "seed": 1,
        "steps": 400,
        "log": [(0, start), (200, []), (400, [])],
        "truth": [record, record],
        "truth_matrices": [None, None, None],
    }
    summary = module.run_summary(run)
    assert summary["background_mean"] is None
    assert summary["truth"]["effective_lineages"] is None
    json.dumps(summary, allow_nan=False)
    assert "n/a" in module.markdown([summary])
