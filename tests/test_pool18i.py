"""Goal: pool18i.py rebuilds the committed 18g+18i pool. Method: rerun it on the committed JSON."""

import json
import math
from pathlib import Path
from typing import Any

import pool18i
import pytest

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "part18" / "results"
BLOCKS = ("aggregates_18i", "aggregates_pool", "cue_vs_nocue_18i", "cue_vs_nocue_pool")
MEAN_TOLERANCE = 1e-9
# The issue allowed 0.01 for an unknown bootstrap seed. Seed 0 reproduces the committed ends.
INTERVAL_TOLERANCE = 1e-9


def rebuilt_summary() -> dict[str, Any]:
    """Run the pooling on the two committed summaries."""
    records = pool18i.load_pool(
        RESULTS / pool18i.SUMMARY_NAME["18g"], RESULTS / pool18i.SUMMARY_NAME["18i"]
    )
    return pool18i.build_summary(records)


def statistics_by_seed(part: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return {name: statistic} for one arm or difference block, with values keyed by seed."""
    seeds = part["seeds"] if "seeds" in part else [row["seed"] for row in part["per_seed"]]
    found = {}
    for name, statistic in part.items():
        if isinstance(statistic, dict) and "ci95" in statistic:
            found[name] = dict(statistic, values=dict(zip(seeds, statistic["values"])))
    return found


@pytest.mark.parametrize("stored", [False, True], ids=["rebuilt", "pool18i_summary.json"])
def test_pool_matches_committed_summary(stored: bool) -> None:
    """Every mean and interval in full18i_summary.json is reproduced, fresh and as stored."""
    committed = json.loads((RESULTS / pool18i.SUMMARY_NAME["18i"]).read_text())
    summary = json.loads(pool18i.OUTPUT.read_text()) if stored else rebuilt_summary()
    compared_count = 0
    for block in BLOCKS:
        parts = [(committed[block], summary[block])]
        if block.startswith("aggregates"):
            parts = [(committed[block][arm], summary[block][arm]) for arm in pool18i.ARMS]
        for expected_part, actual_part in parts:
            expected, actual = statistics_by_seed(expected_part), statistics_by_seed(actual_part)
            assert sorted(actual) == sorted(expected)
            for name, statistic in expected.items():
                assert actual[name]["values"] == pytest.approx(statistic["values"], abs=1e-12)
                assert actual[name]["mean"] == pytest.approx(statistic["mean"], abs=MEAN_TOLERANCE)
                assert actual[name]["ci95"] == pytest.approx(
                    statistic["ci95"], abs=INTERVAL_TOLERANCE
                )
                compared_count += 1
    assert compared_count == 2 * 3 * 12 + 2 * 3


@pytest.mark.parametrize("stored", [False, True], ids=["rebuilt", "pool18i_summary.json"])
def test_paired_difference_signs_and_p_values(stored: bool) -> None:
    """n_pos is 7/8 and 2/8, and the within_aligned interval excludes 0 while both p > 0.05."""
    summary = json.loads(pool18i.OUTPUT.read_text()) if stored else rebuilt_summary()
    pool = summary["cue_vs_nocue_pool"]
    assert pool["n"] == 8
    signs = {name: (pool[name]["n_pos"], pool[name]["n_neg"]) for name in pool18i.DIFFERENCE_NAMES}
    assert signs == {"d_gAminusgB": (4, 4), "d_poison_B": (2, 6), "d_within_aligned": (7, 1)}
    within = pool["d_within_aligned"]
    assert within["ci95"][0] > 0.0
    assert within["t_test_p"] == pytest.approx(0.05674, abs=1e-5)
    assert within["wilcoxon_p"] == pytest.approx(14 / 256, abs=1e-12)
    assert pool["d_poison_B"]["t_test_p"] == pytest.approx(0.08399, abs=1e-5)
    assert pool["d_poison_B"]["wilcoxon_p"] == pytest.approx(38 / 256, abs=1e-12)
    assert pool["d_gAminusgB"]["wilcoxon_p"] == pytest.approx(1.0, abs=1e-12)


def test_bad_input_gives_clear_error(tmp_path: Path) -> None:
    """A missing file, broken JSON, a null value and a missing run each raise PoolError."""
    valid = json.loads((RESULTS / pool18i.SUMMARY_NAME["18g"]).read_text())
    assert len(pool18i.load_runs(RESULTS / pool18i.SUMMARY_NAME["18g"], "18g")) == 12
    path = tmp_path / "summary.json"
    with pytest.raises(pool18i.PoolError, match="cannot read"):
        pool18i.load_runs(path, "18g")
    path.write_text("{not json")
    with pytest.raises(pool18i.PoolError, match="not valid JSON"):
        pool18i.load_runs(path, "18g")
    for bad_value in (None, math.nan):
        broken = json.loads(json.dumps(valid))
        broken["runs"][0]["mean_within_aligned"] = bad_value
        path.write_text(json.dumps(broken))
        with pytest.raises(pool18i.PoolError, match="bias_cue_s851.*mean_within_aligned"):
            pool18i.load_runs(path, "18g")
    path.write_text(json.dumps({"runs": valid["runs"][1:]}))
    with pytest.raises(pool18i.PoolError, match="expected one each"):
        pool18i.load_runs(path, "18g")
    assert pool18i.main(["--summary-18g", str(path), "--output", str(tmp_path / "out.json")]) == 1
    assert not (tmp_path / "out.json").exists()
    copy = tmp_path / "copy18i.json"
    copy.write_text((RESULTS / pool18i.SUMMARY_NAME["18i"]).read_text())
    before = copy.read_text()
    assert pool18i.main(["--summary-18i", str(copy), "--output", str(copy)]) == 1
    assert copy.read_text() == before
