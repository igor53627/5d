"""Part 18i pool: rebuild the pooled 18g+18i statistics from the committed run records.

Reads the `runs` lists of full18g_summary.json (seeds 851-854) and full18i_summary.json
(seeds 855-858) and writes pool18i_summary.json. It does not rewrite either input.

For each arm and statistic: the seed-level mean, the sample standard deviation and a
percentile bootstrap 95% interval for the mean (10,000 resamples of the seeds, with
replacement). Each statistic uses a fresh `numpy.random.default_rng(0)`. With the values in
`runs` order (18g, then 18i) this reproduces the intervals stored in full18i_summary.json.
For each paired cue - nocue difference the script adds n_pos, n_neg, a two-sided one-sample
t-test and an exact two-sided Wilcoxon signed-rank test, both against 0.

With --raw-directory the per-run within_aligned values are recomputed from the raw run_eco
pickles (Hugging Face igor53627/5d, part18/eco18g and part18/eco18i). The script stops with
an error when they differ from the committed per-run values.

    python experiments/part18/pool18i.py
    python experiments/part18/pool18i.py --raw-directory /path/to/5d-data/part18
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

# The import of _path puts src/ and experiments/part*/ on sys.path, so it must come first.
# isort: off
import _path  # noqa: F401
import analyze18c
import fl_season
import metrics18c
import run18g
import run18i
# isort: on

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "part18" / "results"
OUTPUT = RESULTS / "pool18i_summary.json"
SUMMARY_NAME = {"18g": "full18g_summary.json", "18i": "full18i_summary.json"}
RAW_PATTERN = {"18g": "eco18g/b18g_*.pkl", "18i": "eco18i/b18i_*.pkl"}
SEED_OF_FOUNDER = {"18g": dict(run18g.SEED_OF_FOUNDER), "18i": dict(run18i.SEED_OF_FOUNDER)}
ARMS = tuple(run18g.ARMS)
ARM_CUE = "bias_cue"
ARM_NOCUE = "bias_nocue"

BOOTSTRAP_SEED = 0
BOOTSTRAP_RESAMPLE_COUNT = 10_000
INTERVAL_PERCENTILES = (2.5, 97.5)
GRID_SIDE = 48
RAW_TOLERANCE = 1e-9

RUN_FIELD_OF_STATISTIC = {
    "within_aligned": "mean_within_aligned",
    "replacement_aligned": "mean_replacement_aligned",
    "matched": "matched_mean",
    "poison_B": "poison_B",
    "mass": "mass_end",
    "gA": "mean_g_A",
    "gB": "mean_g_B",
}
DIRECTIONS = ("AtoB", "BtoA")
DIRECTION_FIELDS = ("within_aligned", "replacement_aligned")
DIFFERENCE_NAMES = ("d_gAminusgB", "d_poison_B", "d_within_aligned")
RAW_LOAD_ERRORS = (
    OSError,
    EOFError,
    pickle.UnpicklingError,
    AttributeError,
    ImportError,
    IndexError,
    KeyError,
    TypeError,
)


class PoolError(Exception):
    """An input file is missing, malformed or inconsistent with the committed values."""


@dataclass(frozen=True)
class RunRecord:
    """The per-run values that the pooled analysis needs, all finite."""

    tag: str
    arm: str
    seed: int
    source: str
    crossed: bool
    statistics: dict[str, float]


def main(argv: list[str] | None = None) -> int:
    """Rebuild the pool, optionally check it against the raw logs, and write the summary."""
    arguments = parse_arguments(argv)
    try:
        refuse_output_over_input(arguments)
        records = load_pool(arguments.summary_18g, arguments.summary_18i)
        summary = build_summary(records)
        if arguments.raw_directory is None:
            raw_check = None
        else:
            raw_check = check_raw(records, arguments.raw_directory)
        write_summary(summary, arguments.output)
    except PoolError as error:
        print(f"pool18i: error: {error}", file=sys.stderr)
        return 1
    print_report(summary, raw_check)
    print("wrote", arguments.output)
    return 0


def refuse_output_over_input(arguments: argparse.Namespace) -> None:
    """Raise PoolError when the output path names one of the two input summaries."""
    output = Path(arguments.output).resolve()
    for summary in (arguments.summary_18g, arguments.summary_18i):
        if output == Path(summary).resolve():
            raise PoolError(f"--output {output} is an input summary; choose another path")


def parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line; every path has a default inside the repository."""
    parser = argparse.ArgumentParser(description="Rebuild the pooled 18g+18i statistics.")
    parser.add_argument("--summary-18g", type=Path, default=RESULTS / SUMMARY_NAME["18g"])
    parser.add_argument("--summary-18i", type=Path, default=RESULTS / SUMMARY_NAME["18i"])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--raw-directory",
        type=Path,
        default=None,
        help="directory that holds eco18g/ and eco18i/ with the raw pickles (optional check)",
    )
    return parser.parse_args(argv)


def load_pool(summary_18g: Path, summary_18i: Path) -> list[RunRecord]:
    """Return the 18g records followed by the 18i records, each in `runs` order."""
    return load_runs(summary_18g, "18g") + load_runs(summary_18i, "18i")


def load_runs(path: Path, source: str) -> list[RunRecord]:
    """Read one committed summary and return one validated record for each arm and seed."""
    try:
        document = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError) as error:
        raise PoolError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise PoolError(f"{path} is not valid JSON: {error}") from error
    runs = document.get("runs") if isinstance(document, dict) else None
    if not isinstance(runs, list):
        raise PoolError(f"{path}: no 'runs' list")
    records = [
        parse_run(run, source, f"{path.name} runs[{index}]") for index, run in enumerate(runs)
    ]
    found = sorted((record.arm, record.seed) for record in records)
    expected = sorted((arm, seed) for arm in ARMS for seed in SEED_OF_FOUNDER[source])
    if found != expected:
        raise PoolError(f"{path}: runs are {found}, expected one each of {expected}")
    return records


def parse_run(run: Any, source: str, where: str) -> RunRecord:
    """Extract the pooled statistics of one run; raise PoolError on a missing or NaN value."""
    if not isinstance(run, dict):
        raise PoolError(f"{where}: a run must be an object")
    for field in ("tag", "arm", "seed", "crossed", "by_direction"):
        if field not in run:
            raise PoolError(f"{where}: field '{field}' is missing")
    if not isinstance(run["crossed"], bool):
        raise PoolError(f"{where}: field 'crossed' is not a boolean: {run['crossed']!r}")
    if not isinstance(run["seed"], int):
        raise PoolError(f"{where}: field 'seed' is not an integer: {run['seed']!r}")
    where = f"{where} ({run['tag']})"
    statistics = {
        statistic: read_number(run, field, where)
        for statistic, field in RUN_FIELD_OF_STATISTIC.items()
    }
    statistics["crossed"] = 1.0 if run["crossed"] else 0.0
    for direction in DIRECTIONS:
        if not isinstance(run["by_direction"], dict) or direction not in run["by_direction"]:
            raise PoolError(f"{where}: by_direction has no '{direction}'")
        for field in DIRECTION_FIELDS:
            statistics[f"{field}_{direction}"] = read_number(
                run["by_direction"][direction], field, f"{where} by_direction.{direction}"
            )
    return RunRecord(
        tag=str(run["tag"]),
        arm=str(run["arm"]),
        seed=run["seed"],
        source=source,
        crossed=run["crossed"],
        statistics=statistics,
    )


def read_number(container: Any, field: str, where: str) -> float:
    """Return container[field] as a finite float; raise PoolError that names the place."""
    if not isinstance(container, dict) or field not in container:
        raise PoolError(f"{where}: field '{field}' is missing")
    value = container[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PoolError(f"{where}: field '{field}' is not a number: {value!r}")
    if not math.isfinite(value):
        raise PoolError(f"{where}: field '{field}' is not finite: {value!r}")
    return float(value)


def build_summary(records: list[RunRecord]) -> dict[str, Any]:
    """Return the JSON-ready summary: the 18i-only block and the pooled 18g+18i block."""
    records_18i = [record for record in records if record.source == "18i"]
    seed_of_founder = {**SEED_OF_FOUNDER["18g"], **SEED_OF_FOUNDER["18i"]}
    return {
        "part": "18i",
        "inputs": [SUMMARY_NAME["18g"], SUMMARY_NAME["18i"]],
        "pool_seeds": sorted(seed_of_founder),
        "seed_of_founder": {str(seed): seed_of_founder[seed] for seed in sorted(seed_of_founder)},
        "bootstrap": {
            "method": "percentile interval for the mean, seeds resampled with replacement",
            "generator": "numpy.random.default_rng(seed), fresh for each statistic",
            "seed": BOOTSTRAP_SEED,
            "resample_count": BOOTSTRAP_RESAMPLE_COUNT,
            "percentiles": list(INTERVAL_PERCENTILES),
        },
        "tests": {
            "t_test_p": "scipy.stats.ttest_1samp against 0, two-sided",
            "wilcoxon_p": "scipy.stats.wilcoxon against 0, two-sided, exact",
        },
        "aggregates_18i": {arm: pool_arm(records_18i, arm) for arm in ARMS},
        "aggregates_pool": {arm: pool_arm(records, arm) for arm in ARMS},
        "cue_vs_nocue_18i": pool_differences(records_18i),
        "cue_vs_nocue_pool": pool_differences(records),
    }


def pool_arm(records: list[RunRecord], arm: str) -> dict[str, Any]:
    """Seed-level statistics of one arm, with the values in the order of `records`."""
    members = [record for record in records if record.arm == arm]
    pooled: dict[str, Any] = {
        "n_seeds": len(members),
        "seeds": [record.seed for record in members],
        "sources": [record.source for record in members],
        "crossed_count": sum(record.crossed for record in members),
    }
    for statistic in members[0].statistics:
        pooled[statistic] = summarize_values([record.statistics[statistic] for record in members])
    return pooled


def pool_differences(records: list[RunRecord]) -> dict[str, Any]:
    """Paired cue - nocue differences by seed, with intervals, sign counts and p-values."""
    by_arm_and_seed = {(record.arm, record.seed): record for record in records}
    seeds = sorted({record.seed for record in records})
    per_seed = []
    for seed in seeds:
        cue, nocue = by_arm_and_seed[(ARM_CUE, seed)], by_arm_and_seed[(ARM_NOCUE, seed)]
        assert cue.source == nocue.source
        cue_terms, nocue_terms = difference_terms(cue), difference_terms(nocue)
        row: dict[str, Any] = {"seed": seed, "source": cue.source}
        for name in DIFFERENCE_NAMES:
            row[name] = cue_terms[name] - nocue_terms[name]
        row.update(cue_crossed=cue.crossed, nocue_crossed=nocue.crossed)
        per_seed.append(row)
    pooled: dict[str, Any] = {"n": len(seeds), "per_seed": per_seed}
    for name in DIFFERENCE_NAMES:
        values = [row[name] for row in per_seed]
        pooled[name] = {**summarize_values(values), **difference_tests(np.asarray(values, float))}
    pooled["cue_crossed_count"] = sum(row["cue_crossed"] for row in per_seed)
    pooled["nocue_crossed_count"] = sum(row["nocue_crossed"] for row in per_seed)
    return pooled


def difference_terms(record: RunRecord) -> dict[str, float]:
    """The per-run quantities whose cue - nocue difference is reported."""
    values = record.statistics
    return {
        "d_gAminusgB": values["gA"] - values["gB"],
        "d_poison_B": values["poison_B"],
        "d_within_aligned": values["within_aligned"],
    }


def summarize_values(values: list[float]) -> dict[str, Any]:
    """Mean, bootstrap 95% interval, count and sample standard deviation of seed values."""
    array = np.asarray(values, float)
    assert array.ndim == 1
    assert array.size >= 2
    assert np.all(np.isfinite(array))
    return {
        "mean": float(np.mean(array)),
        "ci95": bootstrap_interval(array),
        "n": int(array.size),
        "std": float(np.std(array, ddof=1)),
        "values": [float(value) for value in array],
    }


def bootstrap_interval(values: np.ndarray) -> list[float]:
    """Percentile bootstrap interval for the mean; the draw depends on the value order."""
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    resamples = generator.choice(values, size=(BOOTSTRAP_RESAMPLE_COUNT, values.size), replace=True)
    lower, upper = np.percentile(resamples.mean(axis=1), INTERVAL_PERCENTILES)
    assert lower <= upper
    return [float(lower), float(upper)]


def difference_tests(values: np.ndarray) -> dict[str, Any]:
    """Sign counts and two-sided p-values against 0 for one set of paired differences."""
    magnitudes = np.abs(values)
    if np.any(magnitudes == 0.0):
        raise PoolError("a paired difference is exactly 0; the exact Wilcoxon test excludes it")
    if np.unique(magnitudes).size != magnitudes.size:
        raise PoolError("paired differences have tied magnitudes; no exact Wilcoxon p-value")
    t_test = stats.ttest_1samp(values, popmean=0.0, alternative="two-sided")
    wilcoxon = stats.wilcoxon(values, zero_method="wilcox", alternative="two-sided", method="exact")
    assert math.isfinite(t_test.pvalue)
    assert math.isfinite(wilcoxon.pvalue)
    return {
        "n_pos": int(np.sum(values > 0.0)),
        "n_neg": int(np.sum(values < 0.0)),
        "t_test_p": float(t_test.pvalue),
        "wilcoxon_p": float(wilcoxon.pvalue),
    }


def check_raw(records: list[RunRecord], raw_directory: Path) -> dict[str, Any]:
    """Recompute within_aligned from the raw logs; raise PoolError unless every run matches."""
    committed = {record.tag: record.statistics["within_aligned"] for record in records}
    recomputed: dict[str, float] = {}
    for pattern in RAW_PATTERN.values():
        recomputed.update(load_raw_within_aligned(raw_directory, pattern))
    if sorted(recomputed) != sorted(committed):
        raise PoolError(
            f"raw logs under {raw_directory} hold runs {sorted(recomputed)}, "
            f"expected {sorted(committed)}"
        )
    differences = {tag: abs(recomputed[tag] - committed[tag]) for tag in committed}
    failed = sorted(tag for tag, difference in differences.items() if difference > RAW_TOLERANCE)
    if failed:
        raise PoolError(f"within_aligned from the raw logs differs from the summary for {failed}")
    return {"run_count": len(differences), "difference_abs_max": max(differences.values())}


def load_raw_within_aligned(raw_directory: Path, pattern: str) -> dict[str, float]:
    """Return {tag: within_aligned} for every raw pickle that matches `pattern`."""
    paths = sorted(raw_directory.glob(pattern))
    if not paths:
        raise PoolError(f"no raw pickles match {raw_directory / pattern}")
    values: dict[str, float] = {}
    for path in paths:
        try:
            runs = analyze18c.load(glob.escape(str(path)))
            for run in runs:
                values[run["tag"]] = within_aligned_from_log(run["log"], float(run["eco"]["Ts"]))
        except RAW_LOAD_ERRORS as error:
            raise PoolError(f"cannot use raw pickle {path}: {error!r}") from error
    return values


def within_aligned_from_log(log: list[Any], season_period: float) -> float:
    """Mean over the season switches of within * sign(sigma_after), as in analyze18h.py."""
    sigmas = [fl_season.snapshot_season(time, season_period) for time, _ in log]
    switches = metrics18c.decompose_switch(log, GRID_SIDE, sigmas, analyze18c.I_GA, analyze18c.I_GB)
    aligned = [
        switch["within"] * (1.0 if switch["sigma_after"] > 0 else -1.0) for switch in switches
    ]
    if not aligned:
        raise PoolError("a raw log has no season switch")
    if not all(math.isfinite(value) for value in aligned):
        raise PoolError("a raw log gives a non-finite within value at a season switch")
    return float(np.mean(aligned))


def write_summary(summary: dict[str, Any], output: Path) -> None:
    """Write the summary as strict JSON; a write failure becomes a PoolError."""
    text = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    try:
        output.write_text(text)
    except OSError as error:
        raise PoolError(f"cannot write {output}: {error}") from error


def print_report(summary: dict[str, Any], raw_check: dict[str, Any] | None) -> None:
    """Print the pooled within_aligned of each arm and the three paired differences."""
    for arm, pooled in summary["aggregates_pool"].items():
        statistic = pooled["within_aligned"]
        lower, upper = statistic["ci95"]
        print(f"{arm}: within_aligned {statistic['mean']:+.3f} [{lower:+.3f}, {upper:+.3f}]")
    differences = summary["cue_vs_nocue_pool"]
    for name in DIFFERENCE_NAMES:
        statistic = differences[name]
        lower, upper = statistic["ci95"]
        print(
            f"{name}: {statistic['mean']:+.3f} [{lower:+.3f}, {upper:+.3f}] "
            f"n_pos {statistic['n_pos']}/{statistic['n']} "
            f"t-test p {statistic['t_test_p']:.3f} Wilcoxon p {statistic['wilcoxon_p']:.3f}"
        )
    if raw_check is None:
        print("raw logs: not checked (no --raw-directory)")
    else:
        print(
            f"raw logs: within_aligned of {raw_check['run_count']} runs matches the summaries, "
            f"largest difference {raw_check['difference_abs_max']:.1e}"
        )


if __name__ == "__main__":
    sys.exit(main())
