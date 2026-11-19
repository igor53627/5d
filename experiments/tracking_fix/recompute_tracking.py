"""Recompute lineage-based numbers of Parts 10-17 with trait-fingerprint tracking.

Usage: python recompute_tracking.py DATA_ROOT [OUTPUT_DIR]

DATA_ROOT is a local copy of the Hugging Face dataset ``igor53627/5d``. The pickles are the
project's own data; ``pickle.load`` runs code, so do not point this script at other files.

For each run the script reports the Price selection term three ways:

* ``published``: ``analyze_eco.selection_diff`` (unweighted, nearest-in-space ancestors);
* ``nearest``: mass-weighted term, nearest-in-space ancestors;
* ``tracked``: mass-weighted term, ancestors identified by all heritable traits;
* ``check``: as ``tracked``, but ancestors identified by traits that do not enter the
  analysed trait (p0 and p1 for brain outputs, p1 alone when p0 is analysed).

It also reports how often the two ancestor rules differ, and the division count, the
parent-child correlation and the effective number of founder lineages under each rule.
Runs without transport of the heritable field have no trait fingerprint; for them every
column uses the nearest object in space.
"""

from __future__ import annotations

import glob
import json
import pickle
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy import stats

# The import of _path puts src/ and experiments/part*/ on sys.path, so it must come first.
# isort: off
import _path  # noqa: F401
import analyze12
import analyze15b
import analyze17
import analyze_eco
import fl_brain
import lineage_tracking
import price_equation
# isort: on

LAG_SNAPSHOTS: int = 5
GATE_RADIUS_WITHOUT_TAXIS: float = 20.0
GATE_RADIUS_WITH_TAXIS: float = 30.0
TRACKING_TIME_MIN: float = 2000.0
BRAIN_TRAIT_COUNT: int = fl_brain.K0 + fl_brain.NW

TraitFunction = Callable[[list, dict], np.ndarray]
Window = tuple[float, float]


@dataclass(frozen=True)
class Run:
    """One simulation log with the facts the analysis needs."""

    part: str
    tag: str
    condition: str
    box_size: int
    log: list
    eco: dict
    transport: bool


@dataclass(frozen=True)
class PartConfig:
    """How one part is loaded and which traits and phases its notes report."""

    pattern: str
    tag_filter: str
    box_size: int | None
    fingerprint_count: int
    check_fingerprint: tuple[int, ...]
    gate_radius: float
    traits: dict[str, TraitFunction]
    phases: Callable[[Run], dict[str, list[Window]]]


def trait_p0(objects: list, eco: dict) -> np.ndarray:
    """Heritable parameter p0 of each object."""
    return np.array([o[2][0] for o in objects], float)


def trait_gain_at_zero_input(output_index: int) -> TraitFunction:
    """Part 15: brain output at zero input, as in ``analyze15.augment``."""

    def trait(objects: list, eco: dict) -> np.ndarray:
        zero_input: np.ndarray = np.zeros(fl_brain.NIN)
        weights = [np.asarray(o[2], float)[fl_brain.K0 : BRAIN_TRAIT_COUNT] for o in objects]
        return np.array([fl_brain.brain(w, zero_input, 2.0)[output_index] for w in weights])

    return trait


def trait_gain_at_reference_input(output_index: int) -> TraitFunction:
    """Parts 15b and 16: brain output at the reference input ``analyze15b.SSTAR``."""

    def trait(objects: list, eco: dict) -> np.ndarray:
        weights: np.ndarray = np.array([o[2][fl_brain.K0 : BRAIN_TRAIT_COUNT] for o in objects])
        gains: np.ndarray = analyze15b.brain_batch(weights, analyze15b.SSTAR)[:, 0, output_index]
        is_clamped_repulsion: bool = output_index == 1 and eco.get("rclamp", 0) > 0
        return np.maximum(gains, 0.0) if is_clamped_repulsion else gains

    return trait


def trait_own_gain(output_index: int) -> TraitFunction:
    """Part 17: realised gain of each object at its own inputs (``analyze17.own_gains``)."""

    def trait(objects: list, eco: dict) -> np.ndarray:
        return analyze17.own_gains(objects, eco.get("blind", 0) > 0)[:, output_index]

    return trait


def whole_run(run: Run) -> dict[str, list[Window]]:
    """One phase that covers the run."""
    return {"all": [(0.0, float(run.log[-1][0]))]}


def growth_and_capacity(run: Run) -> dict[str, list[Window]]:
    """Part 10 phases from ``plot_eco10.py``; Part 12 plateau from ``analyze12.metrics``."""
    time_end: float = float(run.log[-1][0])
    return {
        "growth": [(0.0, 7000.0)],
        "capacity": [(10000.0, time_end)],
        "plateau": [(float(time_end // 2), time_end)],
    }


def early_phase(run: Run) -> dict[str, list[Window]]:
    """Parts 15b and 16: the first 5000 steps."""
    return {"0-5k": [(0.0, 5000.0)]}


def sliding_growth_and_plateau(run: Run) -> dict[str, list[Window]]:
    """Part 15 phases from ``analyze15.one``: 3000-step windows every 2000 steps."""
    times: np.ndarray = np.array([snapshot[0] for snapshot in run.log], float)
    counts = [sum(1 for o in objects if o[0] >= 0.3) for _, objects in run.log]
    time_plateau, _ = analyze12.t_plateau(times, counts)
    time_end: float = float(times[-1])
    starts = range(0, int(time_end) - 1000, 2000)
    growth_limit: float = max(time_plateau, 2000)
    return {
        "growth": [(float(a), a + 3000.0) for a in starts if a < growth_limit],
        "plateau": [(float(a), a + 3000.0) for a in starts if a >= time_end // 2],
    }


def _part(pattern: str, traits: dict, phases: Callable, **overrides: object) -> PartConfig:
    """A part without brain taxis: two heritable traits, p1 as the independent check."""
    settings: dict = {
        "pattern": pattern,
        "tag_filter": r".",
        "box_size": None,
        "fingerprint_count": 2,
        "check_fingerprint": (1,),
        "gate_radius": GATE_RADIUS_WITHOUT_TAXIS,
        "traits": traits,
        "phases": phases,
    }
    return PartConfig(**{**settings, **overrides})


def _brain_part(pattern: str, traits: dict, phases: Callable, **overrides: object) -> PartConfig:
    """A part with a micro-brain: all weights identify the ancestor, (p0, p1) is the check."""
    brain_settings: dict = {
        "fingerprint_count": BRAIN_TRAIT_COUNT,
        "check_fingerprint": (0, 1),
        "gate_radius": GATE_RADIUS_WITH_TAXIS,
    }
    return _part(pattern, traits, phases, **{**brain_settings, **overrides})


def sliding_growth_and_whole_plateau(run: Run) -> dict[str, list[Window]]:
    """Part 13 phases from ``analyze13.one``: sliding growth windows, one plateau window."""
    time_end: float = float(run.log[-1][0])
    return {
        "growth": sliding_growth_and_plateau(run)["growth"],
        "plateau": [(float(time_end // 2), time_end)],
    }


REFERENCE_GAINS: dict[str, TraitFunction] = {
    "gF": trait_gain_at_reference_input(0),
    "gR": trait_gain_at_reference_input(1),
}
OWN_GAINS: dict[str, TraitFunction] = {
    "gF": trait_own_gain(0),
    "gP": trait_own_gain(1),
    "p0": trait_p0,
}
PARTS: dict[str, PartConfig] = {
    "10": _part("part10/eco10/eco10m_*.pkl", {"p0": trait_p0}, growth_and_capacity, box_size=96),
    "12": _part(
        "part12/eco12/eco12m_*.pkl",
        {"p0": trait_p0},
        growth_and_capacity,
        box_size=96,
        tag_filter=r"^(sel|neu|res2k)_96_",
    ),
    "13": _part("part13/eco13/eco13m_*.pkl", {"p0": trait_p0}, sliding_growth_and_whole_plateau),
    "15": _brain_part(
        "part15/eco15/b15m_*.pkl", {"gF": trait_gain_at_zero_input(0)}, sliding_growth_and_plateau
    ),
    "15b": _brain_part("part15/eco15/b15d5_*.pkl", REFERENCE_GAINS, early_phase),
    "16": _brain_part("part16/eco16/b16_*.pkl", REFERENCE_GAINS, early_phase),
    "17": _brain_part(
        "part17/eco17/b17m_*.pkl",
        OWN_GAINS,
        whole_run,
        box_size=56,
        fingerprint_count=analyze17.iD,
    ),
}


def _tagged_runs(payload: dict) -> list[tuple[str, dict]]:
    """Return ``(tag, run)`` pairs from one result pickle; both stored layouts are accepted."""
    if "runs" in payload:
        return [(str(payload["tag"]), run) for run in payload["runs"][:1]]
    return [
        (tag, value["runs"][0])
        for tag, value in payload.items()
        if isinstance(value, dict) and value.get("runs")
    ]


def load_runs(data_root: Path, part: str) -> list[Run]:
    """Load the complete runs of one part; a missing part is an error, not an empty result."""
    config: PartConfig = PARTS[part]
    files: list[str] = sorted(glob.glob(str(data_root / config.pattern)))
    if not files:
        raise FileNotFoundError(f"no files match {data_root / config.pattern}")
    runs: list[Run] = []
    for file_name in files:
        with open(file_name, "rb") as handle:
            payload = pickle.load(handle)["out"][0]
        for tag, raw in _tagged_runs(payload):
            if not re.search(config.tag_filter, tag) or raw.get("partial"):
                continue
            size_match = re.search(r"_\dD(\d+)", tag)
            box_size = int(size_match.group(1)) if size_match else config.box_size
            if box_size is None:
                raise ValueError(f"cannot find the box size of run {tag}")
            condition: str = re.sub(r"_s\d+$", "", tag)
            runs.append(
                Run(part, tag, condition, box_size, raw["log"], raw["eco"], bool(raw["transport"]))
            )
    return runs


def with_trait(log: list, eco: dict, fingerprint_count: int, trait: TraitFunction) -> list:
    """Copy of the log whose trait lists are the heritable traits followed by one trait value."""
    extended: list = []
    for time, objects in log:
        values: np.ndarray = trait(objects, eco) if objects else np.zeros(0)
        extended.append(
            (
                time,
                [
                    (o[0], o[1], [*o[2][:fingerprint_count], float(value)])
                    for o, value in zip(objects, values)
                ],
            )
        )
    return extended


TERM_NAMES: tuple[str, ...] = (
    "published_S",
    "nearest_S",
    "nearest_T",
    "tracked_S",
    "tracked_T",
    "check_S",
    "check_T",
    "delta",
)


def window_row(sub_log: list, run: Run, config: PartConfig) -> list[float]:
    """Price terms of one window under each estimator, in the order of ``TERM_NAMES``."""
    trait_index: int = config.fingerprint_count
    published = analyze_eco.selection_diff(sub_log, run.box_size, LAG_SNAPSHOTS, trait_index)
    row: list[float] = [published[0]]
    rules: list[list[int] | None] = [None, None, None]
    if run.transport:
        rules = [None, list(range(trait_index)), list(config.check_fingerprint)]
    delta: float = float("nan")
    for fingerprint in rules:
        terms = price_equation.price_terms(
            sub_log,
            run.box_size,
            lag=LAG_SNAPSHOTS,
            trait_index=trait_index,
            fingerprint_indices=fingerprint,
            gate_radius=config.gate_radius,
        )
        row += [terms.mean("selection"), terms.mean("transmission")]
        delta = terms.mean("delta")
    return [*row, delta]


def phase_terms(log: list, run: Run, config: PartConfig, windows: Sequence[Window]) -> dict:
    """Mean Price terms of one trait over the windows of one phase."""
    rows: list[list[float]] = []
    for time_low, time_high in windows:
        sub_log = [snapshot for snapshot in log if time_low <= snapshot[0] <= time_high]
        if len(sub_log) > LAG_SNAPSHOTS:
            rows.append(window_row(sub_log, run, config))
    if not rows:
        return dict.fromkeys(TERM_NAMES, float("nan"))
    table: np.ndarray = np.array(rows, float)
    usable: np.ndarray = table[np.isfinite(table).all(axis=1)]
    if not len(usable):
        return dict.fromkeys(TERM_NAMES, float("nan"))
    return dict(zip(TERM_NAMES, usable.mean(axis=0).tolist()))


def lineage_numbers(log: list, run: Run, config: PartConfig) -> dict:
    """Division count, parent-child correlation and lineage diversity under each rule."""
    trait_index: int = config.fingerprint_count
    rules: dict[str, list[int] | None] = {"nearest": None}
    if run.transport:
        rules["tracked"] = list(range(trait_index))
        rules["check"] = list(config.check_fingerprint)
    numbers: dict = {}
    for rule, fingerprint in rules.items():
        division_count, pairs = lineage_tracking.division_pairs(
            log, run.box_size, trait_index, fingerprint, config.gate_radius
        )
        shares = lineage_tracking.founder_mass_shares(
            log, run.box_size, fingerprint, config.gate_radius
        )
        numbers[rule] = {
            "divisions": division_count,
            "parent_child_r": lineage_tracking.parent_child_correlation(pairs),
            "effective_lineages": lineage_tracking.effective_lineage_count(shares),
        }
    return numbers


def check_indices(config: PartConfig, trait_name: str) -> tuple[int, ...]:
    """Trait indices of the independent rule; p0 is left out when p0 itself is analysed."""
    is_p0: bool = trait_name == "p0"
    return tuple(index for index in config.check_fingerprint if not (is_p0 and index == 0))


def analyse_run(run: Run) -> dict:
    """All recomputed numbers of one run."""
    config: PartConfig = PARTS[run.part]
    record: dict = {
        "part": run.part,
        "tag": run.tag,
        "condition": run.condition,
        "transport": run.transport,
        "tracking_disagreement": float("nan"),
        "price": {},
        "lineage": {},
    }
    if run.transport:
        fraction, _ = lineage_tracking.tracking_disagreement(
            run.log,
            run.box_size,
            list(range(config.fingerprint_count)),
            config.gate_radius,
            TRACKING_TIME_MIN,
        )
        record["tracking_disagreement"] = fraction
    phases = config.phases(run)
    for trait_name, trait in config.traits.items():
        log = with_trait(run.log, run.eco, config.fingerprint_count, trait)
        trait_config = replace(config, check_fingerprint=check_indices(config, trait_name))
        record["price"][trait_name] = {
            phase: phase_terms(log, run, trait_config, windows) for phase, windows in phases.items()
        }
        record["lineage"][trait_name] = lineage_numbers(log, run, trait_config)
    return record


def _one_sample_p(values: Sequence[float]) -> float:
    """Two-sided one-sample t-test against 0 across seeds; NaN below 3 finite values."""
    finite = [v for v in values if np.isfinite(v)]
    if len(finite) < 3:
        return float("nan")
    return float(stats.ttest_1samp(finite, 0.0).pvalue)


def _finite_mean(values: Sequence[float]) -> float:
    """Mean of the finite values; NaN when there is none."""
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def summarise(records: Sequence[dict]) -> list[dict]:
    """Seed means per part, condition, trait and phase, with t-tests of the tracked terms."""
    groups: dict[tuple, list[dict]] = {}
    for record in records:
        for trait_name, phases in record["price"].items():
            for phase, terms in phases.items():
                key = (record["part"], record["condition"], trait_name, phase)
                groups.setdefault(key, []).append(terms)
    rows: list[dict] = []
    for (part, condition, trait_name, phase), terms_list in sorted(groups.items()):
        row: dict = {"part": part, "condition": condition, "trait": trait_name, "phase": phase}
        row["seeds"] = len(terms_list)
        for name in TERM_NAMES:
            row[name] = _finite_mean([terms[name] for terms in terms_list])
        tracked_selection = [terms["tracked_S"] for terms in terms_list]
        row["tracked_S_p"] = _one_sample_p(tracked_selection)
        row["tracked_T_p"] = _one_sample_p([terms["tracked_T"] for terms in terms_list])
        row["tracked_S_positive"] = int(sum(v > 0 for v in tracked_selection))
        rows.append(row)
    return rows


def tracking_summary(records: Sequence[dict]) -> list[dict]:
    """Per part and condition: ancestor disagreement and the lineage numbers of each rule."""
    groups: dict[tuple, list[dict]] = {}
    for record in records:
        groups.setdefault((record["part"], record["condition"]), []).append(record)
    rows: list[dict] = []
    for (part, condition), group in sorted(groups.items()):
        trait_name: str = next(iter(group[0]["lineage"]))
        row: dict = {"part": part, "condition": condition, "seeds": len(group)}
        row["trait"] = trait_name
        row["disagreement"] = _finite_mean([r["tracking_disagreement"] for r in group])
        for rule in ("nearest", "tracked", "check"):
            for name in ("divisions", "parent_child_r", "effective_lineages"):
                values = [
                    r["lineage"][trait_name].get(rule, {}).get(name, float("nan")) for r in group
                ]
                row[f"{name}_{rule}"] = _finite_mean(values)
        rows.append(row)
    return rows


PRICE_HEADER: str = (
    "| part | condition | trait | phase | seeds | S published | S nearest | S tracked "
    "| T tracked | S check | delta | p(S tracked) | p(T tracked) | S tracked > 0 |"
)
PRICE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("published_S", "+.3f"),
    ("nearest_S", "+.3f"),
    ("tracked_S", "+.3f"),
    ("tracked_T", "+.3f"),
    ("check_S", "+.3f"),
    ("delta", "+.3f"),
    ("tracked_S_p", ".3g"),
    ("tracked_T_p", ".3g"),
)
TRACKING_HEADER: str = (
    "| part | condition | seeds | nearest differs from tracked | divisions: nearest "
    "| tracked | check | trait | parent-child r: nearest | tracked | check "
    "| effective lineages: nearest | tracked | check |"
)
TRACKING_COLUMNS: tuple[tuple[str, str], ...] = (
    ("disagreement", ".1%"),
    ("divisions_nearest", ".0f"),
    ("divisions_tracked", ".0f"),
    ("divisions_check", ".0f"),
    ("trait", ""),
    ("parent_child_r_nearest", "+.2f"),
    ("parent_child_r_tracked", "+.2f"),
    ("parent_child_r_check", "+.2f"),
    ("effective_lineages_nearest", ".2f"),
    ("effective_lineages_tracked", ".2f"),
    ("effective_lineages_check", ".2f"),
)


def _cell(value: object, format_spec: str) -> str:
    """Format one table cell; a value that is not finite is shown as ``n/a``."""
    if isinstance(value, float) and not np.isfinite(value):
        return "n/a"
    return format(value, format_spec)


def _table_row(leading: Sequence[object], row: dict, columns: Sequence[tuple[str, str]]) -> str:
    """One Markdown table row: the leading cells, then the formatted columns of ``row``."""
    cells: list[str] = [str(cell) for cell in leading]
    cells += [_cell(row[name], format_spec) for name, format_spec in columns]
    return "| " + " | ".join(cells) + " |"


def markdown(price_rows: Sequence[dict], tracking_rows: Sequence[dict]) -> str:
    """Render both summaries as Markdown tables."""
    lines: list[str] = ["## Price terms (mean over seeds, lag 1000 steps)", "", PRICE_HEADER]
    lines.append("|" + "---|" * 14)
    for row in price_rows:
        leading = [row[name] for name in ("part", "condition", "trait", "phase", "seeds")]
        positive: str = f" {row['tracked_S_positive']}/{row['seeds']} |"
        lines.append(_table_row(leading, row, PRICE_COLUMNS) + positive)
    lines += ["", "## Tracking (mean over seeds)", "", TRACKING_HEADER, "|" + "---|" * 14]
    for row in tracking_rows:
        leading = [row[name] for name in ("part", "condition", "seeds")]
        lines.append(_table_row(leading, row, TRACKING_COLUMNS))
    return "\n".join(lines) + "\n"


def json_safe(value: object) -> object:
    """Copy of nested results in which every value that is not finite is ``None``.

    The nesting is fixed by this script (records, phases, terms), so the depth is bounded.
    """
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main(arguments: Sequence[str]) -> int:
    """Recompute every configured part and write the JSON and Markdown results."""
    if not 1 <= len(arguments) <= 2:
        print(__doc__, file=sys.stderr)
        return 2
    data_root: Path = Path(arguments[0])
    default_output: Path = Path(__file__).resolve().parent / "results"
    output_dir: Path = Path(arguments[1]) if len(arguments) == 2 else default_output
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for part in PARTS:
        runs = load_runs(data_root, part)
        print(f"part {part}: {len(runs)} runs", file=sys.stderr)
        records += [analyse_run(run) for run in runs]
    price_rows = summarise(records)
    tracking_rows = tracking_summary(records)
    result = {"runs": records, "price": price_rows, "tracking": tracking_rows}
    json_text: str = json.dumps(json_safe(result), indent=1, allow_nan=False)
    (output_dir / "tracking_recompute.json").write_text(json_text + "\n")
    table: str = markdown(price_rows, tracking_rows)
    (output_dir / "tracking_recompute.md").write_text(table)
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
