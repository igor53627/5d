"""Compare the ancestor rules with the true ancestors that ``run_truth.py`` recorded.

Usage: python analyze_truth.py DATA_ROOT [OUTPUT_DIRECTORY]

DATA_ROOT is a local copy of the Hugging Face dataset ``igor53627/5d``; the truth files are in
its folder ``tracking_truth``. The raw files are not in this repository.

For every ``truth_*.json`` file the script takes three ancestor rules of
``lineage_tracking.assign_ancestors`` and the measured tracer shares:

* ``nearest``: the nearest object in space (the rule of the published analysis);
* ``tracked``: the object with the closest (p0, p1) inside the gate;
* ``check``: the object with the closest p1 inside the gate;
* ``truth``: the share of the mass of a descendant that comes from each earlier object.
  The truth files hold these shares relative to the traced mass, and the share of the mass
  that was outside every earlier object ("background") next to them. The script converts
  them to shares of the whole mass, so that a chain of steps credits the background to no
  ancestor.

It reports how often each rule picks the object that gave a descendant most of its mass,
which rule is right where the nearest object and the (p0, p1) rule differ, and what each rule
gives for the Price terms of p0, the division count and the effective number
of founder lineages. It writes ``truth_summary.md`` and ``truth_summary.json`` to
OUTPUT_DIRECTORY, which is ``results`` next to this script when it is not given.

A share is measured over one snapshot step (200 simulation steps), because ``run_truth.py``
gives every body a new tracer at each snapshot. Over a longer span the ``truth`` estimator
multiplies these shares, as the rules do. The product is exact only when two conditions hold.
First, the mass that a body passes on has the same mix of origins as the whole body. Second, no
mass passes from one body to another through background: the product credits no ancestor for
mass that is outside every body at a snapshot. The agreement counts and the division counts use
single steps and do not depend on this.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

# The import of _path puts src/ and experiments/part*/ on sys.path, so it must come first.
# isort: off
import _path  # noqa: F401
import lineage_tracking
import price_equation
# isort: on

DEFAULT_OUTPUT: Path = Path(__file__).resolve().parent / "results"
DATA_FOLDER: str = "tracking_truth"
GATE_RADIUS: float = 20.0
LAG_SNAPSHOTS: int = 5
TRAIT_INDEX: int = 0
PURE_SHARE_MIN: float = 0.9
RULES: dict[str, list[int] | None] = {"nearest": None, "tracked": [0, 1], "check": [1]}
ESTIMATORS: tuple[str, ...] = ("truth", *RULES)

AGREEMENT_HEADER: str = (
    "| run | N | box | seed | steps | objects | descendants | pure | nearest right "
    "| tracked right | check right | nearest right, pure | tracked right, pure "
    "| check right, pure | background | fraction sum error |"
)
ESTIMATE_HEADER: str = (
    "| run | seed | estimator | S first half | S second half | T first half "
    "| T second half | divisions | effective lineages |"
)

CHAIN_NOTE: str = (
    "The `truth` rows multiply the measured 200-step shares over the span. "
    "The notes state the assumption (Limits of the chained numbers)."
)
DISAGREEMENT_HEADER: str = (
    "| run | seed | descendants | nearest differs from tracked | tracked is right "
    "| nearest is right | neither is right |"
)

ShareMatrices = list


def load_run(path: Path) -> dict:
    """One truth file: the log as tuples, the true shares as dense matrices of whole mass."""
    raw: dict = json.loads(path.read_text())
    log: list = [(time, [tuple(item) for item in objects]) for time, objects in raw["log"]]
    matrices: ShareMatrices = [None]
    for step, record in enumerate(raw["truth"]):
        ancestors, descendants = log[step][1], log[step + 1][1]
        if not ancestors or not descendants:
            matrices.append(None)
            continue
        shares: np.ndarray = np.zeros((len(descendants), len(ancestors)))
        for descendant, ancestor, share in record["shares"]:
            shares[descendant, ancestor] = share
        traced: np.ndarray = 1.0 - np.array(record["background"], float)
        matrices.append(shares * traced[:, None])
    return {**raw, "log": log, "truth_matrices": matrices}


def rule_matrices(run: dict) -> dict[str, ShareMatrices]:
    """Share matrices of every estimator: measured for the truth, one-hot for the rules."""
    matrices: dict[str, ShareMatrices] = {"truth": run["truth_matrices"]}
    for rule, fingerprint in RULES.items():
        matrices[rule] = price_equation.step_share_matrices(
            run["log"], run["box_size"], fingerprint, GATE_RADIUS
        )
    return matrices


def agreement_counts(matrices: dict[str, ShareMatrices]) -> dict[str, int]:
    """Counts of descendants, and of descendants for which each rule picks the main ancestor.

    The main ancestor of a descendant is the earlier object that gave it most of its mass.
    A descendant is "pure" when that share of its whole mass is at least ``PURE_SHARE_MIN``.
    A pair of snapshots with one earlier object is skipped: every rule is right there by
    construction.
    """
    counts: dict[str, int] = dict.fromkeys(
        ["descendants", "traced", "pure", *[f"{r}_right" for r in RULES]], 0
    )
    counts.update(dict.fromkeys([f"{rule}_right_pure" for rule in RULES], 0))
    for step, truth in enumerate(matrices["truth"]):
        if truth is None or truth.shape[1] < 2:
            continue
        traced: np.ndarray = truth.sum(axis=1) > 0
        main: np.ndarray = truth.argmax(axis=1)
        pure: np.ndarray = traced & (truth.max(axis=1) >= PURE_SHARE_MIN)
        counts["descendants"] += len(traced)
        counts["traced"] += int(traced.sum())
        counts["pure"] += int(pure.sum())
        for rule in RULES:
            right: np.ndarray = matrices[rule][step].argmax(axis=1) == main
            counts[f"{rule}_right"] += int((right & traced).sum())
            counts[f"{rule}_right_pure"] += int((right & pure).sum())
    return counts


def disagreement_counts(matrices: dict[str, ShareMatrices]) -> dict[str, int]:
    """Descendants for which ``nearest`` and ``tracked`` differ, and which of them is right.

    ``differ`` over ``traced`` is the disagreement rate that the notes of the tracking fix
    report for the published logs, where no true ancestor exists.
    """
    names: tuple[str, ...] = ("traced", "differ", "tracked_right", "nearest_right", "neither")
    counts: dict[str, int] = dict.fromkeys(names, 0)
    for step, truth in enumerate(matrices["truth"]):
        if truth is None or truth.shape[1] < 2:
            continue
        traced: np.ndarray = truth.sum(axis=1) > 0
        main: np.ndarray = truth.argmax(axis=1)
        by_space: np.ndarray = matrices["nearest"][step].argmax(axis=1)
        by_traits: np.ndarray = matrices["tracked"][step].argmax(axis=1)
        differ: np.ndarray = traced & (by_space != by_traits)
        counts["traced"] += int(traced.sum())
        counts["differ"] += int(differ.sum())
        counts["tracked_right"] += int((differ & (by_traits == main)).sum())
        counts["nearest_right"] += int((differ & (by_space == main)).sum())
        counts["neither"] += int((differ & (by_traits != main) & (by_space != main)).sum())
    return counts


def division_count(log: list, step_matrices: ShareMatrices) -> int:
    """Divisions as ``lineage_tracking.division_pairs`` counts them, from share matrices."""
    mass_min: float = lineage_tracking.DIVISION_MASS_MIN
    total: int = 0
    for step, shares in enumerate(step_matrices):
        if shares is None:
            continue
        ancestors, descendants = log[step - 1][1], log[step][1]
        main: np.ndarray = shares.argmax(axis=1)
        counted: np.ndarray = np.array([o[0] >= mass_min for o in descendants])
        counted &= shares.sum(axis=1) > 0
        children: np.ndarray = np.bincount(main[counted], minlength=len(ancestors))
        heavy: np.ndarray = np.array([o[0] >= mass_min for o in ancestors])
        total += int(((children >= 2) & heavy).sum())
    return total


def effective_lineages(log: list, step_matrices: ShareMatrices) -> float | None:
    """Effective number of founder lineages in the final mass; None when it is not defined.

    The founder credit is the product of the share matrices of all steps. It is ancestry
    composed from object to object, not a founder tracer that stays for the whole run: the
    product gives every descendant of an object the mean founder mix of that object. It is
    not defined across an empty snapshot or when no final mass has a founder.
    """
    if any(matrix is None for matrix in step_matrices[1:]):
        return None
    credit: np.ndarray = np.array([o[0] for o in log[-1][1]], float)
    for matrix in reversed(step_matrices[1:]):
        credit = credit @ matrix
    if credit.sum() <= 0:
        return None
    return lineage_tracking.effective_lineage_count(credit / credit.sum())


def run_summary(run: dict) -> dict:
    """All numbers of one run."""
    log: list = run["log"]
    matrices = rule_matrices(run)
    half_time: float = log[-1][0] / 2
    background: list[float] = [value for record in run["truth"] for value in record["background"]]
    summary: dict = {
        "tag": run["tag"],
        "condition": run["tag"].split("_")[0],
        "dimension": run["dimension"],
        "box_size": run["box_size"],
        "seed": run["seed"],
        "steps": run["steps"],
        "objects_mean": float(np.mean([len(objects) for _, objects in log])),
        "background_mean": float(np.mean(background)) if background else None,
        "fraction_sum_error_max": max(r["fraction_sum_error"] for r in run["truth"]),
        "counts": agreement_counts(matrices),
        "disagreement": disagreement_counts(matrices),
    }
    for name in ESTIMATORS:
        terms = price_equation.price_terms_from_shares(
            log, matrices[name], LAG_SNAPSHOTS, TRAIT_INDEX
        )
        early: np.ndarray = terms.times < half_time
        summary[name] = {
            "S_first_half": float(terms.selection[early].mean()) if early.any() else None,
            "S_second_half": float(terms.selection[~early].mean()) if (~early).any() else None,
            "T_first_half": float(terms.transmission[early].mean()) if early.any() else None,
            "T_second_half": float(terms.transmission[~early].mean()) if (~early).any() else None,
            "divisions": division_count(log, matrices[name]),
            "effective_lineages": effective_lineages(log, matrices[name]),
        }
    return summary


def share(part: int, whole: int) -> str:
    """A percentage, or n/a for an empty whole."""
    return f"{part / whole:.1%}" if whole else "n/a"


def signed(value: float | None) -> str:
    """A signed number with four decimals, or n/a."""
    return "n/a" if value is None or not np.isfinite(value) else f"{value:+.4f}"


def count(value: float | None, digits: int = 2) -> str:
    """A number with the given number of decimals, or n/a."""
    return "n/a" if value is None else f"{value:.{digits}f}"


def separator(header: str) -> str:
    """Markdown separator row with one cell per column of the header."""
    return "|" + "---|" * (header.count("|") - 1)


def disagreement_rows(summaries: Sequence[dict]) -> list[str]:
    """Rows of the table that says which rule is right where the two rules differ."""
    rows: list[str] = []
    for s in summaries:
        d = s["disagreement"]
        rows.append(
            f"| {s['tag']} | {s['seed']} | {d['traced']} | {share(d['differ'], d['traced'])} "
            f"| {share(d['tracked_right'], d['differ'])} "
            f"| {share(d['nearest_right'], d['differ'])} | {share(d['neither'], d['differ'])} |"
        )
    return rows


def markdown(summaries: Sequence[dict]) -> str:
    """The three tables of the summary file."""
    lines: list[str] = ["## Does each rule pick the main ancestor?", ""]
    lines += [AGREEMENT_HEADER, separator(AGREEMENT_HEADER)]
    for s in summaries:
        c = s["counts"]
        right = [share(c[f"{rule}_right"], c["traced"]) for rule in RULES]
        right_pure = [share(c[f"{rule}_right_pure"], c["pure"]) for rule in RULES]
        lines.append(
            f"| {s['tag']} | {s['dimension']} | {s['box_size']} | {s['seed']} | {s['steps']} "
            f"| {s['objects_mean']:.1f} | {c['traced']} | {share(c['pure'], c['traced'])} "
            f"| {' | '.join(right)} | {' | '.join(right_pure)} "
            f"| {count(s['background_mean'], 3)} | {s['fraction_sum_error_max']:.4f} |"
        )
    lines += ["", "## Where the nearest object and the (p0, p1) rule differ", ""]
    lines += [DISAGREEMENT_HEADER, separator(DISAGREEMENT_HEADER), *disagreement_rows(summaries)]
    lines += ["", "## What each rule gives (p0, lag 1000 steps)", ""]
    lines += [CHAIN_NOTE, ""]
    lines += [ESTIMATE_HEADER, separator(ESTIMATE_HEADER)]
    for s in summaries:
        for name in ESTIMATORS:
            e = s[name]
            lines.append(
                f"| {s['tag']} | {s['seed']} | {name} | {signed(e['S_first_half'])} "
                f"| {signed(e['S_second_half'])} | {signed(e['T_first_half'])} "
                f"| {signed(e['T_second_half'])} | {e['divisions']} "
                f"| {count(e['effective_lineages'])} |"
            )
    return "\n".join(lines) + "\n"


def main(arguments: Sequence[str]) -> int:
    """Analyse every truth file of the data folder and write the summary files."""
    if not 1 <= len(arguments) <= 2:
        print(__doc__, file=sys.stderr)
        return 2
    directory: Path = Path(arguments[0]) / DATA_FOLDER
    output_directory: Path = Path(arguments[1]) if len(arguments) == 2 else DEFAULT_OUTPUT
    files: list[Path] = sorted(directory.glob("truth_*_L*_s*.json"))
    if not files:
        raise FileNotFoundError(f"no truth files in {directory}")
    summaries: list[dict] = [run_summary(load_run(path)) for path in files]
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "truth_summary.json").write_text(
        json.dumps(summaries, indent=1, allow_nan=False) + "\n"
    )
    text: str = markdown(summaries)
    (output_directory / "truth_summary.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
