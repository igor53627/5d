"""Compare the ancestor rules with the true ancestors that ``run_truth.py`` recorded.

Usage: python analyze_truth.py [RESULTS_DIRECTORY]

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

It reports how often each rule picks the object that gave a descendant most of its mass, and
what each rule gives for the Price terms of p0, the division count and the effective number
of founder lineages. It writes ``truth_summary.md`` and ``truth_summary.json``.
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

DEFAULT_DIRECTORY: Path = Path(__file__).resolve().parent / "results"
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


def effective_lineages(log: list, step_matrices: ShareMatrices) -> float:
    """Effective number of founder lineages in the final mass; NaN across an empty snapshot."""
    if any(matrix is None for matrix in step_matrices[1:]):
        return float("nan")
    credit: np.ndarray = np.array([o[0] for o in log[-1][1]], float)
    for matrix in reversed(step_matrices[1:]):
        credit = credit @ matrix
    if credit.sum() <= 0:
        return float("nan")
    return lineage_tracking.effective_lineage_count(credit / credit.sum())


def run_summary(run: dict) -> dict:
    """All numbers of one run."""
    log: list = run["log"]
    matrices = rule_matrices(run)
    half_time: float = log[-1][0] / 2
    summary: dict = {
        "tag": run["tag"],
        "condition": run["tag"].split("_")[0],
        "dimension": run["dimension"],
        "box_size": run["box_size"],
        "seed": run["seed"],
        "steps": run["steps"],
        "objects_mean": float(np.mean([len(objects) for _, objects in log])),
        "background_mean": float(np.mean([v for r in run["truth"] for v in r["background"]])),
        "fraction_sum_error_max": max(r["fraction_sum_error"] for r in run["truth"]),
        "counts": agreement_counts(matrices),
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


def separator(header: str) -> str:
    """Markdown separator row with one cell per column of the header."""
    return "|" + "---|" * (header.count("|") - 1)


def markdown(summaries: Sequence[dict]) -> str:
    """The two tables of the summary file."""
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
            f"| {s['background_mean']:.3f} | {s['fraction_sum_error_max']:.4f} |"
        )
    lines += ["", "## What each rule gives (p0, lag 1000 steps)", ""]
    lines += [ESTIMATE_HEADER, separator(ESTIMATE_HEADER)]
    for s in summaries:
        for name in ESTIMATORS:
            e = s[name]
            lines.append(
                f"| {s['tag']} | {s['seed']} | {name} | {signed(e['S_first_half'])} "
                f"| {signed(e['S_second_half'])} | {signed(e['T_first_half'])} "
                f"| {signed(e['T_second_half'])} | {e['divisions']} "
                f"| {e['effective_lineages']:.2f} |"
            )
    return "\n".join(lines) + "\n"


def main(arguments: Sequence[str]) -> int:
    """Analyse every truth file of one directory."""
    if len(arguments) > 1:
        print(__doc__, file=sys.stderr)
        return 2
    directory: Path = Path(arguments[0]) if arguments else DEFAULT_DIRECTORY
    files: list[Path] = sorted(directory.glob("truth_*_L*_s*.json"))
    if not files:
        raise FileNotFoundError(f"no truth files in {directory}")
    summaries: list[dict] = [run_summary(load_run(path)) for path in files]
    (directory / "truth_summary.json").write_text(json.dumps(summaries, indent=1) + "\n")
    text: str = markdown(summaries)
    (directory / "truth_summary.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
