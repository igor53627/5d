"""Check the trait-fingerprint ancestor rule against independent evidence.

Usage: python validate_tracking.py DATA_ROOT [OUTPUT_DIR]

Three checks. The first and the third use logs with a micro-brain (24 or more heritable
weights); the second uses the logs of Parts 10, 12 and 13, which have two heritable traits:

* Split-half: the weights are split at random into two disjoint halves. Each half picks the
  ancestor of every descendant on its own. Agreement of the two halves means that the
  ancestor is identified by the traits and not by chance. A descendant with no ancestor or
  one ancestor inside the gate gets the same ancestor from both halves whatever the weights
  are, so the agreement is also reported for descendants with two or more ancestors inside
  the gate.
* Single traits: in Parts 10, 12 and 13 the fingerprint is (p0, p1). Here p0 alone and p1
  alone each pick the ancestor. One trait identifies a body less exactly than two, so this
  agreement is a lower bound on the reproducibility of the two-trait rule.
* Part 18 pairs: ``metrics18c.match_bodies`` pairs a body with the only body within 8 cells
  one snapshot later. The check counts how many pairs agree with the trait rule.
"""

from __future__ import annotations

import glob
import pickle
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

# The import of _path puts src/ and experiments/part*/ on sys.path, so it must come first.
# isort: off
import _path  # noqa: F401
import fl_season
import lineage_tracking
import metrics18c
import recompute_tracking
# isort: on

SPLIT_SEED: int = 0
SPLIT_TIME_MIN: float = 2000.0
PART_18_BOX_SIZE: int = 48
PART_18_PATTERN: str = "part18/eco18[gi]/b18[gi]_*.pkl"
SPLIT_COUNT_NAMES: tuple[str, ...] = (
    "descendants",
    "halves_agree",
    "nearest_agrees",
    "neutral_agrees",
    "gate_0",
    "gate_1",
    "gate_2_plus",
    "halves_agree_gate_2_plus",
)
SPLIT_HEADER: str = (
    "| part | condition | runs | descendants | two halves agree | nearest in space agrees "
    "| (p0, p1) agrees | ancestors in gate: 0 / 1 / 2+ | two halves agree, 2+ in gate |"
)
SINGLE_TRAIT_PARTS: tuple[str, ...] = ("10", "12", "13")
SINGLE_TRAIT_HEADER: str = (
    "| part | condition | runs | descendants | ancestors in gate: 0 / 1 / 2+ "
    "| p0 alone and p1 alone agree, 2+ in gate |"
)
PAIR_HEADER: str = (
    "| arm | pairs within 8 cells | agree with trait rule | share of old mass in pairs "
    "| nearest in space differs |"
)


def gated_ancestor_counts(
    ancestors: Sequence, descendants: Sequence, box_size: float, gate_radius: float
) -> np.ndarray:
    """Number of ancestors inside the gate of each descendant."""
    ancestor_centres: np.ndarray = np.array([o[1] for o in ancestors], float)
    descendant_centres: np.ndarray = np.array([o[1] for o in descendants], float)
    distances = lineage_tracking.periodic_distances(descendant_centres, ancestor_centres, box_size)
    return (distances <= gate_radius).sum(axis=1)


def agreement_counts(run: recompute_tracking.Run, halves: Sequence[list[int]]) -> np.ndarray:
    """Counts for one run, in the order of ``SPLIT_COUNT_NAMES``."""
    config = recompute_tracking.PARTS[run.part]
    rules: list[list[int] | None] = [
        list(range(config.fingerprint_count)),
        halves[0],
        halves[1],
        None,
        [0, 1],
    ]
    counts: np.ndarray = np.zeros(len(SPLIT_COUNT_NAMES))
    for (time, ancestors), (_, descendants) in zip(run.log[:-1], run.log[1:]):
        if time < SPLIT_TIME_MIN or len(ancestors) < 2 or not descendants:
            continue
        full, first_half, second_half, by_space, by_neutral = (
            lineage_tracking.assign_ancestors(
                ancestors, descendants, run.box_size, indices, config.gate_radius
            )
            for indices in rules
        )
        in_gate: np.ndarray = gated_ancestor_counts(
            ancestors, descendants, run.box_size, config.gate_radius
        )
        halves_agree: np.ndarray = first_half == second_half
        counts += [
            len(descendants),
            int(halves_agree.sum()),
            int((by_space == full).sum()),
            int((by_neutral == full).sum()),
            int((in_gate == 0).sum()),
            int((in_gate == 1).sum()),
            int((in_gate >= 2).sum()),
            int((halves_agree & (in_gate >= 2)).sum()),
        ]
    return counts


def split_half_rows(data_root: Path) -> list[str]:
    """Markdown rows of the split-half check for the brain parts 15, 15b, 16 and 17."""
    rows: list[str] = []
    for part in ("15", "15b", "16", "17"):
        config = recompute_tracking.PARTS[part]
        weights = np.arange(2, config.fingerprint_count)
        shuffled = np.random.default_rng(SPLIT_SEED).permutation(weights)
        halves = [sorted(shuffled[: len(shuffled) // 2]), sorted(shuffled[len(shuffled) // 2 :])]
        totals: dict[str, tuple[int, np.ndarray]] = {}
        for run in recompute_tracking.load_runs(data_root, part):
            run_count, counts = totals.get(run.condition, (0, np.zeros(len(SPLIT_COUNT_NAMES))))
            totals[run.condition] = (run_count + 1, counts + agreement_counts(run, halves))
        for condition, (run_count, counts) in sorted(totals.items()):
            count = dict(zip(SPLIT_COUNT_NAMES, counts))
            total: float = max(count["descendants"], 1)
            contested: float = count["gate_2_plus"]
            contested_share: str = (
                f"{count['halves_agree_gate_2_plus'] / contested:.1%}" if contested else "n/a"
            )
            rows.append(
                f"| {part} | {condition} | {run_count} | {count['descendants']:.0f} "
                f"| {count['halves_agree'] / total:.1%} | {count['nearest_agrees'] / total:.1%} "
                f"| {count['neutral_agrees'] / total:.1%} "
                f"| {count['gate_0']:.0f} / {count['gate_1']:.0f} / {contested:.0f} "
                f"| {contested_share} |"
            )
    return rows


def single_trait_rows(data_root: Path) -> list[str]:
    """Markdown rows: p0 alone against p1 alone in the parts with two heritable traits."""
    rows: list[str] = []
    for part in SINGLE_TRAIT_PARTS:
        totals: dict[str, tuple[int, np.ndarray]] = {}
        for run in recompute_tracking.load_runs(data_root, part):
            if not run.transport:
                continue
            run_count, counts = totals.get(run.condition, (0, np.zeros(len(SPLIT_COUNT_NAMES))))
            totals[run.condition] = (run_count + 1, counts + agreement_counts(run, [[0], [1]]))
        for condition, (run_count, counts) in sorted(totals.items()):
            count = dict(zip(SPLIT_COUNT_NAMES, counts))
            contested: float = count["gate_2_plus"]
            share: str = (
                f"{count['halves_agree_gate_2_plus'] / contested:.1%}" if contested else "n/a"
            )
            rows.append(
                f"| {part} | {condition} | {run_count} | {count['descendants']:.0f} "
                f"| {count['gate_0']:.0f} / {count['gate_1']:.0f} / {contested:.0f} | {share} |"
            )
    return rows


def part_18_pair_rows(data_root: Path) -> list[str]:
    """Markdown rows that compare the Part 18 radius-8 pairs with the trait rule."""
    files: list[str] = sorted(glob.glob(str(data_root / PART_18_PATTERN)))
    if not files:
        raise FileNotFoundError(f"no files match {data_root / PART_18_PATTERN}")
    fingerprint: list[int] = list(range(fl_season.K0 + fl_season.NW))
    totals: dict[str, np.ndarray] = {}
    for file_name in files:
        with open(file_name, "rb") as handle:
            payload = pickle.load(handle)["out"][0]
        for tag, raw in recompute_tracking._tagged_runs(payload):
            arm: str = tag.rsplit("_s", 1)[0]
            counts = totals.setdefault(arm, np.zeros(6))
            for (_, ancestors), (_, descendants) in zip(raw["log"][:-1], raw["log"][1:]):
                if not ancestors or not descendants:
                    continue
                by_traits = lineage_tracking.assign_ancestors(
                    ancestors, descendants, PART_18_BOX_SIZE, fingerprint
                )
                by_space = lineage_tracking.assign_ancestors(
                    ancestors, descendants, PART_18_BOX_SIZE
                )
                pairs = metrics18c.match_bodies(ancestors, descendants, PART_18_BOX_SIZE)
                counts += [
                    len(pairs),
                    sum(1 for i, j, _ in pairs if by_traits[j] == i),
                    sum(ancestors[i][0] for i, _, _ in pairs),
                    sum(o[0] for o in ancestors),
                    int((by_traits != by_space).sum()),
                    len(descendants),
                ]
    return [
        f"| {arm} | {c[0]:.0f} | {c[1] / max(c[0], 1):.1%} | {c[2] / c[3]:.2f} "
        f"| {c[4] / max(c[5], 1):.1%} |"
        for arm, c in sorted(totals.items())
    ]


def separator(header: str) -> str:
    """Markdown separator row with one cell per column of the header."""
    return "|" + "---|" * (header.count("|") - 1)


def main(arguments: Sequence[str]) -> int:
    """Run the checks and write ``tracking_validation.md``."""
    if not 1 <= len(arguments) <= 2:
        print(__doc__, file=sys.stderr)
        return 2
    data_root: Path = Path(arguments[0])
    default_output: Path = Path(__file__).resolve().parent / "results"
    output_dir: Path = Path(arguments[1]) if len(arguments) == 2 else default_output
    output_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["## Split-half agreement (snapshots from step 2000)", "", SPLIT_HEADER]
    lines += [separator(SPLIT_HEADER), *split_half_rows(data_root), ""]
    lines += ["## Single traits in Parts 10, 12 and 13 (snapshots from step 2000)", ""]
    lines += [SINGLE_TRAIT_HEADER, separator(SINGLE_TRAIT_HEADER), *single_trait_rows(data_root)]
    lines += ["", "## Part 18g and 18i: pairs of `metrics18c.match_bodies`", "", PAIR_HEADER]
    lines += [separator(PAIR_HEADER), *part_18_pair_rows(data_root)]
    text: str = "\n".join(lines) + "\n"
    (output_dir / "tracking_validation.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
