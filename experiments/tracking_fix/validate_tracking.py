"""Check the trait-fingerprint ancestor rule against independent evidence.

Usage: python validate_tracking.py DATA_ROOT [OUTPUT_DIR]

Two checks, both on logs with a micro-brain (31 or more heritable weights):

* Split-half: the weights are split at random into two disjoint halves. Each half picks the
  ancestor of every descendant on its own. Agreement of the two halves means that the
  ancestor is identified by the traits and not by chance.
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
SPLIT_HEADER: str = (
    "| part | condition | runs | descendants | two halves agree | nearest in space agrees "
    "| (p0, p1) agrees |"
)
PAIR_HEADER: str = (
    "| arm | pairs within 8 cells | agree with trait rule | share of old mass in pairs "
    "| nearest in space differs |"
)


def agreement_counts(run: recompute_tracking.Run, halves: Sequence[list[int]]) -> np.ndarray:
    """Counts for one run: descendants, halves agree, nearest agrees, (p0, p1) agrees."""
    config = recompute_tracking.PARTS[run.part]
    rules: list[list[int] | None] = [
        list(range(config.fingerprint_count)),
        halves[0],
        halves[1],
        None,
        [0, 1],
    ]
    counts: np.ndarray = np.zeros(4)
    for (time, ancestors), (_, descendants) in zip(run.log[:-1], run.log[1:]):
        if time < SPLIT_TIME_MIN or len(ancestors) < 3 or not descendants:
            continue
        full, first_half, second_half, by_space, by_neutral = (
            lineage_tracking.assign_ancestors(
                ancestors, descendants, run.box_size, indices, config.gate_radius
            )
            for indices in rules
        )
        counts += [
            len(descendants),
            int((first_half == second_half).sum()),
            int((by_space == full).sum()),
            int((by_neutral == full).sum()),
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
            run_count, counts = totals.get(run.condition, (0, np.zeros(4)))
            totals[run.condition] = (run_count + 1, counts + agreement_counts(run, halves))
        for condition, (run_count, counts) in sorted(totals.items()):
            share = counts[1:] / max(counts[0], 1)
            rows.append(
                f"| {part} | {condition} | {run_count} | {counts[0]:.0f} | {share[0]:.1%} "
                f"| {share[1]:.1%} | {share[2]:.1%} |"
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


def main(arguments: Sequence[str]) -> int:
    """Run both checks and write ``tracking_validation.md``."""
    if not 1 <= len(arguments) <= 2:
        print(__doc__, file=sys.stderr)
        return 2
    data_root: Path = Path(arguments[0])
    default_output: Path = Path(__file__).resolve().parent / "results"
    output_dir: Path = Path(arguments[1]) if len(arguments) == 2 else default_output
    output_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["## Split-half agreement (snapshots from step 2000)", "", SPLIT_HEADER]
    lines += ["|" + "---|" * 7, *split_half_rows(data_root), ""]
    lines += ["## Part 18g and 18i: pairs of `metrics18c.match_bodies`", "", PAIR_HEADER]
    lines += ["|" + "---|" * 5, *part_18_pair_rows(data_root)]
    text: str = "\n".join(lines) + "\n"
    (output_dir / "tracking_validation.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
