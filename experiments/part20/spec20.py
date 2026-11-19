"""Part 20 spec: the Part 13 ecology in 4D with and without the oscillator feedback.

The Rossler oscillators act on the mass field through two parameters of the soliton: ``beta``
(growth feedback from local synchrony) and ``chi`` (advection along the phase gradient). With
both at 0 the mass field does not depend on the oscillators (``tests/test_oscillator_ablation.py``).

The spec pairs every run with a twin that differs only in ``beta = chi = 0``. ``run_eco``
prepares the soliton with the parameters of the run, so the twin also forms without feedback.

    python experiments/part20/spec20.py      writes archive/specs/b20_pilot.json

The script starts no run. A launch costs money and needs an explicit yes from the owner.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[2]
SOURCE_SPEC: Path = ROOT / "archive" / "specs" / "eco13m.json"
OUTPUT_SPEC: Path = ROOT / "archive" / "specs" / "b20_pilot.json"
SOURCE_TAGS: dict[str, str] = {"sel": "sel_4D56_s201", "neu": "neu_4D56_s201"}
SEEDS: tuple[int, ...] = (221, 222, 223, 224)
FEEDBACK_KEYS: tuple[str, ...] = ("beta", "chi")
OSCILLATOR_ARMS: tuple[str, ...] = ("osc", "noosc")
LAUNCH_COMMAND: str = (
    "modal run src/modal_app.py::spawn_h100 --kind ecolist "
    "--spec archive/specs/b20_pilot.json --prefix b20p "
    "--timeout-seconds 1200 --cap-dollars 15"
)


def source_runs() -> dict[str, dict]:
    """The Part 13 4D runs that the spec copies, keyed by selection arm."""
    runs = [run for call in json.loads(SOURCE_SPEC.read_text()) for run in call["runs"]]
    by_tag: dict[str, dict] = {run["tag"]: run for run in runs}
    missing = [tag for tag in SOURCE_TAGS.values() if tag not in by_tag]
    if missing:
        raise ValueError(f"{SOURCE_SPEC} has no run tagged {missing}")
    return {arm: by_tag[tag] for arm, tag in SOURCE_TAGS.items()}


def build_spec() -> list[dict]:
    """One Modal call per pair of twins: 2 selection arms x 4 seeds = 8 calls, 16 runs.

    Both twins of a pair run in one container, so they share the GPU type.
    """
    calls: list[dict] = []
    for selection_arm, source in source_runs().items():
        for seed in SEEDS:
            twins: list[dict] = []
            for oscillator_arm in OSCILLATOR_ARMS:
                soliton = dict(source["c"])
                if oscillator_arm == "noosc":
                    soliton.update(dict.fromkeys(FEEDBACK_KEYS, 0.0))
                size_label = f"{source['N']}D{source['L']}"
                tag = f"{selection_arm}_{oscillator_arm}_{size_label}_s{seed}"
                twins.append({**source, "c": soliton, "seed": seed, "tag": tag})
            calls.append({"runs": twins})
    return calls


def main() -> int:
    """Write the spec file and report the number of runs."""
    spec = build_spec()
    run_count = sum(len(call["runs"]) for call in spec)
    OUTPUT_SPEC.write_text(json.dumps(spec, indent=1) + "\n")
    print(f"wrote {OUTPUT_SPEC.relative_to(ROOT)}: {len(spec)} calls, {run_count} runs")
    print("no run started; launch (paid, needs the owner's yes):")
    print(f"  {LAUNCH_COMMAND}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
