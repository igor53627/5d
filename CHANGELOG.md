# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `docs/style-profile.md`: the project profile for "Adapting TigerStyle to Your Codebase",
  revision r5 (blocks in force, threat model, mechanisms, limits, exceptions, open questions).
- `helpers.invariant(condition, message)`: an invariant check that `python -O` does not remove.
- `fl_glider.rossler_rk4_step`: the one copy of the Rossler RK4 step.
- `tests/test_style_limits.py`: fails on a line over 100 columns or a function over 70 lines
  outside the legacy list, and on a legacy entry that is missing or meets the limits.
- `tests/test_transport_mass.py`: mass conservation of `_advect` and `_advect_muscl` in N = 2, 3
  and 4, and four strict `xfail` tests for the known MUSCL positivity defect in N >= 3.
- `tests/test_rossler_rk4.py`: the shared RK4 step equals a frozen copy of the old inline block,
  bit for bit.
- `tests/test_invariant.py`: `invariant` raises, also under `python -O`.
- `uv.lock`, and `ruff==0.17.5` in the `dev` extra with a `[tool.ruff]` configuration.
- `.github/workflows/ci.yml`: format check, lint and tests on CPU for each push and pull request.
- `CHANGELOG.md`, and a "Style" section in `README.md`.
- `experiments/part18/pool18i.py` rebuilds the pooled 18g+18i statistics of Part 18i from the
  committed run records (bootstrap seed 0) and adds a t-test, a Wilcoxon test and `n_pos` for
  each paired difference. `tests/test_pool18i.py` checks it against `full18i_summary.json`.
- `src/lineage_tracking.py`: ancestors identified by heritable traits (`assign_ancestors`),
  division pairs, founder shares and the effective number of lineages.
- `src/price_equation.py`: a mass-weighted Price decomposition whose selection and transmission
  terms sum to the change of the mass-weighted mean.
- `experiments/tracking_fix/`: `recompute_tracking.py` and `validate_tracking.py` recompute the
  lineage-based numbers of Parts 10, 12, 13, 15, 15b, 16 and 17 from the published logs;
  `results/tracking_notes.md` reports them. `validate_tracking.py` reports how often two
  disjoint sets of traits pick the same ancestor, also for descendants with two or more
  ancestors inside the gate.
- `tests/test_price_equation.py`: tracking and Price terms on logs with a known answer.
- Part 20 setup (issue #9), no population run: `experiments/part20/spec20.py` writes
  `archive/specs/b20_pilot.json` (16 paired runs of the Part 13 4D world with and without
  `beta = chi = 0`), `probe20.py` is the CPU probe of one soliton, and
  `results/part20_notes.md` reports the probes and the launch conditions.
  `tests/test_oscillator_ablation.py` checks that the mass field ignores the oscillators
  only when both terms are 0, and that the committed spec is the paired design.
- `src/lineage_tracer.py`: passive tracer fractions that move with the mass of an `fl_eco3`
  run and record the share of the mass of every body that comes from each earlier body. The
  simulated state is bitwise the state of `fl_eco3.run_chunk` (`tests/test_lineage_tracer.py`).
  `experiments/tracking_truth/run_truth.py` runs a Part 13 world with tracers, and
  `analyze_truth.py` compares the ancestor rules with the measured ancestors (issue #13).
- `price_equation.price_terms_from_shares` and `step_share_matrices`: the Price terms from
  given share matrices, which can be fractions. `price_terms` uses them and is unchanged.
- `src/spend_cap.py`: the worst-case spend of a Modal launch (calls x timeout x rate) and a
  check that raises `ValueError` when it exceeds a cap. `tests/test_spend_cap.py` checks the
  arithmetic (issue #14).

### Changed

- `modal_app.spawn_h100` takes `--timeout-seconds` (required; the default 0 is refused) and
  `--cap-dollars` (default 15). It prints the worst-case spend and refuses to start over the
  cap, and it passes the timeout to Modal with `Function.with_options(timeout=...)`, which
  needs Modal 1.4.3 or later. Command lines without `--timeout-seconds` no longer start a
  launch; the launch commands in the notes and run scripts of Parts 18 and 19 predate this.
- The eight engines (`fl_glider`, `fl_eco`, `fl_eco2`, `fl_eco3`, `fl_brain`, `fl_poison`,
  `fl_season`, `fl_pred`) call `fl_glider.rossler_rk4_step` in place of their inline RK4 block.
  The outputs are bitwise identical.
- `.gitignore` ignores `.DS_Store` and `.ruff_cache/`.

### Fixed

- Lineage tracking (issues #7, #11). The analysis took the nearest object in space as the
  ancestor of a body, which differs from the ancestor identified by heritable traits for up to
  85% of descendants (a disagreement rate; the logs hold no true ancestor), and mixed an unweighted
  selection term with a mass-weighted mean. With ancestors identified by heritable traits the
  Price selection terms of Parts 15 and 17 change sign, the negative plateau terms of Parts 12
  and 13 vanish, and division counts, parent-child correlations and lineage counts of Parts
  10-13 change. The notes of Parts 10, 12, 13, 15, 15b, 16 and 17 open with a correction, and
  README states the corrected results. `analyze_eco.selection_diff` and the old analysis
  scripts are unchanged, so the published numbers stay reproducible.
- Part 18i notes and README: `n_pos` was 6/8 and 3/8, the run records give 7/8 and 2/8. The
  verdict is now "suggestive": the pooled bootstrap interval excludes 0, but the t-test gives
  p = 0.057 and the pool was formed after the null 18g run. The 18d and 18h bullets say what did
  not replicate and how many seeds they rest on.
- Part 19d notes and README: 7/12 at `kpred=0.005` is not distinguishable from 2/4 at 0.015
  (Fisher exact p = 1.00) or from 1/4 at 0.01 (exact McNemar p = 0.50 on the four shared
  seeds; Fisher exact p = 0.58 on disjoint seeds), so the text no longer calls it better.
