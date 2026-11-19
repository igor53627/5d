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

### Changed

- The eight engines (`fl_glider`, `fl_eco`, `fl_eco2`, `fl_eco3`, `fl_brain`, `fl_poison`,
  `fl_season`, `fl_pred`) call `fl_glider.rossler_rk4_step` in place of their inline RK4 block.
  The outputs are bitwise identical.
- `.gitignore` ignores `.DS_Store` and `.ruff_cache/`.
