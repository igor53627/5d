# Hyperlife style profile

Hyperlife follows TigerStyle as restated in "Adapting TigerStyle to Your Codebase", revision r5:
[kvrail/tigerstyle@r5](https://github.com/kvrail/tigerstyle/blob/r5/TIGER_STYLE.md). The link is
pinned to the tag, so a later revision changes nothing here until this file moves to it. This
file is the project profile that the guide asks for. Where it is silent, the guide holds. The
adoption is light: new code meets the profile, and code that predates it is listed under
Exceptions and is not rewritten.

## Scope

A research repository: `src/` (21 Python modules: the Flow-Lenia engines in JAX, the Modal
launcher, rendering, lineage tracking and the Price decomposition), `experiments/partNN/` (run
and analysis scripts for each part), `archive/` (old searches, frozen) and `tests/`.

Blocks in force:

- Baseline: Limits and errors; Security (threat model, trust boundaries, limits as defense,
  supply chain, verification).
- Profile: Control flow, Assertions, State and numbers, Dependencies and tooling. Performance is
  in force too: GPU time is paid by the hour and each launch has a spend cap.
- Conventions: Functions, Naming, Comments, commits and layout.

Blocks out of scope, because their subject does not exist here: Secrets (the code reads no key,
password or token; the Modal client keeps its credentials outside the repository and CI uses no
secret), Cryptography (the code chooses and calls none), Concurrency (each Modal container owns
its state and writes its own file `<prefix>_<i>.pkl` on the Volume), and the rules for explicit
memory management (Python and JAX manage memory).

## Threat model

- What we protect: the published numbers (a refactor must not change them), the owner's Modal
  spend, and the machine that runs an analysis.
- Pickle: results are `.pkl` files, and `pickle.load` runs code from the file. Load only files
  that this project wrote: the dataset `igor53627/5d` and the project's Modal Volume. Never
  load a pickle from another source.
- Modal launcher (`src/modal_app.py`): it sends this repository's code and a JSON spec to the
  owner's Modal account. Specs are written by the project's own run scripts. The `kind` argument
  selects a module from a fixed table, so no input names a module or a path.
- CI runs with read-only permission and no secret, so a pull request from a fork is safe to run.
- Out of scope: an attacker who can write to the dataset, the Volume or the Modal account.

## Mechanisms

- Assertions: Python `assert` is removed by `python -O`, so new code uses
  `helpers.invariant(condition, message)`, which always raises `AssertionError`. A failed
  invariant stops the process; new code does not catch it. Inside `jax.jit` only static facts
  (shapes, counts) can be checked; numbers are checked outside, in tests and run scripts.
- Errors: a non-finite field or a job that runs out of time is an operating error. `run_eco`
  stops its loop on the first, Modal stops the job on the second, and the partial result that
  the job saved stays on the Volume.
- Recursion: none in the engines. Helpers that convert nested results to JSON (`_py`,
  `jsonable`, `json_ready`) recurse over the project's own data; Python's recursion limit bounds
  them.
- Numbers: lattice fields are float32 and spectra are complex64; JAX 64-bit mode is off. Total
  mass is the central invariant: `_advect` and `_advect_muscl` are in flux form and conserve it,
  and the heritable field `Q = p*A` moves with the same velocities. Tests sum in float64. A
  refactor of an engine must be bitwise neutral (`tests/test_rossler_rk4.py` shows the pattern).
- One copy: the Rossler RK4 step is `fl_glider.rossler_rk4_step`; the eight engines call it.
- Tests: each new test opens with one line on its goal and its method.

## Limits

| Limit           | Value                  | Scope    | Checked                  | On excess       |
|-----------------|------------------------|----------|--------------------------|-----------------|
| Function length | 70 lines               | Function | `test_style_limits.py`   | Fail the build  |
| Line length     | 100 columns            | Line     | `test_style_limits.py`   | Fail the build  |
| Modal job time  | 7,200 s                | Call     | Modal, `timeout=`        | Job is stopped  |
| Spend           | $15 by default         | Launch   | `spawn_h100` only        | Refuse to start |
| Steps           | 20,000 (Parts 19b-19d) | Run      | Loop count; smoke        | Loop ends       |
| Lattice         | N=4, L=48 (Part 19d)   | Run      | `run19d.py smoke`        | Smoke fails     |
| Flow speed      | 0.45/sqrt(N) cell/step | Cell     | Engine step              | Scale down      |
| Oscillator z    | -50 to 80              | Cell     | `rossler_rk4_step`       | Clip            |
| Transport mass  | 1e-5 relative error    | Step     | `test_transport_mass.py` | Fail the build  |
| Non-finite mass | None allowed           | Frame    | `run_eco`                | Stop the loop   |
| CI job time     | 15 min                 | Job      | GitHub Actions           | Cancel the job  |

`gpu_job_h100` and `gpu_job_b300` have 14,400 s. "Smoke" is `run19d.py smoke`, which runs before
a launch. `spawn_h100` replaces the job time of `gpu_job_h100` with `--timeout-seconds`, which
has no usable default. Before it starts a container it computes the worst-case spend, calls x
timeout x $3.95 per H100-hour (`src/spend_cap.py`, checked by `tests/test_spend_cap.py`), and
refuses to start when that exceeds `--cap-dollars`. The figure is the GPU charge without a
preemption: it does not count the CPU and memory charge, and Modal restarts a preempted
Function on the same input, which adds the time before the preemption. The other entry points
have no spend limit; see Open questions.

## Dependencies and tooling

- Runtime: `numpy`, `jax`, `scipy`, `matplotlib`, `imageio`, `imageio-ffmpeg`. Extras: `dev`
  (`pytest`, `ruff==0.17.5`) and `modal`.
- `uv.lock` pins every package by hash. CI installs with `uv sync --locked --extra dev`.
- The GPU image in `src/modal_app.py` pins `jax[cuda12]==0.12.0.1`. It copies source files one
  by one, so a new module that an engine imports must be added there.
- CI (`.github/workflows/ci.yml`) runs on push and pull request, on CPU, with actions pinned by
  commit hash: `ruff format --check`, `ruff check`, `pytest`.
- Formatter and linter: `ruff`, line length 100. The length test uses only the standard library.

## Exceptions

The legacy list has one copy: `extend-exclude` under `[tool.ruff]` in `pyproject.toml`, 113
files, each named. `tests/test_style_limits.py` reads it and fails when a listed file is missing
or meets the limits, so the list can only shrink. Owner of every entry: igor53627.

- E1, 8 engines: `fl_glider.py`, `fl_eco.py`, `fl_eco2.py`, `fl_eco3.py`, `fl_brain.py`,
  `fl_poison.py`, `fl_season.py`, `fl_pred.py` in `src/`. Reason: they produced the published
  numbers of Parts 5-19, and a rewrite risks the numerics.
- E2, 7 other modules in `src/`. Reason: written before the profile.
- E3, 39 files in `archive/`. Reason: a frozen record of old searches.
- E4, 52 scripts in `experiments/`. Reason: each is the record of a published run.
- E5, 7 files in `tests/`. Reason: written before the profile.
- E6, the `_path.py` files (the root bootstrap and its copies) and `analyze19b.py`,
  `analyze19c.py`, `analyze19d.py` in `experiments/part19/`. They meet the length limits, which
  the test enforces, but predate the formatter. They are excluded from `ruff format`; the
  three `analyze` files also from `ruff check`. Markdown files are excluded from `ruff format`
  too.

E1-E5 waive the conventions (length, naming, layout, English docstrings) and the `assert`
mechanism for the listed files. They do not waive a baseline requirement.

## Open questions

- License: the repository has no LICENSE file. The choice belongs to the owner.
- `_advect_muscl` is not positivity-preserving for N >= 3 at the speed cap, and the clamp
  `jnp.maximum(An, 0.0)` in the engine step then creates mass. Four strict `xfail` tests in
  `tests/test_transport_mass.py` record it (issue #10).
- The spend cap is enforced for `spawn_h100` only (issue #14). The other entry points of
  `src/modal_app.py` have no cap and keep the fixed job time of their function: `smoke`, `bench`,
  `res`, `alife`, `spawn`, `spawn_alife`, `spawn_glider`, `spawn_glider_lean`, `spawn_kind`,
  `spawn_cheap` and `spawn_big` (7,200 s for each call), `spawn_b300` and `spawn_gpu` (14,400 s
  for each call). For these the cap is still kept by hand. The check counts the GPU charge of
  one launch at the H100 rate. It does not count CPU and memory charges, the start and idle
  seconds of a container, or earlier launches.
- Legacy files contain 10 `except Exception` clauses (`archive/bench.py`, `glider_rep.py`,
  `perf_eco.py`, `perf11.py`, `analyze19d.py`); two of them pass in silence. "Handle every
  error" is a baseline requirement, so these need a fix or a tracked issue.
- Legacy files check invariants with `assert` (for example `n % m == 0` in `run_chunk`, and the
  `smoke` functions). No command in the repository passes `-O`, but nothing forbids it.
- The GPU image does not install from `uv.lock`: `numpy` and `scipy` are unpinned there, and
  the lockfile resolves `jax` 0.12.1, not 0.12.0.1.
- Engine tests run in 2D. Only the transport tests cover N = 3 and N = 4.
- Results are pickles. A format that cannot run code (`.npz`, JSON) would remove the risk.
