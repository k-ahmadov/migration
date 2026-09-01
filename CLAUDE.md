# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for modelling **fluid injection into a pre-existing fracture**: aperture
diffusion, injection-induced normal stress on an elastic half-space, pressure/stress front
migration, and the seismicity-rate response. Outputs are the figures in `overleaf/` and
`figures/`.

## Environment & commands

- Python **>= 3.12** (uses PEP 695 `type` aliases). The active venv is `migration`
  (`$VIRTUAL_ENV` → `~/.virtualenvs/migration`).
- Install the package (editable): `pip install -e .` — this is required; `scripts/` and
  `explore/` import `fracinj` as an installed package.
- **No test suite and no configured linter.** `pyproject.toml` has a `[tool.ruff.lint]`
  stanza but `ruff` is not installed in the venv. Validate changes by running the relevant
  script or a small `python -c` snippet against a file in `results/`.
- Run the pure-Python simulation: `python scripts/run_simulation.py`
- Figure scripts (`scripts/figN_*.py`) are written as `# %%` interactive cells but also run
  top-to-bottom: `python scripts/fig3_early_late.py`. They read runs from `results/` and
  write via `fracinj.plotting.save_figure`.
- `scripts/run_fvm.py` does `from run_simulation import ...`, so run script files from the
  repo root or from `scripts/`.

## Two-stage architecture

**Stage A — produce a run (HDF5 file).** Two independent producers write the *same* layout:

1. **3DEC** (`3dec/scripts/`): Itasca 3DEC command files (`.dat`) + Python callbacks that
   run *inside* 3DEC (`import itasca`; not runnable here). `inject.dat` is the entry point;
   `store_coordinates.py` writes `coordinates/` + `parameters/` once at init,
   `store_results.py` appends field snapshots on an adaptive schedule.
2. **`scripts/run_simulation.py`**: a pure-Python surrogate — an implicit FVM solve of the
   nonlinear aperture diffusion (`fracinj.solvers.aperture`) followed by a Fredholm
   integral-equation solve for the induced half-space normal stress
   (`fracinj.solvers.elastic`), saved through `fracinj.io.save_run`.

**Stage B — analyse.** `fracinj.io.read_hdf5` → `RunData` → `fracinj.analysis` /
`fracinj.detection` → `scripts/figN_*.py`.

## The HDF5 data contract (central coupling point)

`fracinj/io.py` `FIELD_PATHS` is the single source of truth. A run file has:

- `coordinates/t`, `coordinates/x_vertices` (aperture grid), `coordinates/x_subcontacts`
  (pressure & stress grid) — **the two spatial grids differ in length**.
- `fields/aperture` shaped `(n_t, n_vert)`; `fields/fluid_pressure`, `fields/stress_normal`,
  and optional `fields/stress_shear` shaped `(n_t, n_sc)`. `RunData.tau` is `None` when
  shear stress is absent.
- `parameters/<name>` scalar datasets with `unit` / `description` attrs. On read, only names
  matching `Parameters` fields are kept (e.g. 3DEC's `k_s` is silently dropped).
- `read_halfspace()` reads a legacy variant with older coordinate names.

Spatial arrays from 3DEC are **unsorted**; call `io.sort_fields(x, field)` before any
spatial operation (the detection functions already do).

## Key domain model

- **`Parameters`** (`fracinj/types.py`): one flat dataclass, every field defaulting to
  `0.0`, carrying `unit`/`description` metadata. Holds physics *and* discretisation
  (`Nx_p`, `Nx_sn`, `Nt`, `T`). The injection driver is **mutually exclusive and
  regime-specific**: `q_0` (constant rate) xor `m_q` (linear-ramp slope) xor `DP` (constant
  overpressure). Much of `physics`/`analysis` still assumes `q_0`; ramp support is partial
  (`# TODO` markers in `physics.time_slice`, `analysis._front_amplitude`). A missing driver
  surfaces as a `ValueError` from `physics.dimensionalize`, not a silent 0.

- **Regimes**: `analysis.RIGID` (linear diffusion; front ∝ t^0.5) and `analysis.SOFT`
  (n=3 nonlinear diffusion; front ∝ t^0.8, p_inj ∝ t^0.2). `physics.critical_time` /
  `physics.time_slice` split a run into an early (rigid) and late (soft) window. The `n`
  argument in `solvers.similarity` is the same physics: `n=0` rigid, `n=3` soft.

- **`analysis.Fit`**: one result type for both the `"front"` and `"p_inj"` observables. It
  always carries an empirical power-law fit (`A_emp`, `alpha_emp`, `.empirical()`); passing
  a `regime` to `analyze_front` / `analyze_p_inj` additionally fills the analytical model
  (`A_ana`, `alpha_ana`, `zeta`, `.analytical()`). `front_rigid` / `front_early` /
  `p_inj_late` etc. are thin wrappers over those two entry points.

- **`solvers.similarity.solve_similarity(n, bc, theta_inf)`**: parameterised shooting method
  (RK45 + `scipy.optimize.newton`) using the `u = θ^(n+1)` substitution to regularise the
  front singularity. `solve_dirichlet_n{0,1,3}` / `solve_neumann_n{1,3}` are named wrappers.

- **`fracinj/paths.py`**: all filesystem access is anchored to the repo root via
  `Path(__file__).parents[1]`. Do not reintroduce `Path.cwd()` in scripts. `data/`,
  `results/`, `figures/`, `overleaf/` are gitignored (large / generated).

## Gotchas

- **n=3 similarity solves are very slow / non-converging** with the current NumPy/SciPy.
  This predates the package layout and affects `analysis.analyze_p_inj(regime=SOFT)` and
  `scripts/fig2_similarity.py`. It is *not* a regression to chase when a script hangs there.
- `run_simulation.run_fvm` returns `t` with `Nt` points but `w`/`p` with `Nt-1` rows
  (preserved from the original); analysis slicing on those runs can be one row misaligned.
- **`explore/` and `notebooks/` are scratch and unmaintained.** Many files still import the
  pre-refactor `mypackages` / `mysolvers` names and no longer run. Do not use them as
  reference for the current API.
- The package was recently reorganised from `mypackages` + `mysolvers` into a single
  `fracinj` package; that change and the `scripts/` rewrites are currently uncommitted.
