# migration

Modelling fluid injection into a pre-existing fracture: aperture diffusion,
induced normal stress on an elastic half-space, front migration, and
seismicity-rate response.

## Layout

```
fracinj/                installable package
  types.py              array aliases + the Parameters container
  paths.py              project-anchored data / results / figures dirs
  physics.py            material coefficients, diffusivity, (non)dimensionalisation
  math_utils.py         power-law fitting, crossover functions
  io.py                 read_hdf5 / read_pickle / save_run  (-> RunData)
  detection.py          locate pressure / stress fronts in space-time fields
  analysis.py           Fit + Regime; analyze_front / analyze_p_inj (rigid & soft)
  plotting.py           shared matplotlib helpers + save_figure
  solvers/
    aperture.py         control-volume nonlinear-diffusion solver
    elastic.py          Fredholm (log-kernel) solver for induced normal stress
    similarity.py       shooting-method similarity profiles: solve_similarity(n, bc)
    exact.py            closed-form linear-diffusion similarity solutions
    seismicity_rate.py  Dieterich / Segall-Lu seismicity-rate ODE

scripts/                figure + simulation entry points (run from repo root)
explore/                scratch analyses (not maintained; may lag the package API)
3dec/                   Itasca 3DEC model + export scripts
```

## Pipeline

    3DEC / FVM simulation
        -> HDF5 file  (coordinates/, fields/, parameters/)
        -> fracinj.io.read_hdf5  ->  RunData
        -> fracinj.analysis / fracinj.detection
        -> fracinj.plotting  ->  figures/

## Setup

```
pip install -e .
```

Requires Python >= 3.12.

## Running

```
python scripts/run_simulation.py     # FVM aperture + elastic half-space -> results/
python scripts/fig2_similarity.py     # etc.  (figure scripts use `# %%` cells)
```

Figure scripts read runs from `results/` and write to `figures/` /
`overleaf/` via `fracinj.plotting.save_figure`.

## Known issue

The `n = 3` similarity solves (`solvers.similarity.solve_*_n3`, used by
`analyze_p_inj(..., regime=SOFT)` and `scripts/fig2_similarity.py`) are
very slow / non-converging with recent NumPy/SciPy. This predates the
package reorganisation; the shooting setup needs revisiting.
