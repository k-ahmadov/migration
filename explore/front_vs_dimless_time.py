from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mypackages import file_io, front_analysis, physics, types

# %%


def calc_theta_inf(params: types.Parameters) -> float:
    _, t_char = physics.dimensionalize(params)
    denominator = (params.flux**2 * t_char / physics.parameter_a(params)) ** (1 / 5)
    return params.w_i / denominator


# %%
result_dir = Path.cwd() / "results" / "3dec" / "runs-wi-1e-05"

run_0 = file_io.read_run(result_dir / "run-q-5e-09.hdf5")
result_0 = front_analysis.analyze(run_0, stress_front=True, slc=slice(None, None))

run_1 = file_io.read_run(result_dir / "run-q-5e-07.hdf5")
result_1 = front_analysis.analyze(run_1, stress_front=True, slc=slice(None, None))

run_2 = file_io.read_run(result_dir / "run-q-5e-05.hdf5")
result_2 = front_analysis.analyze(run_2, stress_front=True, slc=slice(None, None))

# %%

print(calc_theta_inf(run_0.params))

print(
    run_1.params.w_i
    / (
        run_1.params.flux
        * run_1.params.L
        / physics.parameter_a(run_1.params)
    ) ** (1 / 4)
)

# %%
