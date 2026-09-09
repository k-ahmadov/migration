from pathlib import Path

import matplotlib.pyplot as plt

# import numpy as np
from mypackages import file_io, front_analysis, physics

# %%
result_dir = Path.cwd() / "results" / "3dec" / "runs"
run = file_io.read_run(result_dir / "run-q-1e-04.hdf5")
result = front_analysis.analyze(run, stress_front=True, slc=slice(None, None))

denominator = (run.params.flux**2 * run.t / physics.parameter_a(run.params)) ** (1 / 5)
theta_inf = run.params.w_i / denominator

# %%

fig = plt.figure(
    figsize=(6.4 / 2.5, 4.8 / 2.5), dpi=200, layout="tight", clear=True, num=1
)
ax = fig.subplots()
ax.plot(run.t, theta_inf, ".", color="tab:gray")
ax.set(
    xlabel="$t$ [s]",
    ylabel=r"$\theta_{\infty}$",
    title=rf"$q={run.params.flux:.0e}~\mathsf{{m^2/s}}$",
    # xscale="log",
    # yscale="log",
)
plt.show()
