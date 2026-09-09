from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mypackages import file_io
from mysolvers.seismicity_rate_solver import solve_seismicity_rate

# %% -- config ---
RESULT_DIR = Path.cwd() / "results" / "3dec" / "runs"
RUN_FILE = RESULT_DIR / "run-q-1e-03.hdf5"
MU = 0.6
DTAU_DT_0 = 1e3 / (365 * 24 * 3600)  # background stressing rate [Pa/s]
A = 0.003
SIGMA_EFF = 1e7
T_A = A * SIGMA_EFF / DTAU_DT_0

run = file_io.read_run(RUN_FILE)

# %%
stop = len(run.t) // 2
mu = MU
assert run.tau is not None, "run.tau is not set"
tau_coulomb = run.tau[:stop] - mu * (run.sn[:stop] - run.p[:stop])
x, tau_coulomb = file_io.sort_fields(run.x_sc, tau_coulomb)
dtau_dt = np.gradient(tau_coulomb.T, run.t[:stop], axis=1)

tau_uncoupled = mu * run.p[:stop]
x_uncoupled, tau_uncoupled = file_io.sort_fields(run.x_sc, tau_uncoupled)
dtau_uncoupled_dt = np.gradient(tau_uncoupled.T, run.t[:stop], axis=1)
assert np.allclose(x, x_uncoupled), "spatial coordinates don't match"
result_coupled = solve_seismicity_rate(
    run.t[:stop], dtau_dt=np.asarray(dtau_dt), dtau_dt_0=DTAU_DT_0, t_a=T_A
)

result_uncoupled = solve_seismicity_rate(
    run.t[:stop], dtau_dt=np.asarray(dtau_uncoupled_dt), dtau_dt_0=DTAU_DT_0, t_a=T_A
)

R_difference = result_coupled.y - result_uncoupled.y

# %%
fig = plt.figure(
    figsize=(6.4 / 1.5, 4.8 / 1.5), dpi=150, layout="tight", clear=True, num=1
)
ax = fig.subplots()
im = ax.imshow(
    np.log(R_difference),
    cmap="viridis",
    aspect="auto",
    origin="lower",
    extent=(run.t[:stop].min(), run.t[:stop].max(), x.min(), x.max()),
    # vmax=15,
    # vmin=-15
)
ax.set(
    xlabel="Time [s]",
    ylabel="Distance [m]",
    title="Effect of normal stress on R",
)
fig.colorbar(im, ax=ax, label="log R [ ]")
plt.show()
