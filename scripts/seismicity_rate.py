from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from fracinj import io, paths
from fracinj.math_utils import fit_power_law
from fracinj.solvers.seismicity_rate import solve_seismicity_rate

# %% -- Functions -------------------------------------------------------


def compute_coulomb_stressing_rate(
    run: io.RunData, mu: float, stop: int | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Coulomb stressing rate dtau/dt along the fault; returns (x, dtau_dt[n_x, n_t]) [Pa/s]."""
    assert run.tau is not None, "run.tau is not set"
    tau_coulomb = run.tau[:stop] - mu * (run.sn[:stop] - run.p[:stop])
    x, tau_coulomb = io.sort_fields(run.x_sc, tau_coulomb)
    dtau_dt = np.gradient(tau_coulomb.T, run.t[:stop], axis=1)
    return x, np.asarray(dtau_dt, dtype=np.float64)


def plot_seismicity_rate(ax, log_R, t, x, title="Seismicity rate", stop=None):
    """Plot log seismicity rate as a space-time image."""
    im = ax.imshow(
        log_R,
        cmap="Greys",
        aspect="auto",
        origin="lower",
        extent=(t[:stop].min(), t[:stop].max(), x.min(), x.max()),
    )
    ax.set(xlabel="Time [s]", ylabel="Distance [m]", title=title)
    return im


# %% -- Configuration --------------------------------------------------

RUN_FILES = {
    "soft": paths.results_dir("3dec", "runs") / "run-q-1e-03.hdf5",
    "rigid": paths.results_dir("3dec", "runs") / "run-q-1e-06.hdf5",
}
MU = 0.6  # friction coefficient
A = 0.003  # rate-and-state parameter
SIGMA_EFF = {"soft": 10e6, "rigid": 10e6}  # effective normal stress [Pa]
DTAU_DT_0 = 1e3 / (365 * 24 * 3600)  # background stressing rate [Pa/s]


class Case(NamedTuple):
    name: str
    run: io.RunData
    x: NDArray[np.float64]
    log_R: NDArray[np.float64]


# %% -- Solve ---------------------------------------------------------

cases: list[Case] = []
for name, run_file in RUN_FILES.items():
    run = io.read_hdf5(run_file)
    stop = len(run.t) // 2
    t_a = A * SIGMA_EFF[name] / DTAU_DT_0
    x, dtau_dt = compute_coulomb_stressing_rate(run, mu=MU, stop=stop)
    result = solve_seismicity_rate(
        t=run.t[:stop], dtau_dt=dtau_dt, dtau_dt_0=DTAU_DT_0, t_a=t_a
    )
    cases.append(Case(name, run, x, np.log(result.y)))

# %% -- Space-time images --------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(6.4, 4.8 / 2), dpi=150, layout="constrained")
for ax, case in zip(axes, cases):
    stop = len(case.run.t) // 2
    im = plot_seismicity_rate(
        ax,
        case.log_R,
        case.run.t,
        case.x,
        title=rf"$q={case.run.params.q_0:.0e}~\mathsf{{m^2/s}}$ - {case.name}",
        stop=stop,
    )
    fig.colorbar(im, ax=ax, label="log R [ ]")
plt.show()

# %% -- Profile at a single time ------------------------------------

soft = cases[0]
idx_t = 100

fig = plt.figure(figsize=(6.4 / 1.5, 4.8 / 1.5), dpi=200, layout="tight", clear=True, num=1)
ax = fig.subplots()
ax.plot(soft.x, soft.log_R[:, idx_t], color="tab:gray", label="Seismicity rate $R$")
ax.set(
    xlabel="Distance (m)",
    ylabel=r"Log seismicity rate, $\log R$",
    title=rf"$q = {soft.run.params.q_0:.0e}\ \mathsf{{m^2/s}}$ at $t={soft.run.t[idx_t]:.0f}~\mathsf{{s}}$",
)
ax_tw = ax.twinx()
ax_tw.plot(soft.run.x_sc, soft.run.p[idx_t] / 1e6, "o", ms=3, color="tab:blue", label="Fluid pressure")
ax_tw.set(ylabel="Fluid pressure (MPa)")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax_tw.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="center right")
plt.show()

# %% -- Migration of the seismicity-rate peak ----------------------

rigid = cases[1]
x_front = rigid.x[np.argmax(rigid.log_R, axis=0)]
t_front = rigid.run.t[: len(rigid.run.t) // 2]
A_emp, alpha_emp = fit_power_law(t_front, x_front)

fig = plt.figure(figsize=(6.4 / 1.5, 4.8 / 1.4), dpi=150, layout="tight", clear=True, num=1)
ax = fig.subplots()
ax.plot(t_front, x_front, ".", color="tab:gray", label="Peak of seismicity rate")
ax.set(
    xlabel="Time $t$, (s)",
    ylabel="Distance $x_f$, (m)",
    title="Migration of seismicity rate peak \n"
    rf"$q={rigid.run.params.q_0:.0e}~\mathsf{{m^2/s}}$",
)
ax.legend()
plt.show()
