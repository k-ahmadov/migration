from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from mypackages import file_io, physics
from mysolvers.seismicity_rate_solver import solve_seismicity_rate

# %% ── Functions ──────────────────────────────────────────────────────────────────


def compute_coulomb_stressing_rate(
    run: file_io.RunData,
    mu: float,
    stop: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute Coulomb stressing rate dτ/dt along the fault.

    Returns x (n_points,) and dtau_dt (n_points, n_times) in [Pa/s].
    """
    assert run.tau is not None, "run.tau is not set"
    tau_coulomb = run.tau[:stop] - mu * (run.sn[:stop] - run.p[:stop])
    x, tau_coulomb = file_io.sort_fields(run.x_sc, tau_coulomb)
    dtau_dt = np.gradient(tau_coulomb.T, run.t[:stop], axis=1)
    return x, np.asarray(dtau_dt, dtype=np.float64)


def plot_seismicity_rate(
    ax,
    log_R: NDArray[np.float64],
    t: NDArray[np.float64],
    x: NDArray[np.float64],
    title: str = "Seismicity rate",
    stop: int | None = None,
) -> plt.cm.ScalarMappable:
    """Plot log seismicity rate as a space-time image."""
    im = ax.imshow(
        log_R,
        cmap="Greys",
        aspect="auto",
        origin="lower",
        extent=(t[:stop].min(), t[:stop].max(), x.min(), x.max()),
        # vmin=vmin or float(np.percentile(log_R, 1)),
        # vmax=vmax or float(np.percentile(log_R, 99)),
    )
    ax.set(xlabel="Time [s]", ylabel="Distance [m]", title=title)
    return im


# %% ── Configuration ─────────────────────────────────────────────────────────────

RESULT_DIR = Path.cwd() / "results" / "3dec" / "runs"

RUNS = {
    "soft": RESULT_DIR / "run-q-1e-03.hdf5",
    "rigid": RESULT_DIR / "run-q-1e-06.hdf5",
}
MU = 0.6  # friction coefficient
A = 0.003  # rate-and-state parameter
SIGMA_EFF = {
    "soft": 10e6,
    "rigid": 10e6,
}  # effective normal stress [Pa]
DTAU_DT_0 = 1e3 / (365 * 24 * 3600)  # background stressing rate [Pa/s]
# T_A = A * SIGMA_EFF / DTAU_DT_0

# %% ── Main ───────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(6.4, 4.8 / 2), dpi=150, layout="constrained")

all_log_R = []

for case, run_file in RUNS.items():
    t_a = A * SIGMA_EFF[case] / DTAU_DT_0
    run = file_io.read_run(run_file)
    stop = len(run.t) // 2
    x, dtau_dt = compute_coulomb_stressing_rate(run, mu=MU, stop=stop)
    result = solve_seismicity_rate(
        t=run.t[:stop],
        dtau_dt=dtau_dt,
        dtau_dt_0=DTAU_DT_0,
        t_a=t_a,
    )
    all_log_R.append((case, run, x, np.log(result.y)))

for ax, (case, run, x, log_R) in zip(axes, all_log_R):
    im = plot_seismicity_rate(
        ax,
        log_R,
        run.t,
        x,
        title=rf"$q={run.params.flux:.0e}~\mathsf{{m^2/s}}$ - {case}",
        stop=len(run.t) // 2,
    )

    fig.colorbar(im, ax=ax, label="log R [ ]")
plt.show()


# %% ── Explore ───────────────────────────────────────────────────────────────────────


log_R = all_log_R[0][3]
x = all_log_R[0][2]
run = all_log_R[0][1]
idx_t = 100

fig = plt.figure(
    figsize=(6.4 / 1.5, 4.8 / 1.5), dpi=200, layout="tight", clear=True, num=1
)
ax = fig.subplots()
ax.plot(x, log_R[:, idx_t], color="tab:gray", label="Seismicity rate $R$")
ax.set(
    xlabel="Distance (m)",
    ylabel=r"Log seismicity rate, $\log R$",
    title=rf"$q = {run.params.flux:.0e}\ \mathsf{{m^2/s}}$ at $t={run.t[idx_t]:.0f}~\mathsf{{s}}$",
)
ax_tw = ax.twinx()
ax_tw.plot(
    run.x_sc, run.p[idx_t] / 1e6, "o", ms=3, color="tab:blue", label="Fluid pressure"
)
ax_tw.set(ylabel="Fluid pressure (MPa)")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax_tw.get_legend_handles_labels()
ax.legend(
    h1 + h2,
    l1 + l2,
    loc="center right",
)
plt.show()


# %%

# TODO: find the seismicity rate front migration

# index 0 -> high injection rate
# index 1 -> low injection rate

run = all_log_R[1][1]
log_R = all_log_R[1][3]
x = all_log_R[1][2]
x_front = x[np.argmax(log_R, axis=0)]
t_front = run.t[: len(run.t) // 2]

A_emp, alpha_emp = physics.fit_front_power_law(t_front, x_front)

# %%

fig = plt.figure(
    figsize=(6.4 / 1.5, 4.8 / 1.4), dpi=150, layout="tight", clear=True, num=1
)
ax = fig.subplots()
ax.plot(t_front, x_front, ".", color="tab:gray", label="Peak of seimicity rate")
# ax.plot(
#     t_front,
#     A_emp * t_front**alpha_emp,
#     "--",
#     color="k",
#     label=rf"Fit: $\alpha={alpha_emp:.2f}$",
# )
ax.set(
    xlabel="Time $t$, (s)",
    ylabel="Distance $x_f$, (m)",
    title="Migration of seismicity rate peak \n"
    rf"$q={run.params.flux:.0e}~\mathsf{{m^2/s}}$",
)
ax.legend()
plt.show()
