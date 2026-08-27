import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from mypackages import front_detection, physics
from mypackages.file_io import RunData, read_halfspace, read_run, sort_fields

importlib.reload(physics)
# %%

result_dir = Path.cwd() / "results" / "3dec" / "runs"

run_0 = read_run(filepath=result_dir / "run-q-1e-06.hdf5")
run_1 = read_run(filepath=result_dir / "run-q-1e-03.hdf5")
run_2 = read_run(filepath=result_dir / "run-q-1e-04.hdf5")
run_3 = read_run(filepath=result_dir / "run-q-1e-05.hdf5")

run_0_fvm = read_halfspace(
    filepath=Path.cwd() / "results" / "fvm-elastic" / "runs" / "run-q-1e-06.hdf5"
)
# %%


@dataclass
class FrontResults:
    x: np.ndarray
    w: np.ndarray
    t: np.ndarray
    zeta: np.ndarray
    theta: np.ndarray


def analyze_stress_front(run: RunData, nondim_fn: Callable, stop: int = -1):
    x_front, idx = front_detection.find_stress_front(run.x_sc, run.sn[:stop])
    # x_front_unique, ind_unique = np.unique(x_front, return_index=True)
    # t_front_unique = run.t[ind_unique]
    t_front = run.t[:idx]

    x_vert, w = sort_fields(run.x_vert, run.w)
    ind = np.searchsorted(x_vert, x_front)
    # ind = np.searchsorted(x_vert, x_front_unique)
    # w_front_unique = w[ind_unique, ind]
    w_front = w[range(idx), ind]

    zeta_front, theta_front = nondim_fn(
        x=x_front, t=t_front, w=w_front, params=run.params
    )
    return FrontResults(
        x=x_front,
        t=t_front,
        w=w_front,
        zeta=zeta_front,
        theta=theta_front,
    )


def analyze_self_similar(run: RunData, theta_at_front: float, nondim_fn: Callable):
    "w_front definition is self-similar rigid case"
    w_threshold = front_detection.self_similar_front_threshold(
        run, theta_at_front, is_pressure=False
    )
    x_front, has_crossing = front_detection.find_field_front(
        run.x_vert, run.w, threshold=w_threshold
    )
    t_front = run.t[has_crossing]

    zeta_front, theta_front = nondim_fn(
        x=x_front, t=t_front, w=w_threshold[has_crossing], params=run.params
    )
    return FrontResults(
        x=x_front,
        t=t_front,
        w=w_threshold[has_crossing],
        zeta=zeta_front,
        theta=theta_front,
    )


def plot_run(
    run,
    nondim_fn,
    ax1,
    ax2,
    *,
    step: int,
    stop: int | None = None,
    zeta_ana=None,
    theta_ana=None,
    label: str = "",
    cmap=plt.get_cmap("viridis"),
) -> cm.ScalarMappable:
    stop = stop or len(run.t)
    x_vert, w = sort_fields(run.x_vert, run.w)
    zeta, theta = nondim_fn(x=x_vert, t=run.t, w=w, params=run.params)
    norm = mcolors.Normalize(vmin=run.t[step], vmax=run.t[stop - 1])
    for i in range(step, stop, step):
        color = cmap(norm(run.t[i]))
        ax1.plot(x_vert, w[i] * 1e3, color=color)
        ax2.plot(zeta[i], theta[i], color=color)
    if zeta_ana is not None:
        ax2.plot(zeta_ana, theta_ana, "k-", label=label)
        ax2.legend()
    ax1.set(xlabel="$x$, [m]", ylabel="$w$, [mm]")
    ax1.set_title(f"$q_0 = {run.params['q_0']:.0e}~\\mathsf{{m^2/s}}$")
    return cm.ScalarMappable(cmap=cmap, norm=norm)


# %% stress peak as front found from 3dec results

run = run_0
nondim_fn = physics.nondimensionalize_rigid
step = 5
stop = len(run.t) // 4
results = analyze_stress_front(run, nondim_fn, stop=stop)

# %%
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(6.4, 4.8 / 2), dpi=200, constrained_layout=True
)
sm = plot_run(
    run,
    nondim_fn,
    ax1,
    ax2,
    step=step,
    stop=stop,
)
slc = slice(step, stop, step)
fig.colorbar(sm, ax=[ax1, ax2], label="$t$, [s]", pad=0.02, aspect=10)
ax2.set(
    xlim=(-0.5, 5),
    xlabel=r"$\zeta = \dfrac{x}{M^{1/5} \, q^{3/5} \, t^{4/5}}$",
    ylabel=r"$\theta = \dfrac{w}{(q^{2} \, t / D)^{1/5}}$",
)
ax1.plot(results.x[slc], results.w[slc] * 1e3, "k.")
ax2.plot(results.zeta[slc], results.theta[slc], "k.")
# figpath = Path.cwd() / "figures" / "similarity" / f"q-{run.params['q_0']:.0e}-3.png"
# plt.savefig(figpath, dpi=150)
plt.show()

# %% self similar w front
run = run_0
nondim_fn = physics.nondimensionalize_rigid
results = analyze_self_similar(run, theta_at_front=0.1, nondim_fn=nondim_fn)


# %%
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(6.4, 4.8 / 2), dpi=200, constrained_layout=True
)
step = 5
stop = len(run.t) // 4
sm = plot_run(
    run,
    nondim_fn,
    ax1,
    ax2,
    step=step,
    stop=stop,
)
slc = slice(step, stop, step)
fig.colorbar(sm, ax=[ax1, ax2], label="$t$, [s]", pad=0.02, aspect=10)
ax2.set(
    xlim=(-0.5, 5),
    xlabel=r"$\zeta = \dfrac{x}{M^{1/5} \, q^{3/5} \, t^{4/5}}$",
    ylabel=r"$\theta = \dfrac{w}{(q^{2} \, t / D)^{1/5}}$",
)
ax1.plot(results.x[slc], results.w[slc] * 1e3, "k.")
ax2.plot(results.zeta[slc], results.theta[slc], "k.")
# figpath = Path.cwd() / "figures" / "similarity" / f"q-{run.params['q_0']:.0e}-3.png"
# plt.savefig(figpath, dpi=150)
plt.show()


# %%

run = run_0

nondim_fn = physics.nondimensionalize_rigid
step = 5
stop = len(run.t) // 4
results = analyze_stress_front(run_0_fvm, nondim_fn, stop)


# %% stress peak as front derived from fvm solution
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(6.4, 4.8 / 2), dpi=200, constrained_layout=True
)
sm = plot_run(
    run,
    nondim_fn,
    ax1,
    ax2,
    step=step,
    stop=stop,
)
slc = slice(step, stop, step)
fig.colorbar(sm, ax=[ax1, ax2], label="$t$, [s]", pad=0.02, aspect=10)
ax2.set(
    xlim=(-0.5, 5),
    xlabel=r"$\zeta = \dfrac{x}{M^{1/5} \, q^{3/5} \, t^{4/5}}$",
    ylabel=r"$\theta = \dfrac{w}{(q^{2} \, t / D)^{1/5}}$",
)
ax1.plot(results.x[slc], results.w[slc] * 1e3, "k.")
ax2.plot(results.zeta[slc], results.theta[slc], "k.")
# figpath = Path.cwd() / "figures" / "similarity" / f"q-{run.params['q_0']:.0e}-3.png"
# plt.savefig(figpath, dpi=150)
plt.show()
