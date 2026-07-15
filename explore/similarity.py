# %%
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import colors as mcolors

from mypackages import front_analysis, front_detection, physics
from mypackages.file_io import RunData, read_run, sort_fields
from mysolvers import similarity_solutions
from mysolvers.exact_solutions import solve_linear_diffusion_const_flux

# %%
# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

result_dir = Path.cwd() / "results" / "3dec" / "runs"

run_0 = read_run(filepath=result_dir / "run-q-1e-06.hdf5")
run_1 = read_run(filepath=result_dir / "run-q-1e-03.hdf5")
run_2 = read_run(filepath=result_dir / "run-q-1e-04.hdf5")
run_3 = read_run(filepath=result_dir / "run-q-1e-05.hdf5")

# Analytical / semi-analytical reference curves
zeta_n0_ana = np.linspace(0, 5, 100)
theta_n0_ana = solve_linear_diffusion_const_flux(zeta_n0_ana)

# Semi-analytical for n=3: evaluate at a representative time snapshot
i_ref = len(run_1.t) // 8
x_vert_1, w_1 = sort_fields(run_1.x_vert, run_1.w)
_, theta_ref = physics.nondimensionalize_soft(
    x=x_vert_1, t=run_1.t[i_ref], w=w_1[i_ref], params=run_1.params
)
zeta_n3_semi, theta_n3_semi = similarity_solutions.solve_neumann_n3(theta_ref[-1])

# Semi-analytical for n=3: evaluate at a representative time snapshot of run_2
i_ref_2 = len(run_1.t)
x_vert_2, w_2 = sort_fields(run_2.x_vert, run_2.w)
_, theta_ref_2 = physics.nondimensionalize_soft(
    x=x_vert_2, t=run_2.t[i_ref_2], w=w_2[i_ref_2], params=run_2.params
)
zeta_n3_semi_2, theta_n3_semi_2 = similarity_solutions.solve_neumann_n3(theta_ref_2[-1])

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


# %%
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


# %% ---------------------------------------------------------------------------
# Figure 1: run_0 and run_1 (2x2)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(
    2, 2, figsize=(6.4, 4.8 / 1.5), constrained_layout=True, dpi=200
)

sm0 = plot_run(
    run_0,
    physics.nondimensionalize_rigid,
    axes[0, 0],
    axes[0, 1],
    step=100,
    zeta_ana=zeta_n0_ana,
    theta_ana=theta_n0_ana,
    label="Analytical",
)
axes[0, 1].set(
    xlim=(0, 5),
    xlabel=r"$\zeta = \dfrac{x}{(D \, t)^{1/2}}$",
    ylabel=r"$\theta = \dfrac{w - w_i}{q(t/D)^{1/2}}$",
)

sm1 = plot_run(
    run_1,
    physics.nondimensionalize_soft,
    axes[1, 0],
    axes[1, 1],
    step=5,
    stop=len(run_1.t) // 2,
    zeta_ana=zeta_n3_semi,
    theta_ana=theta_n3_semi,
    label="Semi-analytical",
)
axes[1, 1].set(
    xlim=(0, 5),
    xlabel=r"$\zeta = \dfrac{x}{M^{1/5} \, q^{3/5} \, t^{4/5}}$",
    ylabel=r"$\theta = \dfrac{w}{(q^{2} \, t / D)^{1/5}}$",
)

fig.colorbar(sm0, ax=axes[0, :], label="$t$, [s]", pad=0.02, aspect=10)
fig.colorbar(sm1, ax=axes[1, :], label="$t$, [s]", pad=0.02, aspect=10)
# plt.savefig(Path.cwd()/ 'figures'/ 'similarity'/ 'rigid-soft.png', dpi=200)
plt.show()

# %% ---------------------------------------------------------------------------
# Figure 3: run_3 q_0=1e-05
# ---------------------------------------------------------------------------

fig3, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(6.4, 4.8 / 2), dpi=200, constrained_layout=True
)

sm3 = plot_run(
    run_3,
    physics.nondimensionalize_rigid,
    ax1,
    ax2,
    step=5,
    zeta_ana=zeta_n0_ana,
    theta_ana=theta_n0_ana,
    label="Analytical",
)
ax2.set(
    xlim=(0, 5),
    xlabel=r"$\zeta = \dfrac{x}{(D \, t)^{1/2}}$",
    ylabel=r"$\theta = \dfrac{w - w_i}{q(t/D)^{1/2}}$",
)
ax2.set(xlim=(-0.5, 10))
fig3.colorbar(sm3, ax=[ax1, ax2], label="$t$, [s]", pad=0.02, aspect=10)
# plt.savefig(
#     Path.cwd() / "figures" / "similarity" / f"q-{run_3.params['q_0']:.0e}.png", dpi=200
# )
plt.show()


# %% ---------------------------------------------------------------------------
# Figure 2: run_2 q_0=1e-04 soft
# ---------------------------------------------------------------------------

fig2, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(6.4, 4.8 / 2), dpi=200, constrained_layout=True
)
sm2 = plot_run(
    run_2,
    physics.nondimensionalize_soft,
    ax1,
    ax2,
    step=5,
    zeta_ana=zeta_n3_semi_2,
    theta_ana=theta_n3_semi_2,
    label="Semi-analytical",
)
ax2.set(
    xlim=(0, 5),
    xlabel=r"$\zeta = \dfrac{x}{M^{1/5} \, q^{3/5} \, t^{4/5}}$",
    ylabel=r"$\theta = \dfrac{w}{(q^{2} \, t / D)^{1/5}}$",
    title="soft behavior",
)
ax2.set(xlim=(-0.5, 10))
fig2.colorbar(sm2, ax=[ax1, ax2], label="$t$, [s]", pad=0.02, aspect=10)
# plt.savefig(
#     Path.cwd() / "figures" / "similarity" / f"q-{run_2.params['q_0']:.0e}-soft.png",
#     dpi=200,
# )
plt.show()

# %% ---------------------------------------------------------------------------
# Figure 2: run_2 q_0=1e-04 rigid
# ---------------------------------------------------------------------------

fig2, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(6.4, 4.8 / 2), dpi=200, constrained_layout=True
)

sm2 = plot_run(
    run_2,
    physics.nondimensionalize_rigid,
    ax1,
    ax2,
    step=1,
    zeta_ana=zeta_n0_ana,
    theta_ana=theta_n0_ana,
    label="Analytical",
)
ax2.set(
    xlim=(-0.5, 10),
    xlabel=r"$\zeta = \dfrac{x}{(D \, t)^{1/2}}$",
    ylabel=r"$\theta = \dfrac{w - w_i}{q(t/D)^{1/2}}$",
    title="rigid behavior",
)
fig2.colorbar(sm2, ax=[ax1, ax2], label="$t$, [s]", pad=0.02, aspect=10)
# plt.savefig(
#     Path.cwd() / "figures" / "similarity" / f"q-{run_2.params['q_0']:.0e}-rigid.png",
#     dpi=200,
# )
plt.show()
# %%


def nondim_powerlaw(
    *,
    x,
    t,
    w,
    params,
    alpha_x: float,
    alpha_w: float,
    wi: float = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Power-law nondimensionalization with custom exponents:

        zeta = x / t^alpha_x
        theta = w / t^alpha_w

    Used to empirically find the best-fit scaling exponents.
    For the rigid case the theoretical values are alpha_x=0.5, alpha_w=0.5;
    for the soft case alpha_x=0.8, alpha_w=0.2.
    """
    w_ = np.asarray(w)
    if w_.ndim == 1:
        return x / t**alpha_x, (w - wi) / t**alpha_w
    elif w_.ndim == 2:
        return x / t[:, None] ** alpha_x, (w - wi) / t[:, None] ** alpha_w
    else:
        raise ValueError("w should either be 1 or 2 dimensional")


@dataclass
class NondimFrontResults:
    x: np.ndarray
    w: np.ndarray
    t: np.ndarray
    zeta: np.ndarray
    theta: np.ndarray


def analyze_nondim(run, alpha_x, alpha_w, pct_increase, wi=0.0):
    w_front = front_detection.constant_aperture_threshold(run, pct_increase)
    x_front, has_crossing = front_detection.find_field_front(run.x_vert, run.w, w_front)
    t_front = run.t[has_crossing]
    zeta_front, theta_front = nondim_powerlaw(
        x=x_front,
        t=t_front,
        w=w_front[has_crossing],
        params=None,
        alpha_x=alpha_x,
        alpha_w=alpha_w,
        wi=wi,
    )
    return NondimFrontResults(
        x=x_front,
        t=t_front,
        w=w_front[has_crossing],
        zeta=zeta_front,
        theta=theta_front,
    )


alpha_x = 0.67
alpha_w = 0.0
pct_increase = 0.1
nondim_front_res = analyze_nondim(
    run=run_1, alpha_x=alpha_x, alpha_w=alpha_w, pct_increase=pct_increase
)

# %% ---------------------------------------------------------------------------
# Figure (exploratory): find best power law similarity exponents run_1 q_0=1e-03
# ---------------------------------------------------------------------------

fig2, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(6.4, 4.8 / 2), dpi=200, constrained_layout=True
)
sm2 = plot_run(
    run_1,
    partial(nondim_powerlaw, alpha_x=alpha_x, alpha_w=alpha_w),
    ax1,
    ax2,
    step=5,
    stop=len(run_1.t) // 2,
)
ax2.set(
    xlim=(-0.5, 15),
    title=rf"$x \propto t^{{{alpha_x}}}$, $w \propto t^{{{alpha_w}}}$",
    xlabel=rf"$\zeta=\dfrac{{x}}{{t^{{{alpha_x}}}}}$",
    ylabel=rf"$\theta=\dfrac{{w}}{{t^{{{alpha_w}}}}}$",
)
ax1.plot(nondim_front_res.x, nondim_front_res.w * 1e3, "k-")
ax2.plot(nondim_front_res.zeta, nondim_front_res.theta, "k.")
fig2.colorbar(sm2, ax=[ax1, ax2], label="$t$, [s]", pad=0.02, aspect=10)
figpath = Path.cwd() / "figures" / "similarity" / f"q-{run_1.params['q_0']:.0e}-3.png"
# plt.savefig(figpath, dpi=150)
plt.show()


# %%
alpha_x = 0.66
alpha_w = 0.33
pct_increase = 0.1
nondim_front_res = analyze_nondim(
    run=run_2, alpha_x=alpha_x, alpha_w=alpha_w, pct_increase=pct_increase
)

# %% ---------------------------------------------------------------------------
# Figure (exploratory): find best power law similarity exponents run_2 q_0=1e-04
# ---------------------------------------------------------------------------

fig2, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(6.4, 4.8 / 2), dpi=200, constrained_layout=True
)
sm2 = plot_run(
    run_2,
    partial(nondim_powerlaw, alpha_x=alpha_x, alpha_w=alpha_w),
    ax1,
    ax2,
    step=5,
    stop=len(run_2.t),
)
ax2.set(
    xlim=(-0.5, 15),
    title=rf"$x \propto t^{{{alpha_x}}}$, $w \propto t^{{{alpha_w}}}$",
    xlabel=rf"$\zeta=\dfrac{{x}}{{t^{{{alpha_x}}}}}$",
    ylabel=rf"$\theta=\dfrac{{w}}{{t^{{{alpha_w}}}}}$",
)
ax1.plot(nondim_front_res.x, nondim_front_res.w * 1e3, "k-")
ax2.plot(nondim_front_res.zeta, nondim_front_res.theta, "k.")
fig2.colorbar(sm2, ax=[ax1, ax2], label="$t$, [s]", pad=0.02, aspect=10)
figpath = Path.cwd() / "figures" / "similarity" / f"q-{run_2.params['q_0']:.0e}-2.png"
# plt.savefig(figpath, dpi=150)
plt.show()

# %%
alpha_x = 0.5
alpha_w = 0.5
pct_increase = 0.01
nondim_front_res = analyze_nondim(
    run=run_0, alpha_x=alpha_x, alpha_w=alpha_w, pct_increase=pct_increase, wi=run_0.params["w_i"]
)


# %% ---------------------------------------------------------------------------
# Figure (exploratory): constant aperture defintion for front rigid case run_0 q_0=1e-06 rigid
# ---------------------------------------------------------------------------

fig2, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(6.4, 4.8 / 2), dpi=200, constrained_layout=True
)
sm2 = plot_run(
    run_0,
    partial(nondim_powerlaw, alpha_x=alpha_x, alpha_w=alpha_w),
    ax1,
    ax2,
    step=5,
    stop=len(run_0.t),
)
ax2.set(
    xlim=(-0.5, 5),
    title=rf"$x \propto t^{{{alpha_x}}}$, $w \propto t^{{{alpha_w}}}$",
    xlabel=rf"$\zeta=\dfrac{{x}}{{t^{{{alpha_x}}}}}$",
    ylabel=rf"$\theta=\dfrac{{w}}{{t^{{{alpha_w}}}}}$",
)
ax1.plot(nondim_front_res.x, nondim_front_res.w * 1e3, "k-")
ax2.plot(nondim_front_res.zeta, nondim_front_res.theta, "k.")
fig2.colorbar(sm2, ax=[ax1, ax2], label="$t$, [s]", pad=0.02, aspect=10)
figpath = Path.cwd() / "figures" / "similarity" / f"q-{run_0.params['q_0']:.0e}-2.png"
# plt.savefig(figpath, dpi=150)
plt.show()

#


# %% ---------------------------------------------------------------------------
# Figure (exploratory): stress peak defition for front soft case run_1 q_0=1e-03
# ---------------------------------------------------------------------------


def analyze_stress_front(run: RunData, nondim_fn: Callable):
    x_front, idx = front_detection.find_stress_front(run.x_sc, run.sn)
    t_front = run.t[:idx]
    idx_x_front_vert = np.isclose(run.x_vert, x_front)
    x_front_vert = run.x_vert[idx_x_front_vert]
    w_front = run.w[idx_x_front_vert]
    zeta_front, theta_front = nondim_fn(
        x=x_front_vert, t=run.t, w=w_front, params=run.params
    )
    return NondimFrontResults(
        x=x_front,
        t=t_front,
        w=w_front,
        zeta=zeta_front,
        theta=theta_front,
    )
