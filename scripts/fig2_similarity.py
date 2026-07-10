# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import colors as mcolors

from mypackages import file_io, physics
from mysolvers import similarity_solutions
from mysolvers.exact_solutions import solve_linear_diffusion_const_flux

# %%
# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

result_dir = Path.cwd() / "results" / "3dec" / "runs-wi-1e-05"

run_rigid = file_io.read_run(filepath=result_dir / "run-q-5e-09.hdf5")
run_soft = file_io.read_run(filepath=result_dir / "run-q-5e-05.hdf5")

# Analytical / semi-analytical reference curves
zeta_n0_ana = np.linspace(0, 5, 100)
theta_n0_ana = solve_linear_diffusion_const_flux(zeta_n0_ana)

x_vert, w = file_io.sort_fields(run_soft.x_vert, run_soft.w)
_, theta_arr = physics.nondimensionalize_soft(
    x=x_vert, t=run_soft.t[-1], w=w[-1], params=run_soft.params
)
zeta_n3_semi, theta_n3_semi = similarity_solutions.solve_neumann_n3(theta_arr[-1])


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
    x_vert, w = file_io.sort_fields(run.x_vert, run.w)
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
    ax1.set_title(f"$q_0 = {run.params.flux:.0e}~\\mathsf{{m^2/s}}$")
    return cm.ScalarMappable(cmap=cmap, norm=norm)


# %% ---------------------------------------------------------------------------
# Figure 1: run_0 and run_1 (2x2)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(
    2, 2, figsize=(6.4, 4.8 / 1.5), constrained_layout=True, dpi=200
)

sm0 = plot_run(
    run_rigid,
    physics.nondimensionalize_rigid,
    axes[0, 0],
    axes[0, 1],
    step=10,
    stop=len(run_rigid.t) // 4,
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
    run_soft,
    physics.nondimensionalize_soft,
    axes[1, 0],
    axes[1, 1],
    step=10,
    zeta_ana=zeta_n3_semi,
    theta_ana=theta_n3_semi,
    label="Semi-analytical",
)
axes[1, 1].set(
    xlim=(0, 5),
    xlabel=r"$\zeta = \dfrac{x}{M^{1/5} \, q^{3/5} \, t^{4/5}}$",
    ylabel=r"$\theta = \dfrac{w}{(q^{2} \, t / D)^{1/5}}$",
)

axes[0, 0].annotate("(a)", (-0.15, 1.1), xycoords="axes fraction")
axes[1, 0].annotate("(b)", (-0.15, 1.1), xycoords="axes fraction")

fig.colorbar(sm0, ax=axes[0, :], label="$t$, [s]", pad=0.02, aspect=10)
fig.colorbar(sm1, ax=axes[1, :], label="$t$, [s]", pad=0.02, aspect=10)
# plt.savefig(Path.cwd() / "figures" / "paper" / "fig-similarity.png", dpi=200)
plt.savefig(Path.cwd() / "overleaf" / "figures_main" / "Fig2.eps")
# plt.show()

# plt.show()
