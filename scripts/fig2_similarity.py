# %%
import matplotlib.pyplot as plt
import numpy as np

from fracinj import io, paths, physics, plotting
from fracinj.solvers.exact import solve_linear_diffusion_const_flux
from fracinj.solvers.similarity import solve_neumann_n3

# %% --- Data ---------------------------------------------------------------

result_dir = paths.results_dir("3dec", "runs-wi-1e-05")
run_rigid = io.read_hdf5(result_dir / "run-q-5e-09.hdf5")
run_soft = io.read_hdf5(result_dir / "run-q-5e-05.hdf5")

# Analytical / semi-analytical reference curves
zeta_n0 = np.linspace(0, 5, 100)
theta_n0 = solve_linear_diffusion_const_flux(zeta_n0)

x_vert, w = io.sort_fields(run_soft.x_vert, run_soft.w)
_, theta_soft = physics.nondimensionalize_soft(
    x=x_vert, t=run_soft.t[-1], w=w[-1], params=run_soft.params
)
zeta_n3, theta_n3 = solve_neumann_n3(theta_soft[-1])

# %% --- Figure -----------------------------------------------------------

fig, axes = plt.subplots(
    2, 2, figsize=(6.4, 4.8 / 1.5), constrained_layout=True, dpi=200
)

sm0 = plotting.plot_nondimensionalization(
    axes[0, 0],
    axes[0, 1],
    run_rigid,
    physics.nondimensionalize_rigid,
    step=10,
    stop=len(run_rigid.t) // 4,
    ana_curve=(zeta_n0, theta_n0),
    label="Analytical",
    title=rf"$q_0 = {run_rigid.params.q_0:.0e}~\mathsf{{m^2/s}}$",
)
axes[0, 1].set(
    xlim=(0, 5),
    xlabel=r"$\zeta = \dfrac{x}{(D \, t)^{1/2}}$",
    ylabel=r"$\theta = \dfrac{w - w_i}{q(t/D)^{1/2}}$",
)

sm1 = plotting.plot_nondimensionalization(
    axes[1, 0],
    axes[1, 1],
    run_soft,
    physics.nondimensionalize_soft,
    step=10,
    ana_curve=(zeta_n3, theta_n3),
    label="Semi-analytical",
    title=rf"$q_0 = {run_soft.params.q_0:.0e}~\mathsf{{m^2/s}}$",
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

plotting.save_figure(fig, "Fig2", overleaf=True)
