import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

from fracinj import io, paths, physics, plotting

plt.style.use(["science", "notebook"])  # pyright: ignore[reportAttributeAccessIssue]

# %%

run = io.read_halfspace(paths.results_dir("halfspace", "linear") / "m_q-1e-08.hdf5")

# %%

fig = plt.figure(
    figsize=(6.4, 4.8 / 2.5), dpi=150, layout="constrained", clear=True, num=1
)
axes = fig.subplots(1, 2)

sm = plotting.plot_nondimensionalization(
    axes[0], axes[1], run, physics.nondimensionalize_soft, bc="ramp_rate", step=5
)

axes[1].set_xlim(0, 3)
axes[0].set_title(rf"$m_q={plotting.sci_latex(run.params.m_q)}\,\mathrm{{m^2 / s^{{2}} }}$", fontsize='x-large', pad=10)
axes[1].set_title("Dimensionless", fontsize='x-large', pad=10)
axes[0].annotate("(a)", (-0.25, 1.15), xycoords="axes fraction", fontsize="x-large")
axes[1].annotate("(b)", (-0.25, 1.15), xycoords="axes fraction", fontsize="x-large")

fig.colorbar(sm, ax=axes, label="$t$, [s]", pad=0.02, aspect=10)

fig.canvas.draw_idle()
plt.pause(0.01)
