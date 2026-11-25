import matplotlib as mpl
import matplotlib.figure
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import AutoMinorLocator

from fracinj import analysis, io, paths, physics, plotting
from fracinj.solvers.exact import solve_linear_diffusion_ramp_flux
from fracinj.solvers.similarity import solve_neumann_ramp_n3
from fracinj.types import Parameters

M_TO_MICRON = 1e6
COLUMN_WIDTH = 3.25
GOLDEN_RATIO = (1 + 5**0.5) / 2

# %%
filepath = paths.results_dir("halfspace", "ramp", "long") / "m_q-1e-13.hdf5"
run = io.read_halfspace(filepath)

RIGID = analysis.Regime(
    name="rigid", rate="ramp", n=0, i=1, coefficient=physics.diffusivity, theta0=None
)
SOFT = analysis.Regime(
    name="soft", rate="ramp", n=3, i=1, coefficient=physics.mobility, theta0=None
)

# Early-time similarity variable eta = x / (2 sqrt(D0 t)).
# solve_linear_diffusion_ramp_flux takes x / sqrt(D0 t) = 2 * eta.
eta_ana = np.linspace(0, 2, 200)
theta_ana = solve_linear_diffusion_ramp_flux(eta_ana)

zeta_semi_soft, theta_semi_soft = solve_neumann_ramp_n3(zeta_max=0.60, n_points=2000)

# %% zoom inside main


def dimensionless(
    w: np.ndarray, x: np.ndarray, t: float, regime: analysis.Regime, params: Parameters
) -> tuple[np.ndarray, np.ndarray]:
    theta = w / (regime.aperture_amplitude(params) * t**regime.aperture_exponent)
    zeta = x / (regime.distance_amplitude(params) * t**regime.distance_exponent)
    return zeta, theta


def plot_aperture_profiles(
    ax_main: Axes,
    ax_zoom: Axes,
    ax_similarity_early: Axes,
    ax_similarity_late: Axes,
    run: io.RunData,
    rigid: analysis.Regime,
    soft: analysis.Regime,
    *,
    t_window: tuple[float, float] = (0.0, 1.0),
    n_profiles: int = 12,
    cmap: str = "cividis",
):
    n_t = len(run.t)
    start = max(int(t_window[0] * n_t), 1)  # skip t = 0 (log scale)
    stop = int(t_window[1] * n_t)

    # log-spaced in *time*, snapped to nearest available sample
    t_targets = np.geomspace(run.t[start], run.t[stop - 1], n_profiles)
    idx = np.unique(np.searchsorted(run.t, t_targets).clip(start, stop - 1))

    t_hours = run.t / 3600  # for the colour scale only
    norm = mpl.colors.LogNorm(vmin=t_hours[idx[0]], vmax=t_hours[idx[-1]])  # pyright: ignore
    colors = mpl.colormaps[cmap](norm(t_hours[idx]))  # pyright: ignore
    p = run.params

    for k, color in zip(idx, colors):
        ax_main.plot(run.x_vert, run.w[k] * M_TO_MICRON, color=color)
        ax_zoom.plot(run.x_vert, run.w[k] * M_TO_MICRON, color=color)
        zeta, theta = dimensionless(
            run.w[k] - run.params.w_i, run.x_vert, float(run.t[k]), rigid, p
        )
        # rigid distance scale is sqrt(D0 t); divide by 2 to get eta
        ax_similarity_early.plot(zeta / 2, theta, color=color, zorder=-k, lw=0.6)
        zeta, theta = dimensionless(
            run.w[k] - run.params.w_i, run.x_vert, float(run.t[k]), soft, p
        )
        ax_similarity_late.plot(zeta, theta, color=color, lw=0.6)
    return mpl.cm.ScalarMappable(norm=norm, cmap=cmap)  # pyright: ignore


# %%
mosaic: list = [["main", "similarity-early"], ["main", "similarity-late"]]
fig = plt.figure(
    figsize=(2 * COLUMN_WIDTH, 1.2 * COLUMN_WIDTH / GOLDEN_RATIO),
    layout="constrained",
    dpi=200,
    num=1,
    clear=True,
)
axes: dict[str, plt.Axes] = fig.subplot_mosaic(mosaic, width_ratios=[GOLDEN_RATIO, 1])  # pyright: ignore

x0, x1, y0, y1 = -3, 100, 9.95, 12.1
ax_zoom = axes["main"].inset_axes(
    [0.22, 0.2, 0.45, 0.45 / GOLDEN_RATIO],
    transform=axes["main"].transAxes, 
)

sm = plot_aperture_profiles(
    axes["main"],
    ax_zoom,
    axes["similarity-early"],
    axes["similarity-late"],
    run,
    RIGID,
    SOFT,
    t_window=(0.002, 1.0),
    n_profiles=12,
    cmap="viridis_r",
)

axes["similarity-early"].plot(eta_ana, theta_ana, "k-", label="Early-time")
axes["similarity-early"].legend()

axes["similarity-late"].plot(zeta_semi_soft, theta_semi_soft, "k-", label="Late-time")
axes["similarity-late"].legend(loc="upper right")

numerical = tuple(
    matplotlib.lines.Line2D([], [], color=c) for c in sm.cmap(np.linspace(0, 1, 12))
)
axes["main"].legend(
    [numerical],
    ["Numerical"],
    handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
)

ax_zoom.set(
    xlim=(x0, x1),
    ylim=(y0, y1),
)
ax_zoom.tick_params(
    axis="both", which="both", length=0, labelbottom=False, labelleft=False
)

axes["main"].set(
    xlabel=r"Distance $x$ (km)",
    ylabel=r"Aperture $w$ ($\mu$m)",
    yticks=[10, 20, 40, 60, 80],
    xticks=[0, 1000, 2000],
)
axes["main"].set_xticklabels(
    [int(t / 1000) for t in axes["main"].get_xticks()]
)  # # pyright: ignore
axes["main"].yaxis.set_minor_locator(AutoMinorLocator(2))

for key, xlabel, ylabel in [
    (
        "similarity-early",
        r"$x  /  (2\sqrt{ D_{0} t })$",
        r"$ \dfrac{\sqrt{D_{0}} \Delta w }{m_{\mathrm{q}}\, t^{3/2}} $",
    ),
    (
        "similarity-late",
        r"$x / \left( a\, m_{\mathrm{q}}^{3}\, t^{7} \right)^{1/5}$",
        r"$\dfrac{\Delta w }{\left( m_{\mathrm{q}}^{2}\, t^{3} / a \right)^{1/5}} $",
    ),
]:
    ax = axes[key]
    ax.set(xlim=(0, 1.5), yticks=[0, 1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

axes["similarity-early"].sharey(axes["similarity-late"])

plotting.add_rectangle(
    axes["main"],
    (x0, x1),
    (y0, y1),
    edgecolor="0.3",
    facecolor="none",
    lw=0.6,
    zorder=1000,
)

plotting.connect_rectangle_to_zoom(
    fig, ax_zoom, axes["main"], (x0, y1), (x1, y0), lw=0.6, color="0.3"
)

fig.colorbar(
    sm,
    ax=list(axes.values()),
    label="Time $t$ (h)",
    location="right",
)

plotting.annotate_panel_labels(
    [axes["main"], ax_zoom, axes["similarity-early"], axes["similarity-late"]],
    xy=(0.2, 0.98),
)

fig.canvas.draw_idle()
plt.pause(0.01)

# %%
plotting.save_figure(fig, "similarity-inset", ext="pdf")

# fig.savefig("similarity-inset.eps")
