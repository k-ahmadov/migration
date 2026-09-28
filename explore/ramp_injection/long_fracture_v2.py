import matplotlib as mpl
import matplotlib.figure
import matplotlib.lines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple

from fracinj import analysis, io, paths, physics
from fracinj.solvers.exact import solve_linear_diffusion_ramp_flux
from fracinj.solvers.similarity import solve_neumann_ramp_n3

M_TO_MICRON = 1e6
COLUMN_WIDTH = 3.4
GOLDEN_RATIO = (1 + 5**0.5) / 2


# %%
def plot_aperture_profiles(
    ax_main: plt.Axes,
    ax_zoom: plt.Axes,
    ax_similarity_early: plt.Axes,
    ax_similarity_late: plt.Axes,
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

    norm = mpl.colors.LogNorm(vmin=run.t[idx[0]], vmax=run.t[idx[-1]])
    colors = mpl.colormaps[cmap](norm(run.t[idx]))  # pyright: ignore
    p = run.params

    for k, color in zip(idx, colors):
        ax_main.plot(run.x_vert, run.w[k] * M_TO_MICRON, color=color)
        ax_zoom.plot(run.x_vert, run.w[k] * M_TO_MICRON, color=color)
        theta_rigid = (run.w[k] - run.params.w_i) / (
            rigid.aperture_amplitude(p) * run.t[k] ** rigid.aperture_exponent
        )
        zeta_rigid = run.x_vert / (
            rigid.distance_amplitude(p) * run.t[k] ** rigid.distance_exponent
        )
        ax_similarity_early.plot(zeta_rigid, theta_rigid, color=color, zorder=-k)
        theta_soft = (run.w[k] - run.params.w_i) / (
            soft.aperture_amplitude(p) * run.t[k] ** soft.aperture_exponent
        )
        zeta_soft = run.x_vert / (
            soft.distance_amplitude(p) * run.t[k] ** soft.distance_exponent
        )
        ax_similarity_late.plot(zeta_soft, theta_soft, color=color)
    return mpl.cm.ScalarMappable(norm=norm, cmap=cmap)  # pyright: ignore


def add_rectangle(
    ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]
) -> None:
    x0, x1 = xlim
    y0, y1 = ylim
    rect = mpatches.Rectangle(
        xy=(x0, y0),
        width=x1 - x0,
        height=y1 - y0,
        edgecolor="k",
        facecolor="none",
        zorder=1000,
        linewidth=0.6,
    )
    ax.add_patch(rect)


def connect_rectangle_to_zoom(
    fig: matplotlib.figure.Figure,
    ax_zoom: plt.Axes,
    ax_main: plt.Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
):
    x0, x1 = xlim
    y0, y1 = ylim

    c1 = mpatches.ConnectionPatch(
        xyA=(x0, y0),
        coordsA=ax_main.transData,
        xyB=(x0, y1),
        coordsB=ax_zoom.transData,
        lw=0.6,
    )
    fig.add_artist(c1)
    c2 = mpatches.ConnectionPatch(
        xyA=(x1, y0),
        coordsA=ax_main.transData,
        xyB=(x1, y1),
        coordsB=ax_zoom.transData,
        lw=0.6,
    )
    fig.add_artist(c2)


def annotate_increasing_time(
    ax: plt.Axes, p1: tuple[float, float], p2: tuple[float, float]
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->"},
    )

    ax.text(
        (x1 + x2) / 2,
        (y1 + y2) / 1.9,
        r"Time $t$",
        transform=ax.transAxes,
        rotation=angle,
        transform_rotates_text=True,
        rotation_mode="anchor",
        ha="center",
        va="bottom",
    )


def annotate_panel_labels(
    axes: list[plt.Axes], xy: tuple[float, float] = (0.25, 0.95)
) -> None:
    ascii_value = ord("a")
    for i, ax in enumerate(axes):
        ax.annotate(
            f"({chr(ascii_value + i)})",
            xy,
            xycoords="axes fraction",
            backgroundcolor="white",
        )


# %%

filepath = paths.results_dir("halfspace", "ramp", "L-2km") / "m_q-1e-13.hdf5"
run = io.read_halfspace(filepath)

RIGID = analysis.Regime(
    name="rigid", rate="ramp", n=0, i=1, coefficient=physics.diffusivity, theta0=None
)
SOFT = analysis.Regime(
    name="soft", rate="ramp", n=3, i=1, coefficient=physics.mobility, theta0=None
)
# regime = RIGID
# aperture_amplitude = regime.aperture_amplitude(run.params)
# distance_amplitude = regime.distance_amplitude(run.params)

zeta_ana = np.linspace(0, 4, 200)
theta_ana = solve_linear_diffusion_ramp_flux(zeta_ana)

zeta_semi_soft, theta_semi_soft = solve_neumann_ramp_n3(zeta_max=2.0)

# %%


mosaic: list = [["main", "similarity-late"], ["zoom-early", "similarity-early"]]
fig = plt.figure(
    figsize=(2 * COLUMN_WIDTH, 2 * COLUMN_WIDTH / GOLDEN_RATIO),
    layout="constrained",
    dpi=100,
    num=1,
    clear=True,
)
axes: dict[str, plt.Axes] = fig.subplot_mosaic(mosaic, width_ratios=[1, 1])  # pyright: ignore

sm = plot_aperture_profiles(
    axes["main"],
    axes["zoom-early"],
    axes["similarity-early"],
    axes["similarity-late"],
    run,
    RIGID,
    SOFT,
    t_window=(0.002, 1.0),
    n_profiles=12,
    cmap="viridis_r",
)

axes["similarity-early"].plot(zeta_ana, theta_ana, "k-", label="Analytical")
axes["similarity-early"].legend()

axes["similarity-late"].plot(zeta_semi_soft, theta_semi_soft, "k-", label="Semi-analytical")
axes["similarity-late"].legend()

numerical = tuple(
    matplotlib.lines.Line2D([], [], color=c) for c in sm.cmap(np.linspace(0, 1, 12))
)
axes["main"].legend(
    [numerical],
    ["Numerical"],
    handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},  # pyright: ignore
)

x0, x1, y0, y1 = -3, 80, 9.95, 11

add_rectangle(axes["main"], (x0, x1), (y0, y1))

connect_rectangle_to_zoom(fig, axes["zoom-early"], axes["main"], (x0, x1), (y0, y1))

annotate_increasing_time(axes["main"], p1=(0.25, 0.25), p2=(0.7, 0.7))

axes["main"].set(xlabel=r"Distance $x$ (m)", ylabel=r"Aperture $w$ ($\mu$m)")
axes["zoom-early"].set(
    xlabel=r"Distance $x$ (m)",
    ylabel=r"Aperture $w$ ($\mu$m)",
    xlim=(x0, x1),
    ylim=(y0, y1),
)
axes["similarity-early"].set(
    xlabel=r"$x / \sqrt{ D t }$",
    ylabel=r"$(w - w_{\mathrm{i}}) / m_{\mathrm{q}} \sqrt{ t^{3}/D } $",
    xlim=(-0.2, 4),
)
axes["similarity-late"].set(
    xlabel=r"$x / \left( a m_{\mathrm{q}}^{3} t^{7} \right) ^{1/5}$",
    ylabel=r"$(w - w_{\mathrm{i}}) / \left( m_{\mathrm{q}}^{2} t^{3} / a \right)^{1/5}$",
    xlim=(-0.2, 4),
)
axes["similarity-early"].sharey(axes["similarity-late"])

annotate_panel_labels(list(axes.values()))

fig.colorbar(
    sm,
    ax=list(axes.values()),
    label="Time $t$ (s)",
    location="top",
    aspect=50,
)

plt.pause(0.01)

fig.savefig("/home/kahmadov/phd/migration/figures/paper/ramp-profiles-similarity-v3.eps")
fig.savefig("/home/kahmadov/phd/migration/tex/2026-09-10_summary/figures/ramp-profiles-similarity-v3.eps")
