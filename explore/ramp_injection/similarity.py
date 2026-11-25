import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from fracinj import io, paths, physics
from fracinj.analysis import Regime

COLUMN_WIDTH = 3.25  # in

# %%

RIGID_RAMP = Regime(
    name="rigid",
    rate="ramp",
    n=0,
    i=1,
    coefficient=physics.diffusivity,
    theta0=None,
)
SOFT_RAMP = Regime(
    name="soft",
    rate="ramp",
    n=3,
    i=1,
    coefficient=physics.mobility,
    theta0=None,
)

run_rigid = io.read_halfspace(paths.results_dir("halfspace", "ramp") / "m_q-1e-14.hdf5")
run_soft = io.read_halfspace(paths.results_dir("halfspace", "ramp") / "m_q-1e-07.hdf5")


# %%
def plot_profiles(
    run,
    regime,
    axes,
    *,
    t_window,
    subtract_initial,
    panel_labels=("(a)", "(b)"),
    n_profiles=6,
    zeta_lim=(-0.2, 4),
):
    """Plot aperture profiles and their self-similar collapse on a pair of axes.

    axes: (ax_profiles, ax_collapse).
    t_window: (start, stop) as fractions of the number of time steps.
    subtract_initial: if True, theta uses w - w_i instead of w.
    """
    ax1, ax2 = axes

    n_t = len(run.t)
    start, stop = int(t_window[0] * n_t), int(t_window[1] * n_t)
    idx = np.unique(np.linspace(start, stop - 1, n_profiles).astype(int))
    colors = mpl.colormaps["viridis"](mpl.colors.Normalize()(run.t[idx]))  # pyright: ignore

    aperture_amplitude = regime.aperture_amplitude(run.params)
    distance_amplitude = regime.distance_amplitude(run.params)
    w0 = run.params.w_i if subtract_initial else 0.0
    scales = physics.cubic_scales(run.params, pin="aperture", ramp=True)

    for k, color in zip(idx, colors):
        t_k, w_k = run.t[k], run.w[k]
        ax1.plot(
            run.x_vert / scales.x_c,
            w_k / scales.w_c,
            color=color,
            lw=1.5,
        )
        theta = (w_k - w0) / (aperture_amplitude * t_k**regime.aperture_exponent)
        zeta = run.x_vert / (distance_amplitude * t_k**regime.distance_exponent)
        ax2.plot(
            zeta,
            theta,
            color=color,
            lw=1.5,
            label=rf"$\hat{{t}} = {t_k / scales.t_c:.2g}$",
        )

    ax1.set(
        xlabel=r"Dimensionless distance $\hat{x}$",
        ylabel=r"Dimensionless aperture $\hat{w}$",
    )
    ax2.set(xlabel=r"$\zeta$", ylabel=r"$\theta$", xlim=zeta_lim)

    for ax, label in zip((ax1, ax2), panel_labels):
        ax.annotate(
            label, (0.175, 0.975), xycoords="axes fraction", backgroundcolor="white"
        )

    ax2.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1,
        loc="upper right",
    )


# %%
fig = plt.figure(
    figsize=(2 * COLUMN_WIDTH, 2 * 0.675 * COLUMN_WIDTH),
    dpi=150,
    layout="constrained",
    clear=True,
    num=1,
)
axes = fig.subplots(2, 2)

plot_profiles(
    run_rigid,
    RIGID_RAMP,
    axes[0],
    t_window=(0.01, 0.125),
    subtract_initial=True,
    panel_labels=("(a)", "(b)"),
)
plot_profiles(
    run_soft,
    SOFT_RAMP,
    axes[1],
    t_window=(0.05, 0.95),
    subtract_initial=False,
    panel_labels=("(c)", "(d)"),
)
axes[0][1].set(
    xlabel=r"$\zeta=\hat{x} / \hat{t}^{1/2}$",
    ylabel=r"$\theta= \hat{w} /  \hat{t}^{3/2}$",
)
axes[1][1].set(
    xlabel=r"$\zeta=\hat{x} /  \hat{t}^{7/5}$",
    ylabel=r"$\theta=\hat{w} /  \hat{t}^{3/5}$",
)

fig.canvas.draw_idle()
plt.pause(0.01)

# fig.savefig("/home/kahmadov/phd/migration/figures/paper/profiles-similarity.eps")
