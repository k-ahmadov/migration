import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib.patches import Polygon

from mypackages import file_io, types


def slope_triangle(
    ax,
    x0: float,
    prefactor: float,
    slope: float,
    dx_log: float = 0.3,
    label_slope: bool = True,
    inverse: bool = False,
):
    """Annotate a slope triangle on an existing loglog axis."""

    y0 = prefactor * x0**slope  # assumes curve passes through here; adjust as needed
    x1 = x0 * 10**dx_log
    y2 = y0 * 10 ** (slope * dx_log)

    tri = Polygon(
        [[x0, y0], [x1, y0], [x1, y2]],
        fill=False,
        edgecolor="k",
    )
    ax.add_patch(tri)
    ax.annotate(
        "1",
        xy=((x0 * x1) ** 0.5, y0),
        xytext=(0, -12) if not inverse else (0, 5),
        textcoords="offset points",
        ha="center",
        fontsize=10,
    )
    if label_slope:
        ax.annotate(
            str(slope),
            xy=(x1, (y0 * y2) ** 0.5),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            fontsize=10,
        )


def plot_nondimensionalization(
    ax1,
    ax2,
    run,
    nondim_fn,
    *,
    step: int,
    stop: int | None = None,
    ana_curve: tuple[np.ndarray, np.ndarray] | None = None,
    label: str = "",
    cmap: str | mcolors.Colormap = "viridis",
) -> cm.ScalarMappable:
    stop = len(run.t) if stop is None else stop
    cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap

    x_vert, w = file_io.sort_fields(run.x_vert, run.w)
    zeta, theta = nondim_fn(x=x_vert, t=run.t, w=w, params=run.params)
    norm = mcolors.Normalize(vmin=run.t[step], vmax=run.t[stop - 1])

    for i in range(step, stop, step):
        color = cmap(norm(run.t[i]))
        ax1.plot(x_vert, w[i] * 1e3, color=color)
        ax2.plot(zeta[i], theta[i], color=color)

    if ana_curve is not None:
        ax2.plot(*ana_curve, "k-", label=label)
        ax2.legend()

    ax1.set(xlabel="$x$, [m]", ylabel="$w$, [mm]")
    ax2.set(xlabel=r"$\zeta$", ylabel=r"$\theta$")

    return cm.ScalarMappable(cmap=cmap, norm=norm)


def plot_profiles(
    ax,
    x: types.XPositions,
    field: types.Field,
    t: types.Time,
    *,
    n=10,
    cut=0,
    title="",
    ylabel="Field",
):
    if cut == 0:
        cut = field.shape[0]
    for i in range(2, cut, (cut) // n):
        ax.plot(x, field[i], ".", label=f"t={t[i]} s")
    ax.set(
        xlabel="Distance $x$, [m]",
        ylabel=ylabel,
        title=title,
    )
