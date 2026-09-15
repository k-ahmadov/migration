"""Matplotlib helpers shared across the figure scripts."""

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib.patches import Polygon

from fracinj import io, paths
from fracinj.types import Field, RateBC, Vector


def save_figure(fig, name: str, *, ext: str = "eps", overleaf: bool = False) -> Path:
    """Save ``fig`` under the project figures directory and return the path."""
    path = paths.figure_path(name, ext=ext, overleaf=overleaf)
    fig.savefig(path)
    return path


def slope_triangle(
    ax: plt.Axes,
    x0: float,
    slope: float,
    *,
    prefactor: float | None = None,
    y0_data: float | None = None,
    dx_log: float = 0.3,
    label_slope: bool = True,
    inverse: bool = False,
    color: str = "k",
    **text_kwargs,
):
    """Annotate a slope triangle on an existing log-log axis.

    Anchor it either at a curve value (``prefactor``, giving
    ``y0 = prefactor * x0**slope``) or directly (``y0_data``) -- exactly one
    of the two must be given. ``inverse`` flips the right-angle corner from
    bottom-right to top-left, for annotating a curve from the other side
    (e.g. two power laws sharing the same stretch of the plot).
    """
    if (prefactor is None) == (y0_data is None):
        raise ValueError("provide exactly one of `prefactor` or `y0_data`")
    y0 = prefactor * x0**slope if y0_data is None else y0_data

    x1 = x0 * 10**dx_log
    y2 = y0 * 10 ** (slope * dx_log)
    # the horizontal ("1") and vertical (slope) legs swap sides in inverse mode
    horizontal_y = y2 if inverse else y0
    vertical_x = x0 if inverse else x1

    vertices = (
        [[x0, y0], [x0, y2], [x1, y2]] if inverse else [[x0, y0], [x1, y0], [x1, y2]]
    )
    ax.add_patch(Polygon(vertices, fill=False, edgecolor=color))

    ax.annotate(
        "1",
        xy=((x0 * x1) ** 0.5, horizontal_y),
        xytext=(0, 5) if inverse else (0, -12),
        textcoords="offset points",
        ha="center",
        color=color,
        **text_kwargs,
    )
    if label_slope:
        ax.annotate(
            f"{slope:.2g}",
            xy=(vertical_x, (y0 * y2) ** 0.5),
            xytext=(-6, 0) if inverse else (6, 0),
            textcoords="offset points",
            ha="right" if inverse else "left",
            va="center",
            color=color,
            **text_kwargs,
        )


def plot_nondimensionalization(
    ax1,
    ax2,
    run: io.RunData,
    nondim_fn: Callable,
    bc: RateBC = "const_rate",
    *,
    step: int,
    stop: int | None = None,
    ana_curve: tuple[np.ndarray, np.ndarray] | None = None,
    label: str = "",
    title: str = "",
    cmap: str | mcolors.Colormap = "viridis",
) -> cm.ScalarMappable:
    """Plot aperture profiles (``ax1``) and their similarity collapse (``ax2``)."""
    stop = len(run.t) if stop is None else stop
    cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap

    x_vert, w = io.sort_fields(run.x_vert, run.w)
    zeta, theta = nondim_fn(x=x_vert, t=run.t, w=w, params=run.params, bc=bc)
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
    if title:
        ax1.set_title(title)
    return cm.ScalarMappable(cmap=cmap, norm=norm)


def plot_profiles(
    ax,
    x: Vector,
    field: Field,
    t: Vector,
    *,
    n: int = 10,
    cut: int = 0,
    title: str = "",
    ylabel: str = "Field",
):
    cut = cut or field.shape[0]
    for i in range(2, cut, cut // n):
        ax.plot(x, field[i], ".-", label=f"t={t[i]} s")
    ax.set(xlabel="Distance $x$, [m]", ylabel=ylabel, title=title)


def sci_latex(x: float) -> str:
    mantissa, exponent = f"{x:.1e}".split("e")
    if float(mantissa) == 1.0:
        return rf"10^{{{int(exponent)}}}"
    else:
        return rf"{mantissa}\times 10^{{{int(exponent)}}}"
