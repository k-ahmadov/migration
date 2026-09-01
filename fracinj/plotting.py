"""Matplotlib helpers shared across the figure scripts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib.patches import Polygon

from fracinj import io, paths
from fracinj.types import Field, Vector


def save_figure(fig, name: str, *, ext: str = "eps", overleaf: bool = False) -> Path:
    """Save ``fig`` under the project figures directory and return the path."""
    path = paths.figure_path(name, ext=ext, overleaf=overleaf)
    fig.savefig(path)
    return path


def slope_triangle(
    ax,
    x0: float,
    prefactor: float,
    slope: float,
    dx_log: float = 0.3,
    label_slope: bool = True,
    inverse: bool = False,
    polygon_lw: float = 1.0,
    **text_kwargs,
):
    """Annotate a slope triangle on an existing log-log axis."""
    y0 = prefactor * x0**slope
    x1 = x0 * 10**dx_log
    y2 = y0 * 10 ** (slope * dx_log)

    ax.add_patch(
        Polygon(
            [[x0, y0], [x1, y0], [x1, y2]],
            fill=False,
            edgecolor="k",
            linewidth=polygon_lw,
        )
    )
    ax.annotate(
        "1",
        xy=((x0 * x1) ** 0.5, y0),
        xytext=(0, 5) if inverse else (0, -12),
        textcoords="offset points",
        ha="center",
        **text_kwargs,
    )
    if label_slope:
        ax.annotate(
            str(slope),
            xy=(x1, (y0 * y2) ** 0.5),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            **text_kwargs,
        )


def plot_nondimensionalization(
    ax1,
    ax2,
    run: io.RunData,
    nondim_fn,
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
    mantissa, exponent = f"{x:.0e}".split("e")
    return rf"{mantissa}\cdot 10^{{{int(exponent)}}}"
