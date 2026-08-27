from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from mypackages import math_utils, plotting, typesdefs


# %%
def load(filepath):
    # load data file
    df = pd.read_csv(filepath, sep=";")
    # formatting string for date and time format
    fmt = "%d/%m/%Y %H:%M:%S.%f"
    # import time
    timestamps = pd.to_datetime(
        df["Date (local)"] + " " + df["Time (local)"], format=fmt
    )
    # cut data to injection period
    date_mask = timestamps < timestamps.iloc[0] + pd.Timedelta(days=18)
    df = df.loc[date_mask]
    timestamps = timestamps[date_mask]
    # import time in seconds
    t = (timestamps - timestamps.iloc[0]).dt.total_seconds()
    # import coordinates of events
    x, y, z = df["East (m)"], df["North (m)"], df["Depth (m)"]
    # median of first 10 events
    x0, y0, z0 = x.iloc[:10].median(), y.iloc[:10].median(), z.iloc[:10].median()
    # euclidian distance
    distance = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
    return t.values, distance.values


def percentile_envelope(t_days, distance, bin_size, percentile):
    "Chooses the distance based on the given percentile per bin"
    # number of bins based on bin size
    n_bins = len(distance) // bin_size
    n_trim = n_bins * bin_size
    # trim distance and time data per bin
    d_bins = distance[:n_trim].reshape(n_bins, bin_size)
    t_bins = t_days[:n_trim].reshape(n_bins, bin_size)
    # choose distance value from each bin according to the given percentile
    p_vals = np.percentile(d_bins, percentile, axis=1)
    # find its indices
    idx = np.argmin(np.abs(d_bins - p_vals[:, None]), axis=1)
    # import the corresponding time and distance values
    t_pct = t_bins[np.arange(n_bins), idx]
    d_pct = d_bins[np.arange(n_bins), idx]
    return t_pct, d_pct


# TODO: instead of cumulative time intervals pick percentiles from time bins (sliding windows)
def cumulative_percentile_envelope(
    t_sec, distance, percentile, t_eval=None, min_events=10
):
    """
    Percentile of distance among all events up to time t, evaluated
    on a set of times t_eval (defaults to a log-spaced grid).
    """
    order = np.argsort(t_sec)
    t_sorted = t_sec[order]
    d_sorted = distance[order]

    if t_eval is None:
        t_eval = np.linspace(
            t_sorted[min_events - 1],  # start once we have min_events
            t_sorted[-1],
            200,
        )

    p_vals = np.full_like(t_eval, np.nan, dtype=float)
    for i, t in enumerate(t_eval):
        mask = t_sorted <= t
        n = mask.sum()
        if n >= min_events:
            p_vals[i] = np.percentile(d_sorted[mask], percentile)

    return t_eval, p_vals


def _to_days(t_sec):
    return t_sec / 86400


def fit_crossover_to_data(
    t: typesdefs.Vector,
    d: typesdefs.Vector,
    alpha_early: float,
    alpha_late: float,
    delta: float,
    t_c: float,
    p0: list[float],
) -> tuple[list[float], np.ndarray]:
    """
    Fit a two-regime power-law crossover model to data by optimizing
    the prefactors (a, b), holding the exponents, crossover scale,
    and sharpness fixed.

    Inputs:
        t: Independent variable (e.g. time).
        d: Dependent variable / data to fit against.
        alpha_early, alpha_late: Fixed power-law exponents for the
            early- and late-time regimes.
        delta: Sharpness of the crossover transition.
        t_c: Crossover (transition) scale.
        p0: Initial guess for [a, b].

    Returns:
        popt: Best-fit [a, b].
        perr: 1-sigma uncertainties on [a, b].
    """

    def crossover_fit(t, a_early, a_late):
        return math_utils.crossover(
            t,
            a=a_early,
            alpha=alpha_early,
            b=a_late,
            beta=alpha_late,
            x0=t_c,
            delta=delta,
        )

    popt, pcov = curve_fit(crossover_fit, t, d, p0=p0, maxfev=10000)
    perr = np.sqrt(np.diag(pcov))

    a_fit, b_fit = popt
    print(f"a = {a_fit:.4g} ± {perr[0]:.2g}")
    print(f"b = {b_fit:.4g} ± {perr[1]:.2g}")

    return popt, perr


# %%
t, d = load(
    Path.cwd() / "data" / "SSFS1993-Catalogue_Bourouis.csv",
)

t_pct, d_pct = percentile_envelope(t, d, bin_size=50, percentile=90)

# %%
t_eval = np.geomspace(t_pct[0], t_pct[-1], 500)
alpha_early = 0.5
alpha_late = 0.8
delta = 1e-2
t_c = 4 * 86400

# evaluate fitted curve
(A_early, A_late), perr = fit_crossover_to_data(
    t_pct,
    d_pct,
    alpha_early=alpha_early,
    alpha_late=alpha_late,
    delta=delta,
    t_c=t_c,
    p0=[1, 1],
)

d_fit = math_utils.crossover(
    x=t_eval,
    a=A_early,
    alpha=alpha_early,
    b=A_late,
    beta=alpha_late,
    x0=t_c,
    delta=delta,
)

# %%  Plot
fig, ax = plt.subplots(
    figsize=(6.4 / 1.4, 4.8 / 1.4), dpi=100, layout="constrained", clear=True, num=1
)
ax.scatter(_to_days(t), d, color="tab:gray", s=3)
ax.scatter(_to_days(t_pct), d_pct, color="k", label="P90 per 50 events", s=25)
ax.plot(
    _to_days(t_eval),
    d_fit,
    color="tab:blue",
    lw=3,
    label=rf"Crossover, $\delta={delta}$",
)
ax.axvline(
    _to_days(t_c),
    color="k",
    ls=":",
    label=rf"$t_c=\mathrm{{{_to_days(t_c):.1g}\ days}}$",
)
ax.text(
    0.1,
    0.85,
    rf"$A_{{\mathrm{{early}}}} = {A_early:.2f} \pm {perr[0]:.1g},$"
    "\n"
    rf"$A_{{\mathrm{{late}}}} = {A_late:.2f} \pm {perr[1]:.1g}$",
    transform=ax.transAxes,
)
ax.set(
    xlabel="Time [day]",
    ylabel="Distance [m]",
    title="Soultz 1993",
    xscale="log",
    yscale="log",
    xlim=(0.5, 20),
    ylim=(50, 1200),
)


exponent, log_prefactor = np.polyfit(
    np.log(_to_days(t_eval[: np.searchsorted(t_eval, t_c)])),
    np.log(d_fit[: np.searchsorted(t_eval, t_c)]),
    deg=1,
)
plotting.slope_triangle(
    ax=ax,
    x0=_to_days(t_eval[30]),
    slope=round(exponent, 2),
    prefactor=np.exp(log_prefactor + 1.3),
)

exponent, log_prefactor = np.polyfit(
    np.log(_to_days(t_eval[np.searchsorted(t_eval, t_c) :])),
    np.log(d_fit[np.searchsorted(t_eval, t_c) :]),
    deg=1,
)
plotting.slope_triangle(
    ax=ax,
    x0=_to_days(t_eval[340]),
    slope=round(exponent, 2),
    prefactor=np.exp(log_prefactor - 0.5),
)

ax.spines[["right", "top"]].set_visible(False)
ax.legend(loc="lower right", framealpha=1)

fig.savefig(
    Path.cwd() / "figures" / "dimensionless-front" / f"field-delta-{delta:.0e}.png", dpi=200
)
fig.canvas.draw_idle()
plt.pause(0.01)

# %%

D = A_early**2 / 4
print(D)
mu = 1e-3
w_i = 1e-5
k_n = 12 * D * mu / w_i**3

print(f"k_n = {k_n / 1e9:.5f} GPa/m")

a = k_n / 12 / mu
q_0 = (A_late**5 / a) ** (1 / 3)
print(f"q_0 = {q_0:.1e} m^2/s")

L = 1e3
# dimensionless deformation parameter
epsilon = w_i / (L * q_0 / a) ** (1 / 4)
print(f"epsilon = {epsilon:.2g}")

t_c_val = _to_days(a * w_i**5 / q_0**2)
print(f"t_c = {t_c_val:.2g} days")
