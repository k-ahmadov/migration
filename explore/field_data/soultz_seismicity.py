from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fracinj import detection, io, math_utils, plotting

# %%


def _to_days(t_sec):
    return t_sec / 86400


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


# %%
t, d = io.load("soultz")
t_pct, d_pct = detection.percentile_envelope(t, d, bin_size=50, percentile=90)

# %%
t_eval = np.geomspace(t_pct[0], t_pct[-1], 500)
alpha_early = 0.5
alpha_late = 0.8
delta = 1e-2
t_c = 4 * 86400

# evaluate fitted curve
(A_early, A_late), perr = math_utils.fit_crossover_to_data(
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

# fig.savefig(
#     Path.cwd() / "figures" / "dimensionless-front" / f"field-delta-{delta:.0e}.png",
#     dpi=200,
# )
fig.canvas.draw_idle()
plt.pause(0.01)

# %%

D = A_early**2 / 4
print(D)
mu = 1e-3
# w_i = 1e-5
# k_n = 12 * D * mu / w_i**3
k_n = np.asarray([1e9, 5000e9])
w_i = (12 * D * mu / k_n)**(1/3)

print(f"k_n = {k_n / 1e9} GPa/m")
print(f"w_i = {w_i * 1e6} micro m")

a = k_n / 12 / mu
q_0 = (A_late**5 / a) ** (1 / 3)
print(f"q_0 = {q_0} m^2/s")

L = 1e3
# dimensionless deformation parameter
epsilon = w_i / (L * q_0 / a) ** (1 / 4)
print(f"epsilon = {epsilon}")

t_c_val = _to_days(a * w_i**5 / q_0**2)
print(f"t_c = {t_c_val} days")
