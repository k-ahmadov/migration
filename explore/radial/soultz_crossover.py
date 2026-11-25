"""Radial ramp crossover model fitted to the Soultz 1993 seismicity front.

The 1993 GPK1 stimulation raised the injection rate in steps from ~0.3 to
~37 l/s over the first ~12 days (mean slope ~2.7 l/s/day), i.e. roughly a
linear ramp from a point-like open-hole section: the radial ramp case. Its
front migrates as

    rigid (t << t_*):  ell ~ a t^(1/2)
    soft  (t >> t_*):  ell ~ b t^(7/8)

(see ``front_dimensionless.py``). The crossover uses the same form and
sharpness (delta = 1) as the numerical front fit, but t_* is not imposed: it is
placed where the two asymptotes meet, t_* = (a/b)^(1/(7/8 - 1/2)), so only the
prefactors a, b are fitted and t_* comes out of the fit.

The seismicity front is the 90th-percentile distance per bin of 50 events
(as in ``scripts/fig5_seismicity.py``). Events up to shut-in are fitted; note
that the last ~3 days before it (12-15 d) are at constant rate (~37 l/s), not
a ramp. Post-shut-in events are shown but not used.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from fracinj import detection, io, math_utils, paths, plotting

DAY = 86_400
ALPHA, BETA = 1 / 2, 7 / 8  # radial ramp front exponents
DELTA = 1.0  # crossover sharpness, as for the simulated front
RAMP_END = 12.05  # days: end of the stepped rate increase
SHUT_IN = 15.14  # days: injection stops (hydraulic record)
PERCENTILE, BIN_SIZE = 90, 50

HYDRAULIC = (
    paths.DATA / "SSFS1993-Hydraulic" / "SSFS1993-Hydraulic_september"
    / "SSFS1993-Hydraulic_september.csv"
)
CATALOGUE_T0 = pd.Timestamp("1993-09-02 16:19:11.72")  # first event, local time

# %% data

t_ev, r_ev = io.load("soultz")
t_ev = t_ev / DAY

hyd = pd.read_csv(HYDRAULIC, sep=";", parse_dates=["datetime"])
t_q = (hyd["datetime"] - CATALOGUE_T0).dt.total_seconds().to_numpy() / DAY
q = hyd["q_in_comb_max (l/s)"].to_numpy()

injecting = t_ev < SHUT_IN
t_env, r_env = detection.percentile_envelope(
    t_ev[injecting], r_ev[injecting], bin_size=BIN_SIZE, percentile=PERCENTILE
)
t_env, r_env = np.asarray(t_env), np.asarray(r_env)
t_env, r_env = t_env[t_env > 0], r_env[t_env > 0]

# %% fit: a t^1/2 -> b t^7/8, crossing over where the asymptotes meet


def crossover_time(a, b):
    return (a / b) ** (1 / (BETA - ALPHA))


def front_model(t, a, b):
    return math_utils.crossover(
        t, a, ALPHA, b, BETA, x0=crossover_time(a, b), delta=DELTA
    )


# fit log a, log b in log space, so every decade of distance weighs the same
(log_a, log_b), cov = curve_fit(
    lambda t, la, lb: np.log(front_model(t, np.exp(la), np.exp(lb))),
    t_env,
    np.log(r_env),
    p0=(np.log(200.0), np.log(100.0)),
)
a, b = np.exp(log_a), np.exp(log_b)
err_log_a, err_log_b = np.sqrt(np.diag(cov))
t_star = crossover_time(a, b)
# t_* = (a/b)^(8/3): propagate the (correlated) log errors
err_log_t = (1 / (BETA - ALPHA)) * np.sqrt(
    cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
)
resid = np.log(r_env) - np.log(front_model(t_env, a, b))

print(f"a   = {a:.0f} m/day^1/2   (+-{100 * err_log_a:.0f}%)")
print(f"b   = {b:.0f} m/day^7/8   (+-{100 * err_log_b:.0f}%)")
print(f"t_* = {t_star:.1f} days    (+-{100 * err_log_t:.0f}%)")
print(f"rms misfit (log) = {np.sqrt(np.mean(resid**2)):.3f}")
m_q = np.polyfit(t_q[(t_q > 0) & (t_q < RAMP_END)], q[(t_q > 0) & (t_q < RAMP_END)], 1)[0]
print(f"mean ramp slope = {m_q:.2f} l/s/day = {m_q * 1e-3 / DAY:.2e} m^3/s^2")

t_fit = np.geomspace(t_env[0], SHUT_IN, 200)
t_asym = np.geomspace(0.4, 20, 200)  # asymptotes across the whole axis

# %% figure

CW, GR = 3.25, 1.618
fig = plt.figure(figsize=(CW, 1.5*CW/GR), dpi=300, layout="constrained",
                 clear=True, num=1)
ax_q, ax = fig.subplots(2, 1, sharex=True, height_ratios=(1, 3))

ax_q.plot(t_q[t_q > 0], q[t_q > 0], color="k", lw=0.8)
ax_q.axvspan(SHUT_IN, t_ev.max(), color="0.93", lw=0)
ax_q.set(ylabel="$Q$ (l/s)", ylim=(0, 45))

ax.axvspan(SHUT_IN, t_ev.max(), color="0.93", lw=0)
ax.scatter(t_ev, r_ev, s=2, color="0.7", rasterized=True, label="Events")
ax.scatter(t_env, r_env, s=10, color="k", label=r"Envelope (90\%)")
ax.plot(t_fit, front_model(t_fit, a, b), color="C0", lw=1.5, label="Crossover")
ASYM = {"color": "r", "ls": (0, (4, 2)), "lw": 1.3, "zorder": 3}
LABEL = {"textcoords": "offset points", "color": "k", "zorder": 4,
         "bbox": {"facecolor": "white", "edgecolor": "none", "pad": 0.5}}
ax.plot(t_asym, a * t_asym**ALPHA, **ASYM, label="Asymptotes")
ax.plot(t_asym, b * t_asym**BETA, **ASYM)
ax.axvline(t_star, color="0.1", ls="--", lw=0.8)
ax.annotate(rf"$t_* = {t_star:.1f}$ d", (t_star, 45), color="0.1",
            ha="center", backgroundcolor="1")
ax.annotate(r"$\propto t^{1/2}$", (0.45, a * 0.45**ALPHA), xytext=(0, -6),
            ha="left", va="top", **LABEL)
ax.annotate(r"$\propto t^{7/8}$", (1.5, b * 1.5**BETA), xytext=(6, -6),
            ha="left", va="top", **LABEL)

ax.set(xscale="log", yscale="log", xlim=(0.4, 20), ylim=(30, 1500),
       xlabel="Time since first event (days)", ylabel="Distance (m)")
ax.set_xticks([0.5, 1, 2, 5, 10, 20])
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(loc="upper left", frameon=False, fontsize="small")

fig.canvas.draw_idle()
plt.pause(0.01)

# %%

# plotting.save_figure(fig, "soultz-radial-crossover", ext="pdf")

# %% same figure, linear axes

t_fit_lin = np.linspace(t_env[0], SHUT_IN, 200)
t_asym_lin = np.linspace(0, 20, 200)

fig_lin = plt.figure(figsize=(CW, 1.5*CW/GR), dpi=300, layout="constrained",
                     clear=True, num=2)
ax_q_lin, ax_lin = fig_lin.subplots(2, 1, sharex=True, height_ratios=(1, 3))

ax_q_lin.plot(t_q[t_q > 0], q[t_q > 0], color="k", lw=0.8)
ax_q_lin.axvspan(SHUT_IN, t_ev.max(), color="0.93", lw=0)
ax_q_lin.set(ylabel="$Q$ (l/s)", ylim=(0, 45))

ax_lin.axvspan(SHUT_IN, t_ev.max(), color="0.93", lw=0)
ax_lin.scatter(t_ev, r_ev, s=2, color="0.7", rasterized=True, label="Events")
ax_lin.scatter(t_env, r_env, s=10, color="k", label=r"Envelope (90\%)")
ax_lin.plot(t_fit_lin, front_model(t_fit_lin, a, b), color="C0", lw=1.5,
            label="Crossover")
ax_lin.plot(t_asym_lin, a * t_asym_lin**ALPHA, **ASYM, label="Asymptotes")
ax_lin.plot(t_asym_lin, b * t_asym_lin**BETA, **ASYM)
ax_lin.axvline(t_star, color="0.1", ls="--", lw=0.8)
ax_lin.annotate(rf"$t_* = {t_star:.1f}$ d", (t_star, 60), color="0.1",
                ha="center", backgroundcolor="1")
ax_lin.annotate(r"$\propto t^{1/2}$", (17, a * 17**ALPHA), xytext=(0, -6),
                ha="center", va="top", **LABEL)
ax_lin.annotate(r"$\propto t^{7/8}$", (17, b * 17**BETA), xytext=(-6, 0),
                ha="right", va="bottom", **LABEL)

ax_lin.set(xlim=(0, 20), ylim=(0, 1200),
           xlabel="Time since first event (days)", ylabel="Distance (m)")
ax_lin.legend(loc="upper left", frameon=False, fontsize="small")

fig_lin.canvas.draw_idle()
plt.pause(0.01)

# %%

# plotting.save_figure(fig_lin, "soultz-radial-crossover-linear", ext="pdf")
