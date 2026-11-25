"""1D ramp crossover model fitted to the Basel 2006 seismicity front.

Basel counterpart of ``explore/radial/soultz_crossover.py``, compared with the
1D (plane) flow instead of the radial one. The 2006 Basel-1 stimulation raised
the rate in steps from ~0.2 to ~55 l/s over the first ~5 days, i.e. roughly a
linear ramp. In 1D the ramp front migrates as (``fracinj.analysis`` table)

    rigid (t << t_*):  ell ~ a t^(1/2)
    soft  (t >> t_*):  ell ~ b t^(7/5)

The crossover uses the same form and sharpness (delta = 1) as the Soultz fit:
t_* is not imposed but placed where the asymptotes meet,
t_* = (a/b)^(1/(7/5 - 1/2)) = (a/b)^(10/9), so only a, b are fitted.

The seismicity front is the 75th-percentile distance per bin of 50 events (as
for Basel in ``scripts/fig5_seismicity.py``), distance from the median of the
first 10 events (``io.load("basel")``). Only events up to the rate cut
(~55 -> ~31 l/s, ~6 h before shut-in) are fitted, since the ramp model does not
describe a rate reduction; later events are shown but not used.

Two choices keep t_* free of arbitrary settings:
  * time is counted from the start of injection (the model's t = 0), not from
    the first event, which came ~5 h (0.21 d) later;
  * where the first 50-event bin starts is arbitrary, so the fit is repeated
    for all 50 bin start offsets and t_* is the median over them. The curve
    drawn is the offset whose t_* is closest to that median.
The uncertainty comes from a bootstrap (resampled events, random offset).

Times: the catalogue is in UTC, the digitized hydraulic record (Haring et al.
2008, Fig. 5) in local time (CET = UTC+1); everything here is local time.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from fracinj import detection, io, math_utils, paths, plotting  # noqa: F401

DAY = 86_400
ALPHA, BETA = 1 / 2, 7 / 5  # 1D ramp front exponents
DELTA = 1.0  # crossover sharpness, as for the simulated front
PERCENTILE, BIN_SIZE = 75, 50

HYDRAULIC = paths.DATA / "basel1_2006_digitized_WHP_and_rate.csv"
# first event used by io.load("basel"): 2006-12-02T22:05:17.608 UTC
CATALOGUE_T0 = pd.Timestamp("2006-12-02 22:05:17.608") + pd.Timedelta(hours=1)

# %% data: all times in days since the start of injection

hyd = pd.read_csv(HYDRAULIC, comment="#", parse_dates=["datetime"])
t_q = hyd["hours_since_injection_start"].to_numpy() / 24
q = hyd["flow_rate_Lpm"].to_numpy() / 60  # l/s
injection_start = (
    hyd["datetime"] - pd.to_timedelta(hyd["hours_since_injection_start"], unit="h")
).median()
T_FIRST = (CATALOGUE_T0 - injection_start).total_seconds() / DAY  # ~0.21 d

t_ev, r_ev = io.load("basel")
t_ev = t_ev / DAY + T_FIRST

is_injection = (hyd["phase"] == "injection").to_numpy()
SHUT_IN = t_q[(hyd["phase"] == "shut-in").to_numpy()].min()
i_peak = np.argmax(np.where(is_injection, q, 0))
RAMP_END = t_q[i_peak]  # peak rate
# rate cut: first drop below 70% of the peak rate (as in basel_whp.py)
RATE_CUT = t_q[i_peak:][q[i_peak:] < 0.7 * q[i_peak]].min()
print(f"first event = {T_FIRST:.2f} d, ramp end = {RAMP_END:.2f} d, "
      f"rate cut = {RATE_CUT:.2f} d, shut-in = {SHUT_IN:.2f} d since injection start")

injecting = t_ev < RATE_CUT
t_inj, r_inj = t_ev[injecting], r_ev[injecting]

# %% fit: a t^1/2 -> b t^7/5, crossing over where the asymptotes meet


def crossover_time(a, b):
    return (a / b) ** (1 / (BETA - ALPHA))


def front_model(t, a, b):
    return math_utils.crossover(
        t, a, ALPHA, b, BETA, x0=crossover_time(a, b), delta=DELTA
    )


def envelope(t, r, offset=0):
    """Percentile envelope with the bin boundaries shifted by ``offset`` events.

    The first bin absorbs the shift (BIN_SIZE + offset events), so no early
    event is dropped; the later bins hold BIN_SIZE events each.
    """
    n0 = BIN_SIZE + offset
    t0, r0 = detection.percentile_envelope(t[:n0], r[:n0], bin_size=n0,
                                           percentile=PERCENTILE)
    t1, r1 = detection.percentile_envelope(t[n0:], r[n0:], bin_size=BIN_SIZE,
                                           percentile=PERCENTILE)
    return np.concatenate([t0, t1]), np.concatenate([r0, r1])


def fit_front(t_env, r_env, p0=(np.log(200.0), np.log(100.0))):
    """Fit log a, log b in log space, so every decade of distance weighs the same."""
    return curve_fit(
        lambda t, la, lb: np.log(front_model(t, np.exp(la), np.exp(lb))),
        t_env, np.log(r_env), p0=p0,
    )


# fit every bin start offset; t_* is the median over offsets
fits = []
for offset in range(BIN_SIZE):
    (la, lb), cov = fit_front(*envelope(t_inj, r_inj, offset))
    fits.append((crossover_time(np.exp(la), np.exp(lb)), offset, la, lb, cov))
t_star_offsets = np.array([f[0] for f in fits])
t_star_median = np.median(t_star_offsets)
print(f"t_* over {BIN_SIZE} bin offsets: median {t_star_median:.2f} d, "
      f"range {t_star_offsets.min():.2f}-{t_star_offsets.max():.2f} d")

# representative offset: the one whose t_* is closest to the median
t_star, offset, log_a, log_b, cov = min(fits, key=lambda f: abs(f[0] - t_star_median))
t_env, r_env = envelope(t_inj, r_inj, offset)
a, b = np.exp(log_a), np.exp(log_b)
err_log_a, err_log_b = np.sqrt(np.diag(cov))
resid = np.log(r_env) - np.log(front_model(t_env, a, b))

print(f"representative offset = {offset}")
print(f"a   = {a:.0f} m/day^1/2   (+-{100 * err_log_a:.0f}%)")
print(f"b   = {b:.0f} m/day^7/5   (+-{100 * err_log_b:.0f}%)")
print(f"t_* = {t_star:.2f} days since injection start")
print(f"rms misfit (log) = {np.sqrt(np.mean(resid**2)):.3f}")

# %% bootstrap uncertainty of t_*
# The curve_fit error treats the envelope points as independent, but they all
# come from one catalogue. Instead: resample the events with replacement, pick
# a random bin start offset, rebuild the envelope, refit, and take the spread
# of the resulting t_* values.

N_BOOT = 500
rng = np.random.default_rng(0)
t_star_boot = []
for _ in range(N_BOOT):
    i = np.sort(rng.integers(0, len(t_inj), len(t_inj)))  # keep time order
    try:
        (la, lb), _ = fit_front(
            *envelope(t_inj[i], r_inj[i], int(rng.integers(BIN_SIZE))), p0=(log_a, log_b)
        )
    except RuntimeError:  # fit did not converge for this resample
        continue
    t_star_boot.append(crossover_time(np.exp(la), np.exp(lb)))
t_star_boot = np.array(t_star_boot)
q16, q50, q84, q025, q975 = np.percentile(t_star_boot, [16, 50, 84, 2.5, 97.5])
print(f"bootstrap ({len(t_star_boot)}/{N_BOOT} fits): t_* median = {q50:.2f} d, "
      f"68%: {q16:.2f}-{q84:.2f} d, 95%: {q025:.2f}-{q975:.2f} d")
print(f"fraction of t_* before the rate cut: {np.mean(t_star_boot < RATE_CUT):.2f}")
ramp = is_injection & (t_q < RAMP_END)
m_q = np.polyfit(t_q[ramp], q[ramp], 1)[0]
print(f"mean ramp slope = {m_q:.2f} l/s/day = {m_q * 1e-3 / DAY:.2e} m^3/s^2")

T_MIN, T_MAX = 0.7, 6.2  # from just before the first envelope point (~0.95 d) to just past shut-in
t_fit = np.geomspace(t_env[0], RATE_CUT, 200)
t_asym = np.geomspace(T_MIN, T_MAX, 200)  # asymptotes across the whole axis

# %% figure

CW, GR = 3.25, 1.618
fig = plt.figure(figsize=(CW, 1.5 * CW / GR), dpi=300, layout="constrained",
                 clear=True, num=1)
ax_q, ax = fig.subplots(2, 1, sharex=True, height_ratios=(1, 3))

ax_q.plot(t_q[t_q > 0], q[t_q > 0], color="k", lw=0.8)
ax_q.axvspan(RATE_CUT, T_MAX, color="0.93", lw=0)  # not fitted
ax_q.set(ylabel="$Q$ (l/s)", ylim=(0, 65))

ax.axvspan(RATE_CUT, T_MAX, color="0.93", lw=0)  # not fitted
ax.scatter(t_ev, r_ev, s=2, color="0.7", rasterized=True, label="Events")
ENV_LABEL = rf"Envelope ({PERCENTILE}\%)"
ax.scatter(t_env, r_env, s=10, color="k", label=ENV_LABEL)
ax.plot(t_fit, front_model(t_fit, a, b), color="C0", lw=1.5, label="Crossover")
ASYM = {"color": "r", "ls": (0, (4, 2)), "lw": 1.3, "zorder": 3}
LABEL = {"textcoords": "offset points", "color": "k", "zorder": 4,
         "bbox": {"facecolor": "white", "edgecolor": "none", "pad": 0.5}}
ax.plot(t_asym, a * t_asym**ALPHA, **ASYM, label="Asymptotes")
ax.plot(t_asym, b * t_asym**BETA, **ASYM)
ax.axvline(t_star, color="0.1", ls="--", lw=0.8)
ax.annotate(rf"$t_* = {t_star:.2f}$ d", (t_star, 40), color="0.1",
            ha="center", backgroundcolor="1")
ax.annotate(r"$\propto t^{1/2}$", (0.75, a * 0.75**ALPHA), xytext=(0, -6),
            ha="left", va="top", **LABEL)
ax.annotate(r"$\propto t^{7/5}$", (2.0, b * 2.0**BETA), xytext=(6, -6),
            ha="left", va="top", **LABEL)

ax.set(xscale="log", yscale="log", xlim=(T_MIN, T_MAX), ylim=(30, 800),
       xlabel="Time since injection start (days)", ylabel="Distance (m)")
ax.set_yticks([50, 100, 200, 500])
ax.set_xticks([0.7, 1, 2, 3, 5])
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_formatter(mticker.ScalarFormatter())
    axis.set_minor_formatter(mticker.NullFormatter())
ax.legend(loc="upper left", frameon=False, fontsize="small")

fig.canvas.draw_idle()
plt.pause(0.01)

# %%

# plotting.save_figure(fig, "basel-1d-crossover", ext="pdf")

# %% same figure, linear axes

t_fit_lin = np.linspace(t_env[0], RATE_CUT, 200)
t_asym_lin = np.linspace(0, T_MAX, 200)

fig_lin = plt.figure(figsize=(CW, 1.5 * CW / GR), dpi=300, layout="constrained",
                     clear=True, num=2)
ax_q_lin, ax_lin = fig_lin.subplots(2, 1, sharex=True, height_ratios=(1, 3))

ax_q_lin.plot(t_q[t_q > 0], q[t_q > 0], color="k", lw=0.8)
ax_q_lin.axvspan(RATE_CUT, T_MAX, color="0.93", lw=0)  # not fitted
ax_q_lin.set(ylabel="$Q$ (l/s)", ylim=(0, 65))

ax_lin.axvspan(RATE_CUT, T_MAX, color="0.93", lw=0)  # not fitted
ax_lin.scatter(t_ev, r_ev, s=2, color="0.7", rasterized=True, label="Events")
ax_lin.scatter(t_env, r_env, s=10, color="k", label=ENV_LABEL)
ax_lin.plot(t_fit_lin, front_model(t_fit_lin, a, b), color="C0", lw=1.5,
            label="Crossover")
ax_lin.plot(t_asym_lin, a * t_asym_lin**ALPHA, **ASYM, label="Asymptotes")
ax_lin.plot(t_asym_lin, b * t_asym_lin**BETA, **ASYM)
ax_lin.axvline(t_star, color="0.1", ls="--", lw=0.8)
ax_lin.annotate(rf"$t_* = {t_star:.2f}$ d", (t_star, 660), color="0.1",
                ha="center", backgroundcolor="1")
ax_lin.annotate(r"$\propto t^{1/2}$", (1.5, a * 1.5**ALPHA), xytext=(0, 6),
                ha="center", va="bottom", **LABEL)
ax_lin.annotate(r"$\propto t^{7/5}$", (3.5, b * 3.5**BETA), xytext=(6, -6),
                ha="left", va="top", **LABEL)

ax_lin.set(xlim=(0, T_MAX), ylim=(0, 700),
           xlabel="Time since injection start (days)", ylabel="Distance (m)")
ax_lin.legend(loc="upper left", frameon=False, fontsize="small")

fig_lin.canvas.draw_idle()
plt.pause(0.01)

# %%

# plotting.save_figure(fig_lin, "basel-1d-crossover-linear", ext="pdf")
