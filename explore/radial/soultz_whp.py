"""Soultz 1993 wellhead pressure vs the radial ramp model: a rough check.

The radial ramp model predicts an injection pressure growing as

    rigid (t << t_*):  p ~ t       (times a log factor, local slope ~1.0)
    soft  (t >> t_*):  p ~ t^(1/4) (times a log factor, local slope ~0.3)

(see ``front_dimensionless.py``). Here the wellhead pressure (WHP) is fitted
with a power law in an early and a late window chosen from the WHP itself, with
time counted from the start of injection (first rate above 0.05 l/s,
1993-09-01 13:08 local); the t_* found from the seismicity front
(``soultz_crossover.py``: 9.46 d since the first event, fitted with t = 0 at
the first event) is only shown for reference, shifted to this origin.

Caveats, not modelled: WHP is not the downhole overpressure (friction losses
grow with rate, so the downhole pressure grows even more slowly); the rate is a
staircase, not a ramp, so each step adds a fast pressure jump; the early slope
depends on the chosen injection start; brief stops are not removed (only
samples with zero rate are dropped), so the pressure dips at stops enter the fits.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from fracinj import paths, plotting  # noqa: F401

HYDRAULIC = (
    paths.DATA / "SSFS1993-Hydraulic" / "SSFS1993-Hydraulic_september"
    / "SSFS1993-Hydraulic_september.csv"
)
CATALOGUE_T0 = pd.Timestamp("1993-09-02 16:19:11.72")  # first event, local time
Q_START = 0.05  # l/s: injection starts when the rate first exceeds this
T_STAR_FIRST_EVENT = 9.46  # days since the first event, from soultz_crossover.py
T_SOFT = 1.3  # days since injection start: WHP growth slows down (soft-like) from here

# %% data: time since injection start, drop shut-in and zero-rate samples

hyd = pd.read_csv(HYDRAULIC, sep=";", parse_dates=["datetime"])
q = hyd["q_in_comb_max (l/s)"].to_numpy()
whp = hyd["p_wellhead (bar)"].to_numpy()
injection_start = hyd["datetime"][q > Q_START].iloc[0]
t = (hyd["datetime"] - injection_start).dt.total_seconds().to_numpy() / 86_400
T_FIRST = (CATALOGUE_T0 - injection_start).total_seconds() / 86_400  # ~1.13 d
SHUT_IN = 15.14 + T_FIRST  # days since injection start
T_STAR_SEISMIC = T_STAR_FIRST_EVENT + T_FIRST  # days since injection start

injecting = (t > 0) & (t < SHUT_IN) & (q > Q_START) & (whp > 1)

print(f"injection start = {injection_start:%Y-%m-%d %H:%M} local "
      f"({T_FIRST:.2f} d before the first event)")
print(f"shut-in = {SHUT_IN:.2f} d, seismic t_* = {T_STAR_SEISMIC:.2f} d since start")

# %% power-law fits before and after T_SOFT (where the slow WHP growth starts);
# the seismic t_* is only shown for reference

windows = {"early": (t > 0) & (t < T_SOFT), "late": (t >= T_SOFT)}
fits = {}
for name, window in windows.items():
    m = injecting & window
    slope, log_pref = np.polyfit(np.log(t[m]), np.log(whp[m]), 1)
    fits[name] = (slope, np.exp(log_pref), t[m].min(), t[m].max())
    print(f"{name:>5}: WHP ~ t^{slope:.2f}   ({t[m].min():.1f}-{t[m].max():.1f} d since start)")

# %% figure

CW, GR = 3.25, 1.618
fig = plt.figure(figsize=(CW, 1.5 * CW / GR), dpi=300, layout="constrained",
                 clear=True, num=1)
ax_q, ax = fig.subplots(2, 1, sharex=True, height_ratios=(1, 3))

ax_q.plot(t[t > 0], q[t > 0], color="k", lw=0.8)
ax_q.set(ylabel="$Q$ (l/s)", ylim=(0, 45))

ax.plot(t[injecting], whp[injecting], ".", ms=1.5, color="0.6", rasterized=True,
        label="WHP")
# model slopes for comparison, anchored to the (not drawn) fits at the window
# midpoints; the fitted exponents are printed above
REF = {"color": "r", "ls": (0, (4, 2)), "lw": 1.2}
for name, model_slope, text in [("early", 1.0, r"$t^{1}$"), ("late", 0.25, r"$t^{1/4}$")]:
    slope, pref, t0, t1 = fits[name]
    tm = np.sqrt(t0 * t1)
    tt = np.geomspace(t0, t1, 50)
    ax.plot(tt, pref * tm**slope * (tt / tm) ** model_slope, **REF,
            label="Radial model" if name == "early" else None)
    ax.annotate(text, (t1, pref * tm**slope * (t1 / tm) ** model_slope),
                xytext=(-10, 10), textcoords="offset points", ha="left", va="center",
                color="r")

ax.axvline(T_STAR_SEISMIC, color="0.1", ls="--", lw=0.8)
ax.annotate(rf"$t_* = {T_STAR_SEISMIC:.2f}$ d", (T_STAR_SEISMIC, 35), xytext=(-3, 0),
            textcoords="offset points", color="0.1", ha="right", backgroundcolor="1")

ax.set(xscale="log", yscale="log", xlim=(0.2, 20), ylim=(7, 130),
       xlabel="Time since injection start (days)", ylabel="Wellhead pressure (bar)")
ax.set_xticks([0.15, 0.5, 1, 2, 5, 10, 20])
ax.set_yticks([50, 80, 120])
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    axis.set_minor_formatter(mticker.NullFormatter())
ax.legend(loc="lower right", frameon=False, fontsize="small")

fig.canvas.draw_idle()
plt.pause(0.01)
# %%

# plotting.save_figure(fig, "soultz-whp", ext="pdf")
