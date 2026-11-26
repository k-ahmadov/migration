"""Basel 2006 wellhead pressure vs the 1D ramp model: a rough check.

Basel counterpart of ``explore/radial/soultz_whp.py``, compared with 1D (plane)
flow. For a linear ramp the 1D injection pressure grows as (``fracinj.analysis``
table, p_inj = (1 + 2i)/(n + 2) with i = 1)

    rigid (t << t_*):  p ~ t^(3/2)
    soft  (t >> t_*):  p ~ t^(3/5)

(no log factors in 1D, unlike the radial case). Here the wellhead pressure
(WHP) is fitted with a power law in an early and a late window chosen from the
WHP itself, with time counted from the start of injection; the t_* found from
the seismicity front (``basel_crossover.py``, ~3.2 d) is only shown for
reference. It need not match, and does not: by the seismic t_* most of the late
window would still be rigid (t^(3/2)), but the WHP already grows as ~t^(3/5).

The WHP record has three parts:
  * t < 0.3 d: steep rise to ~110 bar at a tiny rate (< 2 l/s) -> early window;
  * 0.3-0.75 d: decline at constant rate and a brief stop -> not fitted;
  * 0.75 d to the rate cut (~55 -> ~31 l/s, ~5.2 d): slow growth -> late window.
After the rate cut the pressure falls (not a ramp), so it is not fitted.

Caveats, not modelled: WHP is not the downhole overpressure (no hydrostatic
or friction correction; friction grows with rate, so the downhole pressure
grows even more slowly); the rate is a staircase, not a ramp, so each step adds
a fast pressure jump; the data are digitized from Haring et al. (2008, Fig. 5),
~18 min and ~1.3 bar resolution; brief stops are not removed (only samples
with zero rate are dropped), so the pressure dips at stops enter the fits.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from fracinj import paths, plotting  # noqa: F401

HYDRAULIC = paths.DATA / "basel1_2006_digitized_WHP_and_rate.csv"
T_STAR_SEISMIC = 3.15  # days since injection start, from basel_crossover.py
EARLY = (0.0, 0.3)  # days since injection start: steep early rise
LATE_START = 0.75  # days since injection start: slow growth from here
RATE_CUT_FRACTION = 0.7  # rate cut = first drop below this x peak rate

# %% data: time since injection start, injection phase only, drop zero-rate samples

hyd = pd.read_csv(HYDRAULIC, comment="#", parse_dates=["datetime"])
t = hyd["hours_since_injection_start"].to_numpy() / 24
q = hyd["flow_rate_Lpm"].to_numpy() / 60  # l/s
whp = hyd["whp_bar"].to_numpy()
injection_start = (
    hyd["datetime"] - pd.to_timedelta(hyd["hours_since_injection_start"], unit="h")
).median()

injecting = (hyd["phase"] == "injection").to_numpy() & (t > 0) & (q > 0)
i_peak = np.argmax(np.where(injecting, q, 0))
rate_cut = t[i_peak:][q[i_peak:] < RATE_CUT_FRACTION * q[i_peak]].min()

print(f"injection start = {injection_start:%Y-%m-%d %H:%M:%S} local")
print(f"rate cut = {rate_cut:.2f} d, seismic t_* = {T_STAR_SEISMIC:.2f} d since start")

# %% power-law fits in the early and late windows

windows = {
    "early": (t > EARLY[0]) & (t < EARLY[1]),
    "late": (t >= LATE_START) & (t < rate_cut),
}
fits = {}
for name, window in windows.items():
    m = injecting & window
    slope, log_pref = np.polyfit(np.log(t[m]), np.log(whp[m]), 1)
    fits[name] = (slope, np.exp(log_pref), t[m].min(), t[m].max())
    print(f"{name:>5}: WHP ~ t^{slope:.2f}   ({t[m].min():.2f}-{t[m].max():.2f} d since start)")

# %% figure

CW, GR = 3.25, 1.618
fig = plt.figure(figsize=(CW, 1.5 * CW / GR), dpi=300, layout="constrained",
                 clear=True, num=1)
ax_q, ax = fig.subplots(2, 1, sharex=True, height_ratios=(1, 3))

ax_q.plot(t[t > 0], q[t > 0], color="k", lw=0.8)
ax_q.set(ylabel="$Q$ (l/s)", ylim=(0, 65))

ax.plot(t[injecting], whp[injecting], ".", ms=1.5, color="0.6", rasterized=True,
        label="WHP")
# model slopes for comparison, anchored to the (not drawn) fits at the window
# midpoints; the fitted exponents are printed above
REF = {"color": "r", "ls": (0, (4, 2)), "lw": 1.2}
# label each line at its midpoint (early: to the left, the line leaves the axes)
for name, model_slope, text, dx, ha in [("early", 1.5, r"$t^{3/2}$", -4, "right"),
                                        ("late", 0.6, r"$t^{3/5}$", 4, "left")]:
    slope, pref, t0, t1 = fits[name]
    tm = np.sqrt(t0 * t1)
    tt = np.geomspace(t0, t1, 50)
    ax.plot(tt, pref * tm**slope * (tt / tm) ** model_slope, **REF,
            label="1D model" if name == "early" else None)
    ax.annotate(text, (tm, pref * tm**slope), xytext=(dx, -8),
                textcoords="offset points", ha=ha, va="center", color="r")

ax.axvline(T_STAR_SEISMIC, color="0.1", ls="--", lw=0.8)
ax.annotate(rf"$t_* = {T_STAR_SEISMIC:.2f}$ d", (T_STAR_SEISMIC, 15), color="0.1", ha="right",
            backgroundcolor="1")

ax.set(xscale="log", yscale="log", xlim=(0.005, 8), ylim=(5, 400),
       xlabel="Time since injection start (days)", ylabel="Wellhead pressure (bar)")
ax.set_xticks([0.01, 0.1, 1, 5])
ax.set_yticks([10, 30, 100, 300])
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_formatter(mticker.ScalarFormatter())
    axis.set_minor_formatter(mticker.NullFormatter())
ax.legend(loc="upper left", frameon=False, fontsize="small")

fig.canvas.draw_idle()
plt.pause(0.01)

# %%

# plotting.save_figure(fig, "basel-whp", ext="pdf")
