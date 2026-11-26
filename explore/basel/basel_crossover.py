"""1D ramp front model (sum of asymptotes) fitted to the Basel 2006 seismicity front.

Basel counterpart of ``explore/radial/soultz_crossover.py``, compared with the
1D (plane) flow instead of the radial one. The 2006 Basel-1 stimulation raised
the rate in steps from ~0.2 to ~55 l/s over the first ~5 days, i.e. roughly a
linear ramp. In 1D the ramp front migrates as (``fracinj.analysis`` table)

    rigid (t << t_*):  ell = a t^(1/2),    a = ZETA_F sqrt(D)
    soft  (t >> t_*):  ell = b t^(7/5),    b = ZETA_N x_* / t_*^(7/5)

and, as for the simulated stress front (``scripts/fig4_dimensionless.py``),
the whole evolution follows their sum, ell = a t^(1/2) + b t^(7/5), within
~10%. The model is linear in a, b, so they are fitted by non-negative linear
least squares on the relative misfit (close to a log-space fit, with a unique
solution and no starting guess). The asymptotes cross at
t_x = (a/b)^(10/9) (``crossover_time``); with the analytical prefactors
ZETA_F = 1.343 (``closed_form.rigid_stress_front``) and ZETA_N = 0.588
(``closed_form.front_approximation``) the physical crossover time is

    t_* = t_x / (ZETA_F / ZETA_N)^(10/9) = t_x / 2.50.

A constant scale factor on the envelope (percentile choice, distance
reference) changes a and b alike and leaves t_* unchanged.

The seismicity front is the 75th-percentile distance per bin of 50 events (as
for Basel in ``scripts/fig5_seismicity.py``), distance from the median of the
first 10 events (``io.load("basel")``, times since injection start). Only events up to the rate cut
(~55 -> ~31 l/s, ~6 h before shut-in) are fitted, since the ramp model does not
describe a rate reduction; later events are shown but not used.

Time is counted from the start of injection (the model's t = 0), not from
the first event, which came ~5 h (0.21 d) later. The injection start is
2006-12-02 18:00 local = 17:00 UTC (``io.BASEL_INJECTION_START_UTC``);
``io.load("basel")`` returns event times from it, and the hydraulic record
gives hours since the same start (17:59:30 local implied, within 1 min).

Uncertainty of t_* (not computed here): ``fracinj.bootstrap``.

Times: the catalogue is in UTC, the digitized hydraulic record (Haring et al.
2008, Fig. 5) in local time (CET = UTC+1); both are used as time since
injection start, so the offset cancels.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from fracinj import detection, io, paths, plotting  # noqa: F401
from fracinj.solvers import closed_form

DAY = 86_400
ALPHA, BETA = 1 / 2, 7 / 5  # 1D ramp front exponents
ZETA_F = closed_form.rigid_stress_front("neumann", i=1)  # rigid stress front
ZETA_N = closed_form.front_approximation(3, "neumann", i=1).zeta_front  # soft front
T_X_OVER_T_STAR = (ZETA_F / ZETA_N) ** (1 / (BETA - ALPHA))  # ~2.50
PERCENTILE, BIN_SIZE = 75, 50

HYDRAULIC = paths.DATA / "basel1_2006_digitized_WHP_and_rate.csv"

# %% data: all times in days since the start of injection

hyd = pd.read_csv(HYDRAULIC, comment="#", parse_dates=["datetime"])
t_q = hyd["hours_since_injection_start"].to_numpy() / 24
q = hyd["flow_rate_Lpm"].to_numpy() / 60  # l/s

t_ev, r_ev = io.load("basel")  # seconds since injection start
t_ev = t_ev / DAY
T_FIRST = t_ev[0]  # ~0.21 d

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

# %% fit: ell = a t^1/2 + b t^7/5 (linear in a, b)


def crossover_time(a, b):
    """Time where the asymptotes a t^1/2 and b t^7/5 cross."""
    return (a / b) ** (1 / (BETA - ALPHA))


def physical_t_star(a, b):
    """Crossover time scale t_* of the model (t_x / 2.50)."""
    return crossover_time(a, b) / T_X_OVER_T_STAR


def front_model(t, a, b):
    return a * t**ALPHA + b * t**BETA


def fit_front(t_env, r_env):
    """(a, b) >= 0 minimising sum((r - a t^1/2 - b t^7/5) / r)^2 (relative misfit)."""
    design = np.column_stack([t_env**ALPHA, t_env**BETA]) / r_env[:, None]
    (a, b), _ = nnls(design, np.ones_like(r_env))
    return a, b


t_env, r_env = detection.percentile_envelope(t_inj, r_inj, bin_size=BIN_SIZE,
                                             percentile=PERCENTILE)
# prepend the origin (injection start at t = 0, distance 0); the model passes
# through it exactly and the relative misfit is undefined there, so it is
# shown but not fitted
t_env, r_env = np.concatenate([[0.0], t_env]), np.concatenate([[0.0], r_env])
fitted = t_env > 0
a, b = fit_front(t_env[fitted], r_env[fitted])
t_x, t_star = crossover_time(a, b), physical_t_star(a, b)
resid = np.log(r_env[fitted]) - np.log(front_model(t_env[fitted], a, b))
D = (a / ZETA_F) ** 2 / DAY  # a = ZETA_F sqrt(D)

print(f"a   = {a:.0f} m/day^1/2  ->  D = (a / {ZETA_F:.3f})^2 = {D:.2g} m^2/s")
print(f"b   = {b:.1f} m/day^7/5")
print(f"t_x = {t_x:.2f} d (asymptotes cross), t_* = t_x / {T_X_OVER_T_STAR:.2f} "
      f"= {t_star:.2f} d since injection start")
print(f"rms misfit (log) = {np.sqrt(np.mean(resid**2)):.3f}")

T_MIN, T_MAX = 0.3, 6.2  # from just before the first envelope point (~0.37 d) to just past shut-in
t_fit = np.geomspace(t_env[fitted][0], RATE_CUT, 200)

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
FIT_LABEL = r"$a t^{1/2} + b t^{7/5}$"
ax.plot(t_fit, front_model(t_fit, a, b), color="C0", lw=1.5, label=FIT_LABEL)
ax.axvline(t_star, color="0.1", ls="--", lw=0.8)
ax.annotate(rf"$t_* = {t_star:.2f}$ d", (t_star, 600), xytext=(-3, 0),
            textcoords="offset points", color="0.1", ha="right",
            backgroundcolor="1")

ax.set(xscale="log", yscale="log", xlim=(T_MIN, T_MAX), ylim=(30, 800),
       xlabel="Time since injection start (days)", ylabel="Distance (m)")
ax.set_yticks([50, 100, 200, 500])
ax.set_xticks([0.3, 0.5, 1, 2, 3, 5])
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_formatter(mticker.ScalarFormatter())
    axis.set_minor_formatter(mticker.NullFormatter())
ax.legend(loc="upper left", frameon=False, fontsize="small")

fig.canvas.draw_idle()
plt.pause(0.01)

# %%

plotting.save_figure(fig, "basel-1d-crossover", ext="pdf")

# %% same figure, linear axes

t_fit_lin = np.linspace(t_env[0], RATE_CUT, 200)

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
            label=FIT_LABEL)
ax_lin.axvline(t_star, color="0.1", ls="--", lw=0.8)
ax_lin.annotate(rf"$t_* = {t_star:.2f}$ d", (t_star, 600), xytext=(3, 0),
                textcoords="offset points", color="0.1", ha="left",
                backgroundcolor="1")

ax_lin.set(xlim=(0, T_MAX), ylim=(0, 700),
           xlabel="Time since injection start (days)", ylabel="Distance (m)")
ax_lin.legend(loc="upper left", frameon=False, fontsize="small")

fig_lin.canvas.draw_idle()
plt.pause(0.01)

# %%

plotting.save_figure(fig_lin, "basel-1d-crossover-linear", ext="pdf")
