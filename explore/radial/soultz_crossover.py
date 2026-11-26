"""Radial ramp front model (sum of asymptotes) fitted to the Soultz 1993 seismicity front.

The 1993 GPK1 stimulation raised the injection rate in steps from ~0.3 to
~37 l/s over the first ~13 days (mean slope ~2.7 l/s/day), i.e. roughly a
linear ramp from a point-like open-hole section: the radial ramp case. Its
front migrates as (``front_dimensionless.py``)

    rigid (t << t_*):  ell = a t^(1/2),    a = ZETA_F sqrt(D)
    soft  (t >> t_*):  ell = b t^(7/8),    b = ZETA_N x_* / t_*^(7/8)

and, as for the simulated stress front (``front_dimensionless.py``), the whole
evolution follows their sum, ell = a t^(1/2) + b t^(7/8). The model is linear
in a, b, so they are fitted by non-negative linear least squares on the
relative misfit (close to a log-space fit, with a unique solution and no
starting guess). The asymptotes cross at t_x = (a/b)^(8/3)
(``crossover_time``); with the analytical prefactors ZETA_F = 1.610
(``closed_form.rigid_stress_front``) and ZETA_N = 1.046
(``closed_form.front_approximation``) the physical crossover time is

    t_* = t_x / (ZETA_F / ZETA_N)^(8/3) = t_x / 3.16.

A constant scale factor on the envelope (percentile choice, distance
reference) changes a and b alike and leaves t_* unchanged.

The seismicity front is the 90th-percentile distance per bin of 50 events
(as in ``scripts/fig5_seismicity.py``). Events up to shut-in are fitted; note
that the last ~3 days before it are at constant rate (~37 l/s), not a ramp.
Post-shut-in events are shown but not used.

Time is counted from the first event (1993-09-02 16:19 local), taken as the
model's t = 0. The hydraulic record starts earlier, with a single 0.05 l/s
sample (-1.13 d), a ~3 h, 3.4 l/s pulse test (-0.94 d) and ~0.15 l/s from
-0.73 d; none of this is a linear ramp, and the t_* found is very sensitive to
where t = 0 is placed within it (0.1-0.6 d for the hydraulic starts), so the
first event is used instead. Both the catalogue and the hydraulic record are in
local time.

Uncertainty of t_* (not computed here): ``fracinj.bootstrap``.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from fracinj import detection, io, paths, plotting
from fracinj.solvers import closed_form

DAY = 86_400
ALPHA, BETA = 1 / 2, 7 / 8  # radial ramp front exponents
ZETA_F = closed_form.rigid_stress_front("neumann", i=1, geometry="radial")
ZETA_N = closed_form.front_approximation(
    3, "neumann", i=1, geometry="radial"
).zeta_front
T_X_OVER_T_STAR = (ZETA_F / ZETA_N) ** (1 / (BETA - ALPHA))  # ~3.16
PERCENTILE, BIN_SIZE = 90, 50

HYDRAULIC = (
    paths.DATA
    / "SSFS1993-Hydraulic"
    / "SSFS1993-Hydraulic_september"
    / "SSFS1993-Hydraulic_september.csv"
)
CATALOGUE_T0 = pd.Timestamp("1993-09-02 16:19:11.72")  # first event, local time
RAMP_END = 12.05  # days since the first event: end of the stepped rate increase
SHUT_IN = 15.14  # days since the first event: injection stops (hydraulic record)

# %% data: all times in days since the first event

hyd = pd.read_csv(HYDRAULIC, sep=";", parse_dates=["datetime"])
q = hyd["q_in_comb_max (l/s)"].to_numpy()
t_q = (hyd["datetime"] - CATALOGUE_T0).dt.total_seconds().to_numpy() / DAY

t_ev, r_ev = io.load("soultz")  # seconds since the first event
t_ev = t_ev / DAY

injecting = t_ev < SHUT_IN
t_inj, r_inj = t_ev[injecting], r_ev[injecting]

# %% fit: ell = a t^1/2 + b t^7/8 (linear in a, b)


def crossover_time(a, b):
    """Time where the asymptotes a t^1/2 and b t^7/8 cross."""
    return (a / b) ** (1 / (BETA - ALPHA))


def physical_t_star(a, b):
    """Crossover time scale t_* of the model (t_x / 3.16)."""
    return crossover_time(a, b) / T_X_OVER_T_STAR


def front_model(t, a, b):
    return a * t**ALPHA + b * t**BETA


def fit_front(t_env, r_env):
    """(a, b) >= 0 minimising sum((r - a t^1/2 - b t^7/8) / r)^2 (relative misfit)."""
    design = np.column_stack([t_env**ALPHA, t_env**BETA]) / r_env[:, None]
    (a, b), _ = nnls(design, np.ones_like(r_env))
    return a, b


t_env, r_env = detection.percentile_envelope(
    t_inj, r_inj, bin_size=BIN_SIZE, percentile=PERCENTILE
)
t_env, r_env = np.asarray(t_env), np.asarray(r_env)
t_env, r_env = t_env[t_env > 0], r_env[t_env > 0]
# prepend the origin (first event at t = 0, distance 0); the model passes
# through it exactly and the relative misfit is undefined there, so it is
# shown but not fitted
t_env, r_env = np.concatenate([[0.0], t_env]), np.concatenate([[0.0], r_env])
fitted = t_env > 0
a, b = fit_front(t_env[fitted], r_env[fitted])
t_x, t_star = crossover_time(a, b), physical_t_star(a, b)
resid = np.log(r_env[fitted]) - np.log(front_model(t_env[fitted], a, b))
D = (a / ZETA_F) ** 2 / DAY  # a = ZETA_F sqrt(D)

print(f"a   = {a:.0f} m/day^1/2  ->  D = (a / {ZETA_F:.3f})^2 = {D:.2g} m^2/s")
print(f"b   = {b:.1f} m/day^7/8")
print(
    f"t_x = {t_x:.2f} d (asymptotes cross), t_* = t_x / {T_X_OVER_T_STAR:.2f} "
    f"= {t_star:.2f} d since the first event"
)
print(f"rms misfit (log) = {np.sqrt(np.mean(resid**2)):.3f}")
ramp = (t_q > 0) & (t_q < RAMP_END)
m_q = np.polyfit(t_q[ramp], q[ramp], 1)[0]
print(f"mean ramp slope = {m_q:.2f} l/s/day = {m_q * 1e-3 / DAY:.2e} m^3/s^2")

# %% fracture parameters from a, b and the ramp slope
# Radial scales (front_dimensionless.py): k = k_n / (12 mu), D = k w_i^3,
# m_pq = m_q / (2 pi), t_* = k w_i^4 / m_pq, x_* = sqrt(D t_*). Then
#     a = ZETA_F sqrt(D),    b = ZETA_N x_* / t_*^(7/8) = ZETA_N k^(1/8) m_pq^(3/8):
# w_i cancels in b, so the front fixes D and the product k m_pq^3 only.
# (i) one fracture takes the whole injected rate: k = (b/ZETA_N)^8 / m_pq^3,
#     w_i = (D/k)^(1/3). This gives k_n ~ 1 Pa/m and w_i ~ 9 cm: unphysical.
# (ii) k_n taken from the literature (K_N_REF): w_i = (D/k)^(1/3), and b then
#     fixes the effective ramp slope driving the front, m_eff = (b/ZETA_N)^(8/3)
#     / k^(1/3), a fraction f = m_eff / m_pq of the injected one (the rest goes
#     to other fractures / storage not in the single-fracture model).
# t_* = D^(4/3) / (b/ZETA_N)^(8/3) is the same in both.

MU = 1e-3  # Pa s, water
K_N_REF = 100e9  # Pa/m, assumed fracture normal stiffness (typical 1-100 GPa/m)

m_pq = m_q * 1e-3 / DAY / (2 * np.pi)  # m^3/s^2
b_si = b * DAY ** (-BETA)  # m / s^(7/8)

k_single = (b_si / ZETA_N) ** 8 / m_pq**3  # 1/(m s)
w_i_single = (D / k_single) ** (1 / 3)
print(
    f"(i) whole rate in one fracture: k_n = 12 mu (b/ZETA_N)^8 / m_pq^3 = "
    f"{12 * MU * k_single:.2g} Pa/m, w_i = {w_i_single * 1e2:.1f} cm"
)

k_ref = K_N_REF / (12 * MU)
w_i_ref = (D / k_ref) ** (1 / 3)
m_eff = (b_si / ZETA_N) ** (8 / 3) / k_ref ** (1 / 3)
print(
    f"(ii) k_n = {K_N_REF / 1e9:g} GPa/m: w_i = (D/k)^(1/3) = {w_i_ref * 1e6:.0f} um, "
    f"effective rate fraction f = m_eff / m_pq = {m_eff / m_pq:.1e}, "
    f"pressure scale k_n w_i = {K_N_REF * w_i_ref / 1e6:.2f} MPa"
)

T_MIN, T_MAX = 0.4, 20.0  # from just before the first envelope point to past shut-in
t_fit = np.geomspace(t_env[fitted][0], SHUT_IN, 200)

# %% figure

CW, GR = 3.25, 1.618
fig = plt.figure(
    figsize=(CW, 1.5 * CW / GR), dpi=300, layout="constrained", clear=True, num=1
)
ax_q, ax = fig.subplots(2, 1, sharex=True, height_ratios=(1, 3))

ax_q.plot(t_q[t_q > 0], q[t_q > 0], color="k", lw=0.8)
ax_q.axvspan(SHUT_IN, T_MAX, color="0.93", lw=0)  # not fitted
ax_q.set(ylabel="$Q$ (l/s)", ylim=(0, 45))

ax.axvspan(SHUT_IN, T_MAX, color="0.93", lw=0)  # not fitted
ax.scatter(t_ev, r_ev, s=2, color="0.7", rasterized=True, label="Events")
ENV_LABEL = rf"Envelope ({PERCENTILE}\%)"
ax.scatter(t_env, r_env, s=10, color="k", label=ENV_LABEL)
FIT_LABEL = r"$a t^{1/2} + b t^{7/8}$"
ax.plot(t_fit, front_model(t_fit, a, b), color="C0", lw=1.5, label=FIT_LABEL)
ax.axvline(t_star, color="0.1", ls="--", lw=0.8)
ax.annotate(
    rf"$t_* = {t_star:.2f}$ d",
    (t_star, 45),
    xytext=(-3, 0),
    textcoords="offset points",
    color="0.1",
    ha="right",
    backgroundcolor="1",
)

ax.set(
    xscale="log",
    yscale="log",
    xlim=(T_MIN, T_MAX),
    ylim=(30, 1500),
    xlabel="Time since first event (days)",
    ylabel="Distance (m)",
)
ax.set_xticks([0.5, 1, 2, 5, 10, 20])
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    axis.set_minor_formatter(mticker.NullFormatter())
ax.set_yticks([50, 100, 200, 500, 1000])
ax.legend(
    loc="upper left", fontsize="small", frameon=True, framealpha=1, edgecolor="none"
)

fig.canvas.draw_idle()
plt.pause(0.01)

# %%

plotting.save_figure(fig, "soultz-radial-crossover", ext="pdf")

# %% same figure, linear axes

t_fit_lin = np.linspace(t_env[0], SHUT_IN, 200)

fig_lin = plt.figure(
    figsize=(CW, 1.5 * CW / GR), dpi=300, layout="constrained", clear=True, num=2
)
ax_q_lin, ax_lin = fig_lin.subplots(2, 1, sharex=True, height_ratios=(1, 3))

ax_q_lin.plot(t_q[t_q > 0], q[t_q > 0], color="k", lw=0.8)
ax_q_lin.axvspan(SHUT_IN, T_MAX, color="0.93", lw=0)  # not fitted
ax_q_lin.set(ylabel="$Q$ (l/s)", ylim=(0, 45))

ax_lin.axvspan(SHUT_IN, T_MAX, color="0.93", lw=0)  # not fitted
ax_lin.scatter(t_ev, r_ev, s=2, color="0.7", rasterized=True, label="Events")
ax_lin.scatter(t_env, r_env, s=10, color="k", label=ENV_LABEL)
ax_lin.plot(
    t_fit_lin, front_model(t_fit_lin, a, b), color="C0", lw=1.5, label=FIT_LABEL
)
ax_lin.axvline(t_star, color="0.1", ls="--", lw=0.8)
ax_lin.annotate(
    rf"$t_* = {t_star:.2f}$ d",
    (t_star, 1100),
    xytext=(3, 0),
    textcoords="offset points",
    color="0.1",
    ha="left",
    backgroundcolor="1",
)

ax_lin.set(
    xlim=(0, T_MAX),
    ylim=(0, 1200),
    xlabel="Time since first event (days)",
    ylabel="Distance (m)",
)
ax_lin.legend(
    loc="upper left", bbox_to_anchor=(0, 0.9), frameon=False, fontsize="small"
)

fig_lin.canvas.draw_idle()
plt.pause(0.01)

# %%

plotting.save_figure(fig_lin, "soultz-radial-crossover-linear", ext="pdf")
