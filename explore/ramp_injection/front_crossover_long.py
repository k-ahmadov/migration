import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from fracinj import analysis, io, math_utils, paths, physics, plotting

MARKERS = ["o", "s", "^", "D", "v", "p", "<", ">"]

# %%


def marker_style(i, p):
    """Marker style for run i, matching dimensionless.py and front_crossover.py."""
    return {
        "marker": MARKERS[i % len(MARKERS)],
        "mfc": "k" if i % 2 == 0 else "none",
        "mec": "k",
        "mew": 0.7,
        "ms": 3.5,
        "ls": "none",
        "markevery": 0.07,
        "label": rf"${plotting.sci_latex(p.m_q)}$",
    }


# %%

CW, GR = 3.25, 1.618

# Same figure for one long run (L = 3 km) that spans both regimes on its own.
# Asymptotes: each regime's exponent with its prefactor fitted on its own
# window (t < t_c rigid, t > t_c soft). Crossover: the same two-power-law join
# (same exponents, x0 = 1, delta) with both prefactors fitted to this run
# (the pooled figure's prefactors are hand-tuned).
run_long = io.read_hdf5(paths.results_dir("halfspace", "ramp", "long") / "m_q-1e-13.hdf5")
sc_long = physics.ramp_rate_fixed_aperture_cubic(run_long.params)
front_long = analysis.analyze_front(run_long, stress_front=True)
early_long = analysis.front_early_ramp(run_long, stress_front=True)
late_long = analysis.front_late_ramp(run_long, stress_front=True)

valid = front_long.t > 0
t_long = front_long.t[valid] / sc_long.t_c
x_long = front_long.y[valid] / sc_long.x_c
t_nd_long = np.geomspace(t_long.min(), t_long.max(), 200)
rigid_long = early_long.analytical(t_nd_long * sc_long.t_c) / sc_long.x_c
soft_long = late_long.analytical(t_nd_long * sc_long.t_c) / sc_long.x_c

delta = 5e-2
# initial guess: the asymptotes' prefactors in dimensionless units
p0 = [
    early_long.A_ana * sc_long.t_c**early_long.alpha_ana / sc_long.x_c,
    late_long.A_ana * sc_long.t_c**late_long.alpha_ana / sc_long.x_c,
]
# fit in log space on log-spaced samples: linear residuals on the raw samples
# (mostly late, large fronts) would ignore the early regime


def log_crossover(t, log_a, log_b):
    return np.log(math_utils.crossover(
        t, a=np.exp(log_a), alpha=early_long.alpha_ana, b=np.exp(log_b),
        beta=late_long.alpha_ana, x0=1, delta=delta,
    ))


i_fit = np.unique(np.searchsorted(t_long, np.geomspace(t_long[0], t_long[-1], 100)))
i_fit = i_fit[i_fit < len(t_long)]
(log_a, log_b), _ = curve_fit(log_crossover, t_long[i_fit], np.log(x_long[i_fit]), p0=np.log(p0))
a_long, b_long = np.exp(log_a), np.exp(log_b)
crossover_long = math_utils.crossover(
    x=t_nd_long, a=a_long, alpha=early_long.alpha_ana, b=b_long,
    beta=late_long.alpha_ana, x0=1, delta=delta,
)

fig_long = plt.figure(
    figsize=(CW, CW / GR),
    dpi=150,
    layout="constrained",
    clear=True,
    num=3,
)
ax = fig_long.subplots()
# marker of m_q = 1e-13 in the pooled figure of front_crossover.py (index 1)
(h_long,) = ax.plot(
    t_long, x_long, **(marker_style(1, run_long.params) | {"label": "Numerical"})
)
(l_cross,) = ax.plot(
    t_nd_long, crossover_long, "-", color="C0", lw=1.5, label="Crossover"
)
(l_asym,) = ax.plot(
    t_nd_long, rigid_long, color="tab:gray", ls=":", lw=1.2, label="Asymptotes"
)
ax.plot(t_nd_long, soft_long, color="tab:gray", ls=":", lw=1.2)
ax.annotate(
    r"$\eta_{\mathrm{f}}\, 2 \sqrt{ D t }$",
    xy=(t_nd_long[50], rigid_long[50]),
    xytext=(25, 20),
    textcoords="offset points",
    ha="right",
    arrowprops={"arrowstyle": "<-"},
)
ax.annotate(
    r"$\zeta_{\mathrm{f}} \left( a m_{\mathrm{q}}^{3} t^{7} \right)^{1/5}  $",
    xy=(t_nd_long[-30], soft_long[-30]),
    xytext=(-5, -50),
    textcoords="offset points",
    ha="center",
    va="top",
    arrowprops={"arrowstyle": "<-"},
)
ax.axvline(1, color="tab:gray", ls="--")
ax.annotate(
    r"$\hat{t}=1$",
    (1, 5e1),
    color="tab:gray",
    ha="center",
    backgroundcolor="1",
)
ax.set(
    xlabel=r"$\hat{t}$",
    ylabel=r"$\ell / x_{*}$",
    xscale="log",
    yscale="log",
    ylim=(x_long.min() / 2, x_long.max() * 2),
)
ax.legend(handles=[h_long, l_cross, l_asym], loc="upper left")

fig_long.canvas.draw_idle()
plt.pause(0.01)

# %%

plotting.save_figure(fig_long, "ramp-front-dimensionless-long", ext="pdf")
