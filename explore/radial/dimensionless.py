"""Dimensionless front position and injection pressure, radial ramp injection.

Radial counterpart of ``scripts/fig4_dimensionless.py``, plus the injection
pressure. Point source with a linear ramp, -r a w^n w_r -> m_q t / (2 pi).

Intrinsic scales (aperture pinned to w_i; set by where the soft n=3 solution
w = A t^(1/4) theta(zeta) reaches w_i):

    t_* = k w_i^4 / m_pq,      x_* = sqrt(D t_*) = k w_i^(7/2) / sqrt(m_pq)

with k = k_n / (12 mu), D = k w_i^3, m_pq = m_q / (2 pi). In these units

    rigid (t << t_*):  ell / x_* = ZETA_F t^(1/2),   p / (k_n w_i) = t phi(.)
    soft  (t >> t_*):  ell / x_* = ZETA_N t^(7/8),   p / (k_n w_i) ~ t^(1/4) theta(.)

Both front prefactors are analytical: ZETA_F = 1.610 is the most tensile point
of the stress from the exact linear pressure (closed_form.rigid_stress_front),
ZETA_N = 1.046 the n=3 front-form approximation. The curve drawn through the
fronts is their sum (no fitting), as in fig4: the stress front overshoots both
asymptotes in the transition, which a weighted blend cannot follow.

The injection pressure is read at node 0. A point source makes the pressure
log-singular, so p_inj only exists at a finite radius r_w and carries log
factors in rho = r_w / x_*. The source cell's stencil (face transmissibility
r_f/dr at r_f = dr/2) acts as a well of radius r_w = e^-2 dr. At r_w, using
E1(s) ~ -gamma - ln s and theta^4 ~ C - 4 ln zeta for small arguments:

    rigid:  p / (k_n w_i) = (t/2) [ln(4 t / rho^2) - gamma - 1]
    soft:   p / (k_n w_i) = t^(1/4) [(7/2) ln t - 4 ln rho + C]^(1/4) - 1

i.e. the similarity exponents 1 and 1/4 times a slowly varying log factor.
These are parameter-free (no fitting); rho differs per run, since x_* does.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

from fracinj import detection, io, paths, physics, plotting
from fracinj.solvers import closed_form, similarity

sys.path.insert(0, str(paths.ROOT / "scripts"))
from run_simulation import default_parameters, run_simulation  # pyright: ignore

MQ_VALUES = np.geomspace(1e-14, 1e-9, 6)
RESULT_DIR = paths.results_dir("halfspace", "radial", "ramp_sweep")
MARKERS = ["o", "s", "^", "D", "v", "p", "<", ">"]

# moderate resolution: a few MB per run
NX, NT = 2**9, 1500


def intrinsic_scales(p) -> tuple[float, float]:
    """(t_*, x_*) of the radial ramp, see the module docstring."""
    k = physics.mobility(p)
    m_pq = p.m_q / (2 * np.pi)
    t_s = k * p.w_i**4 / m_pq
    return t_s, np.sqrt(physics.diffusivity(p) * t_s)


def simulation_length(p) -> float:
    """Rough time at which the pressure reaches L (the earlier of the two regimes);
    ``run_simulation`` truncates there, so overshooting only wastes steps."""
    k, D = physics.mobility(p), physics.diffusivity(p)
    m_pq = p.m_q / (2 * np.pi)
    A = (m_pq / k) ** 0.25
    B = np.sqrt(k * A**3)
    t_soft = (p.L / (1.3 * B)) ** (8 / 7)
    t_rigid = (p.L / 4) ** 2 / D
    return min(t_soft, t_rigid)


# %% run the sweep (only the missing files)

RESULT_DIR.mkdir(parents=True, exist_ok=True)
for m_q in MQ_VALUES:
    path = RESULT_DIR / f"m_q-{m_q:.0e}.hdf5"
    if path.exists():
        continue
    params = default_parameters("m_q", float(m_q), radial=True)
    params.Nx_p = params.Nx_sn = NX
    params.Nt = NT
    t_c_length = physics.cubic_scales(params, pin="length").t_c
    run_simulation(params, path, duration=1.2 * simulation_length(params) / t_c_length)

# %% load and extract the front and injection-pressure histories

runs = [io.read_halfspace(RESULT_DIR / f"m_q-{m_q:.0e}.hdf5") for m_q in MQ_VALUES]
scales = [intrinsic_scales(run.params) for run in runs]

fronts, pressures = [], []
for run, (t_s, x_s) in zip(runs, scales):
    x, sn = io.sort_fields(run.x_sc, run.sn)
    ell, n_keep = detection.find_stress_front(x, sn, L=run.params.L)
    t_hat = run.t[:n_keep] / t_s
    # drop the first steps, where the front sits within two cells of the source
    ok = ell > 2 * (x[1] - x[0])
    fronts.append((t_hat[ok], ell[ok] / x_s))

    _, p = io.sort_fields(run.x_sc, run.p)
    pressures.append((run.t / t_s, p[:, 0] / (run.params.k_n * run.params.w_i)))


def pooled(series):
    t = np.concatenate([s[0] for s in series])
    y = np.concatenate([s[1] for s in series])
    order = np.argsort(t)
    return t[order], y[order]


# %% front: analytical asymptotes and their sum (no fitting)

ALPHA_F, BETA_F = 1 / 2, 7 / 8
ZETA_F = closed_form.rigid_stress_front("neumann", i=1, geometry="radial")
ZETA_N = closed_form.front_approximation(3, "neumann", i=1, geometry="radial").zeta_front
t_nd_f = np.geomspace(pooled(fronts)[0].min(), pooled(fronts)[0].max(), 200)
rigid_f = ZETA_F * t_nd_f**ALPHA_F
soft_f = ZETA_N * t_nd_f**BETA_F
front_sum = rigid_f + soft_f
print(f"front: ell/x_* = {ZETA_F:.4g} t^1/2 + {ZETA_N:.4g} t^7/8")

# %% injection pressure: log-corrected similarity asymptotes at r_w (no fitting)

soft = similarity.solve(n=3, bc="neumann", i=1, geometry="radial")
z0 = soft.zeta[0]
C_SOFT = float(soft(z0)[0]) ** 4 + 4 * np.log(z0)  # theta^4 ~ C - 4 ln zeta


def p_rigid(t, rho):
    return t / 2 * (np.log(4 * t / rho**2) - np.euler_gamma - 1)


def p_soft(t, rho):
    return t**0.25 * (3.5 * np.log(t) - 4 * np.log(rho) + C_SOFT) ** 0.25 - 1


rhos = [np.exp(-2) * (run.x_sc[1] - run.x_sc[0]) / x_s for run, (_, x_s) in zip(runs, scales)]

# %% injection pressure: pure similarity power laws t and t^(1/4), for reference
#
# Prefactors fitted in log space in each regime separately (t_hat < 0.1 and
# t_hat > 1), exponents fixed. The gap to the data is the log factor in
# rho = r_w / x_* that the pure power laws leave out.

t_pp, y_pp = pooled(pressures)
early, late = t_pp < 0.1, t_pp > 1.0
a_p = np.exp(np.mean(np.log(y_pp[early]) - np.log(t_pp[early])))
b_p = np.exp(np.mean(np.log(y_pp[late]) - 0.25 * np.log(t_pp[late])))
t_pl_early = np.geomspace(t_pp[0], 0.1, 100)
t_pl_late = np.geomspace(1.0, t_pp[-1], 100)
print(f"p_inj: p/(k_n w_i) ~ {a_p:.3g} t (t < 0.1),  {b_p:.3g} t^1/4 (t > 1)")

# %% figure


def marker_style(i, p):
    return {
        "marker": MARKERS[i % len(MARKERS)],
        "mfc": "k" if i % 2 == 0 else "none",
        "mec": "k",
        "mew": 0.7,
        "ms": 3.5,
        "ls": "none",
        "markevery": 0.09,
        "label": rf"${plotting.sci_latex(p.m_q)}$",
    }


ASYM = {"color": "tab:gray", "ls": ":", "lw": 1.2}

CW, GR = 3.25, 1.618
fig = plt.figure(figsize=(2.3 * CW, CW / GR), dpi=150, layout="constrained",
                 clear=True, num=2)
ax_f, ax_p = fig.subplots(1, 2)

handles = []
for i, (run, fr, pr) in enumerate(zip(runs, fronts, pressures)):
    (h,) = ax_f.plot(*fr, **marker_style(i, run.params))
    ax_p.plot(*pr, **marker_style(i, run.params))
    handles.append(h)

(l_cross,) = ax_f.plot(t_nd_f, front_sum, "-", color="C0", lw=1.5,
                       label="Sum of asymptotes")
(l_asym,) = ax_f.plot(t_nd_f, rigid_f, **ASYM, label="Asymptotes")
ax_f.plot(t_nd_f, soft_f, **ASYM)

# one pair of asymptotes per run (rho differs), each over that run's time range,
# clipped to where the regime holds
for (t, _), rho in zip(pressures, rhos):
    if t[0] < 0.1:  # run reaches the rigid regime
        t_r = np.geomspace(t[0], min(t[-1], 0.1), 100)
        ax_p.plot(t_r, p_rigid(t_r, rho), **ASYM)
    if t[-1] > 1.0:  # run reaches the soft regime
        t_s = np.geomspace(max(t[0], 1.0), t[-1], 100)
        ax_p.plot(t_s, p_soft(t_s, rho), **ASYM)
POWER = {"color": "0.75", "ls": "-", "lw": 1.0, "zorder": 0}
(l_pow,) = ax_p.plot(t_pl_early, a_p * t_pl_early, **POWER, label=r"$\propto \hat{t},\ \hat{t}^{1/4}$")
ax_p.plot(t_pl_late, b_p * t_pl_late**0.25, **POWER)

ax_f.annotate(r"$\propto \hat{t}^{1/2}$", xy=(t_nd_f[40], rigid_f[40]),
              xytext=(-8, 8), textcoords="offset points", ha="right", color="tab:gray")
ax_f.annotate(r"$\propto \hat{t}^{7/8}$", xy=(t_nd_f[-40], soft_f[-40]),
              xytext=(8, -8), textcoords="offset points", va="top", color="tab:gray")
(t0, _), rho0 = pressures[0], rhos[0]
ax_p.annotate(r"$\propto \hat{t}\,\ln(\hat{t}/\rho^2)$", xy=(t0[5], p_rigid(t0[5], rho0)),
              xytext=(6, -6), textcoords="offset points", va="top", color="tab:gray")
ax_p.text(0.97, 0.35, r"$\propto \hat{t}^{1/4}\ln^{1/4}(\hat{t}^{7/2}/\rho^4)$",
          transform=ax_p.transAxes, ha="right", color="tab:gray",
          bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})

for ax in (ax_f, ax_p):
    ax.axvline(1, color="tab:gray", ls="--", lw=0.8)
    ax.set(xscale="log", yscale="log", xlabel=r"$\hat{t} = t / t_*$")

t_all, ell_all = pooled(fronts)
_, p_all = pooled(pressures)
ax_f.set(ylabel=r"$\ell / x_*$", ylim=(ell_all.min() / 2, ell_all.max() * 2))
ax_p.set(ylabel=r"$p_{\mathrm{inj}} / (k_n w_i)$", ylim=(p_all.min() / 2, p_all.max() * 2))
ax_f.legend(handles=[l_cross, l_asym], loc="upper left", frameon=False)
ax_p.legend(handles=[l_pow], loc="upper left", frameon=False)
plotting.annotate_panel_labels([ax_f, ax_p], xy=(0.9, 0.06))

fig.legend(
    handles=handles,
    title=r"$m_\mathrm{q}$ ($\mathrm{m^3/s^2}$)",
    loc="outside right center",
    borderpad=1.0,
    frameon=False,
)

fig.canvas.draw_idle()
plt.pause(0.01)

# %%

plotting.save_figure(fig, "front-dimensionless-radial", ext="pdf")
