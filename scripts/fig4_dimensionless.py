
import sys

import matplotlib.pyplot as plt
import numpy as np

from fracinj import analysis, io, paths, physics, plotting
from fracinj.solvers import closed_form

sys.path.insert(0, str(paths.ROOT / "scripts"))
from run_simulation import default_parameters, run_simulation  # pyright: ignore

COLUMN_WIDTH = 3.25  # in
MARKERS = ["o", "s", "^", "D", "v", "p", "<", ">"]
ALPHA, BETA = 1 / 2, 7 / 5  # planar ramp front: rigid t^(1/2), soft t^(7/5)

# analytical asymptotes in the t_c, x_c units used here (x_c^2 = D t_c):
#   rigid: ell / x_c = ZETA_F t_hat^(1/2), ZETA_F = 2 eta_f = 1.343, the most
#          tensile point of the stress from the exact linear (i^3 erfc) pressure
#   soft:  ell / x_c = ZETA_N t_hat^(7/5), ZETA_N = 0.588, the n=3 front-form
#          approximation (the full shooting solution gives 0.586)
ZETA_F = closed_form.rigid_stress_front("neumann", i=1)
ZETA_N = closed_form.front_approximation(3, "neumann", i=1).zeta_front

# one run per rate, alternating decades: the low rates (domain L = 100 m) cover
# the rigid regime and the crossover, the high rates (L = 3 km, so the front
# has room to run) reach deep into the soft regime. Together they span
# t_hat ~ 3e-4 to 2e3 continuously; L does not enter the dimensionless result.
MQ_VALUES = [1e-14, 1e-12, 1e-10]
LONG_DIR = paths.results_dir("halfspace", "ramp", "soft_long")
LONG_MQ = [1e-8, 1e-6]
LONG_L, LONG_NX, LONG_NT = 3000.0, 2**10, 2000

# %%


def marker_style(i, p):
    """Marker style for run i, matching dimensionless.py."""
    return {
        "marker": MARKERS[i % len(MARKERS)],
        "mfc": "k" if i % 2 == 0 else "none",
        "mec": "k",
        "mew": 0.7,
        "ms": 3.5,
        "ls": "none",
        "markevery": 0.1,
        "label": rf"${plotting.sci_latex(p.m_q)}$",
    }


# %% run the long-domain sweep (only the missing files)

LONG_DIR.mkdir(parents=True, exist_ok=True)
for m_q in LONG_MQ:
    path = LONG_DIR / f"m_q-{m_q:.0e}.hdf5"
    if path.exists():
        continue
    params = default_parameters("m_q", float(m_q))
    params.L, params.Nx_p, params.Nx_sn, params.Nt = LONG_L, LONG_NX, LONG_NX, LONG_NT
    sc = physics.ramp_rate_fixed_aperture_cubic(params)
    # soft front reaches L at t_hat = (L / (ZETA_N x_c))^(5/7); run_simulation
    # truncates once the pressure gets there
    t_end = 1.3 * sc.t_c * (LONG_L / (ZETA_N * sc.x_c)) ** (1 / BETA)
    run_simulation(params, path, duration=t_end / physics.cubic_scales(params, pin="length").t_c)

# %%

result_dir = paths.results_dir("halfspace", "ramp")
runs = [io.read_halfspace(result_dir / f"m_q-{m_q:.0e}.hdf5") for m_q in MQ_VALUES]
runs += [io.read_halfspace(LONG_DIR / f"m_q-{m_q:.0e}.hdf5") for m_q in LONG_MQ]

# early runs: keep only the early part (late-time front detection is noisy)
keep_fraction = {0: 1 / 1, 1: 1 / 1, 2: 1 / 1}
slices = [
    slice(int(len(run.t) * keep_fraction.get(i, 1.0))) for i, run in enumerate(runs)
]

results_front = [
    analysis.analyze_front(run, stress_front=True, slc=slc)
    for run, slc in zip(runs, slices)
]
# pool every run onto dimensionless axes (t / t_c, x_f / x_c); drop points
# where the front sits within two cells of the source (not resolved)
scales = [physics.ramp_rate_fixed_aperture_cubic(run.params) for run in runs]
t_pooled, x_pooled = [], []
for run, res, sc in zip(runs, results_front, scales):
    resolved = (res.t > 0) & (res.y > 2 * abs(run.x_sc[1] - run.x_sc[0]))
    t_pooled.append(res.t[resolved] / sc.t_c)
    x_pooled.append(res.y[resolved] / sc.x_c)
t_pooled, x_pooled = np.concatenate(t_pooled), np.concatenate(x_pooled)
t_nd = np.geomspace(t_pooled.min(), t_pooled.max(), 200)

# no fitting: both prefactors are analytical. The stress front overshoots both
# asymptotes in the transition (by ~2x at t_hat ~ 1-10), so a weighted blend
# of them cannot follow it; their sum can, within -3%/+12% over all decades:
# the stress front runs ahead of the soft aperture front by about the rigid
# diffusion length.
rigid_front = ZETA_F * t_nd**ALPHA
soft_front = ZETA_N * t_nd**BETA
x_front_sum = rigid_front + soft_front
T_CROSS = (ZETA_F / ZETA_N) ** (1 / (BETA - ALPHA))  # asymptotes intersect
print(f"front: ell/x_c = {ZETA_F:.4g} t^1/2 + {ZETA_N:.4g} t^7/5, "
      f"asymptotes cross at t_hat = {T_CROSS:.3g}")

# %%


CW, GR = 3.25, 1.618
fig = plt.figure(
    figsize=(1.4 * CW, CW / GR),
    dpi=150,
    layout="constrained",
    clear=True,
    num=2,
)
ax = fig.subplots()

handles = []
for i, (run, rf, sc) in enumerate(zip(runs, results_front, scales)):
    resolved = (rf.t > 0) & (rf.y > 2 * abs(run.x_sc[1] - run.x_sc[0]))  # as in the fit
    (h,) = ax.plot(rf.t[resolved] / sc.t_c, rf.y[resolved] / sc.x_c,
                   **marker_style(i, run.params))
    handles.append(h)

(l_cross,) = ax.plot(
    t_nd, x_front_sum, "-", color="C0", lw=1.5, label="Sum of asymptotes"
)
(l_asym,) = ax.plot(
    t_nd, rigid_front, color="tab:gray", ls=":", lw=1.2, label="Asymptotes"
)
ax.plot(t_nd, soft_front, color="tab:gray", ls=":", lw=1.2)
ax.annotate(
    r"$\eta_{\mathrm{f}}\, 2 \sqrt{ D t }$",
    xy=(t_nd[50], rigid_front[50]),
    xytext=(10, 30),
    textcoords="offset points",
    ha="right",
    arrowprops={"arrowstyle": "<-"},
)
ax.annotate(
    r"$\zeta_{\mathrm{N}} \left( a m_{\mathrm{q}}^{3} t^{7} \right)^{1/5}  $",
    xy=(t_nd[-30], soft_front[-30]),
    xytext=(-5, -50),
    textcoords="offset points",
    ha="center",
    va="top",
    arrowprops={"arrowstyle": "<-"},
)

ax.axvline(1, color="tab:gray", ls="--")
ax.annotate(
    r"$\hat{t}=1$",
    (1, 1e2),
    color="tab:gray",
    ha="center",
    backgroundcolor="1",
)
ax.set(
    xlabel=r"$\hat{t}$",
    ylabel=r"$\ell / x_{*}$",
    xscale="log",
    yscale="log",
    ylim=(x_pooled.min() / 2, x_pooled.max() * 2),
)
ax.legend(handles=[l_cross, l_asym], loc="upper left")
# one-column m_q legend right of the axes; as a figure legend, constrained
# layout sizes it to its contents, so the axes take the remaining width.
# The frame is off, so borderpad (default 0.4) only widens the gap to the axes.
leg = fig.legend(
    handles=handles,
    title=r"$m_\mathrm{q}$ ($\mathrm{m^2/s^2}$)",
    loc="outside right center",
    borderpad=1.0,
    ncols=1,
    frameon=False,
    # handletextpad=0.2,
    # labelspacing=0.3,
)

fig.canvas.draw_idle()
plt.pause(0.01)

# %%

plotting.save_figure(fig, "front-dimensionless", ext='pdf')
