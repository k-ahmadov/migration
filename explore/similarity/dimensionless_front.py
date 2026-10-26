import matplotlib.pyplot as plt
import numpy as np

from fracinj import analysis, io, math_utils, paths, physics, plotting

plt.style.use(["science", "bright"])

# %%

# ---------------------------------------------------------------------------
# Load & analyze runs
# ---------------------------------------------------------------------------

result_dir = paths.results_dir("halfspace", "wi-1e-05")
q0_values = np.geomspace(1e-10, 1e-6, 5)

runs = [io.read_halfspace(result_dir / f"q_0-{q0:.0e}.hdf5") for q0 in q0_values]

keep_fraction = {0: 1 / 50, 1: 1 / 20, 2: 1 / 5, 3: 1 / 2}

results = [
    analysis.analyze_front(
        run, stress_front=True, slc=slice(int(len(run.t) * keep_fraction.get(i, 1.0)))
    )
    for i, run in enumerate(runs)
]

result_rigid = analysis.front_rigid_const(
    runs[0], stress_front=True, slc=slice(int(len(runs[0].t) * keep_fraction[0]))
)
result_soft = analysis.front_soft_const(runs[-1], stress_front=True)

scales = [physics.const_rate_fixed_aperture_cubic(run.params) for run in runs]
tc = [scale.t_c for scale in scales]
xc = [scale.x_c for scale in scales]
t_nd = np.concatenate([res.t / t_c for res, t_c in zip(results, tc)])
x_nd = np.concatenate([res.y / x_c for res, x_c in zip(results, xc)])

t_nd_positive = t_nd[t_nd > 0]
t_range_nd = np.geomspace(t_nd_positive.min(), t_nd_positive.max(), 200)
rigid_y = result_rigid.analytical(t_range_nd * tc[0]) / xc[0]
soft_y = result_soft.analytical(t_range_nd * tc[-1]) / xc[-1]

# %% fit a function that combines both power laws

fig = plt.figure(dpi=200, layout="constrained", clear=True, num=1)
ax = fig.subplots()

sc_plots = []
for run, result, t_c, x_c in zip(runs, results, tc, xc):
    label = rf"$q_0={plotting.sci_latex(run.params.q_0)}\,\mathrm{{m^2 / s}}$"
    sc = ax.scatter(result.t / t_c, result.y / x_c, s=10, label=label)
    sc_plots.append(sc)

ax.axvline(1, color="tab:gray", ls="--")
ax.annotate(
    r"$\bar{t}_{\mathrm{c}}=1$",
    (1, 1e2),
    color="tab:gray",
    ha="center",
    backgroundcolor="1",
)

ax.plot(t_range_nd, rigid_y, "k-")
plotting.slope_triangle(
    ax, x0=t_range_nd[3], y0_data=rigid_y[3], slope=result_rigid.alpha_ana, dx_log=1
)
ax.annotate(
    r"Rigid",
    xy=(t_range_nd[30], rigid_y[30]),
    xytext=(10, 30),
    textcoords="offset points",
    ha="right",
    va="top",
    arrowprops={"arrowstyle": "<-"},
)

ax.plot(t_range_nd, soft_y, "k-")
plotting.slope_triangle(
    ax,
    x0=t_range_nd[-50],
    y0_data=soft_y[-50],
    slope=result_soft.alpha_ana,
    dx_log=1.0,
    inverse=True,
)
ax.annotate(
    r"Soft",
    xy=(t_range_nd[-20], soft_y[-20]),
    xytext=(0, -30),
    textcoords="offset points",
    ha="center",
    va="top",
    arrowprops={"arrowstyle": "<-"},
)

ax.legend(handles=sc_plots, fontsize="small")

ax.set(
    xlabel=r"$t/t_{\mathrm{c}},\ t_{\mathrm{c}} = \dfrac{k_{\mathrm{n}} w_{\mathrm{i}}^{5}}{12 \mu q_{0}^{2}}$",
    ylabel=r"$x_{\mathrm{f}} / x_{\mathrm{c}},\ x_{\mathrm{c}} = \dfrac{k_{\mathrm{n}} w_{\mathrm{i}}^4}{12 \mu q_0}$",
    # title="Dimensionless pressure at the injector",
    xscale="log",
    yscale="log",
)
# fig.savefig(
#     "/home/kahmadov/phd/migration/tex/2026-09-10_summary/figures/dimless-front.eps"
# )
fig.canvas.draw_idle()
plt.pause(0.01)


# %% a plot to check single simulation

idx = -3
title = rf"$q_0={plotting.sci_latex(runs[idx].params.q_0)}\,\mathrm{{m^2/ s}},\ \alpha = {results[idx].alpha_emp:.2f}$"
fig = plt.figure(
    figsize=(6.4 / 1.5, 4.8 / 1.5), dpi=150, layout="tight", clear=True, num=1
)
ax = fig.subplots()
ax.plot(results[idx].t, results[idx].y, ".")
ax.set(xlabel="Time [s]", ylabel="Front position [m]", title=title)
fig.canvas.draw_idle()
plt.pause(0.01)


# %%


delta_crossover = 5e-2
x_front_crossover = math_utils.crossover(
    x=t_range_nd,
    a=result_rigid.A_ana * 13,
    alpha=result_rigid.alpha_ana,
    b=result_soft.A_ana * 14,
    beta=result_soft.alpha_ana,
    x0=1,
    delta=delta_crossover,
)

# %%

fig = plt.figure(dpi=200, layout="constrained", clear=True, num=1)
ax = fig.subplots()

sc_plots = []
for run, result, t_c, x_c in zip(runs, results, tc, xc):
    label = rf"$q_0={plotting.sci_latex(run.params.q_0)}\,\mathrm{{m^2 / s}}$"
    sc = ax.scatter(result.t / t_c, result.y / x_c, s=10, label=label)
    sc_plots.append(sc)

ax.axvline(1, color="tab:gray", ls="--")
ax.annotate(
    r"$\bar{t}_{\mathrm{c}}=1$",
    (1, 1e2),
    color="tab:gray",
    ha="center",
    backgroundcolor="1",
)

(l1,) = ax.plot(
    t_range_nd,
    x_front_crossover,
    "k--",
    label=rf"Crossover, $\delta={delta_crossover}$",
)

leg1 = ax.legend(handles=sc_plots, fontsize="small")
ax.add_artist(leg1)

ax.legend(handles=[l1])

ax.set(
    xlabel=r"$t/t_{\mathrm{c}},\ t_{\mathrm{c}} = \dfrac{k_{\mathrm{n}} w_{\mathrm{i}}^{5}}{12 \mu q_{0}^{2}}$",
    ylabel=r"$x_{\mathrm{f}} / x_{\mathrm{c}},\ x_{\mathrm{c}} = \dfrac{k_{\mathrm{n}} w_{\mathrm{i}}^4}{12 \mu q_0}$",
    # title="Dimensionless pressure at the injector",
    xscale="log",
    yscale="log",
)
# fig.savefig(
#     "/home/kahmadov/phd/migration/tex/2026-09-10_summary/figures/dimless-front-crossover.eps"
# )
fig.canvas.draw_idle()
plt.pause(0.01)
