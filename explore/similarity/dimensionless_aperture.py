import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

from fracinj import analysis, io, paths, physics, plotting

plt.style.use(["science", "bright", "latex-sans"])  # pyright: ignore[reportAttributeAccessIssue]
plt.rcParams["font.family"] = "sans-serif"

# %%
result_dir = paths.results_dir("halfspace", "wi-1e-05-debug")
q0_values = np.geomspace(1e-10, 1e-6, 5)

runs = [io.read_halfspace(result_dir / f"q_0-{q0:.0e}.hdf5") for q0 in q0_values]

keep_fraction = {0: 1 / 20, 1: 1 / 20, 2: 1 / 5}
results = [
    analysis.analyze_p_inj(run, slc=slice(int(len(run.t) * keep_fraction.get(i, 1.0))))
    for i, run in enumerate(runs)
]

result_rigid = analysis.p_inj_rigid(
    runs[0], slc=slice(int(len(runs[0].t) * keep_fraction[0]))
)
result_soft = analysis.p_inj_soft(runs[-1])

tc = [physics.const_rate_fixed_aperture_cubic(run.params).t_c for run in runs]
pc = [run.params.k_n * run.params.w_i for run in runs]
t_nd = np.concatenate([res.t / t_c for res, t_c in zip(results, tc)])
w_nd = np.concatenate([res.y / p_c for res, p_c in zip(results, pc)])

t_nd_positive = t_nd[t_nd > 0]
t_range_nd = np.geomspace(t_nd_positive.min(), t_nd_positive.max(), 200)
rigid_y = result_rigid.analytical(t_range_nd * tc[0]) / pc[0]
soft_y = result_soft.analytical(t_range_nd * tc[-1]) / pc[-1]

# %%

fig = plt.figure(dpi=200, layout="constrained", clear=True, num=1)
ax = fig.subplots()

sc_plots = []
for run, result, t_c, p_c in zip(runs, results, tc, pc):
    label = rf"$q_0={plotting.sci_latex(run.params.q_0)}\,\mathrm{{m^2 / s}}$"
    sc = ax.scatter(result.t / t_c, result.y / p_c, s=10, label=label)
    sc_plots.append(sc)

ax.axvline(1, color="tab:gray", ls="--")
ax.annotate(
    r"$\bar{t}_{\mathrm{c}}=1$",
    (1, 1e1),
    color="tab:gray",
    ha="center",
    backgroundcolor="1",
)

ax.plot(
    t_range_nd,
    rigid_y,
    "k-",
    label="Rigid",
)
plotting.slope_triangle(
    ax, x0=t_range_nd[20], y0_data=rigid_y[20], slope=result_rigid.alpha_ana, dx_log=1
)
ax.annotate(
    r"Rigid, $\theta_0 k_{\mathrm{n}} q_{0} \sqrt{ t/D }$",
    xy=(t_range_nd[60], rigid_y[60]),
    xytext=(20, -20),
    textcoords="offset points",
    ha="left",
    va="top",
    arrowprops={"arrowstyle": "<-"},
)

ax.plot(
    t_range_nd,
    soft_y,
    "k-",
    label="Soft",
)
plotting.slope_triangle(
    ax,
    x0=t_range_nd[-30],
    y0_data=soft_y[-30],
    slope=result_soft.alpha_ana,
    dx_log=1.0,
    inverse=True,
)
ax.annotate(
    r"Soft, $\theta_0 k_{\mathrm{n}} \left( q_{0}^{2} t / a \right)^{1/5}$",
    xy=(t_range_nd[-20], soft_y[-20]),
    xytext=(-20, -30),
    textcoords="offset points",
    ha="center",
    va="top",
    arrowprops={"arrowstyle": "<-"},
)

ax.legend(handles=sc_plots, loc="upper left", fontsize="small")

ax.set(
    xlabel=r"$t/t_{\mathrm{c}},\ t_{\mathrm{c}} = \dfrac{k_{\mathrm{n}} w_{\mathrm{i}}^{5}}{12 \mu q_{0}^{2}}$",
    ylabel=r"$p_{0} / p_{\mathrm{c}}, \ p_{c}= k_{\mathrm{n}} w_{\mathrm{i}}$",
    # title="Dimensionless pressure at the injector",
    xscale="log",
    yscale="log",
)
# fig.savefig(
#     "/home/kahmadov/phd/migration/documents/2026-09-10-summary/figures/dimless-pressure.eps"
# )
fig.canvas.draw_idle()
plt.pause(0.01)
