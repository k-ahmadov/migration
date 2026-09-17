import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

from fracinj import analysis, io, paths, physics, plotting

plt.style.use(["science", "bright", "latex-sans"])  # pyright: ignore[reportAttributeAccessIssue]
plt.rcParams["font.family"] = "sans-serif"

# %%

result_dir = paths.results_dir("halfspace", "linear")
mq_values = np.geomspace(1e-14, 1e-7, 8)
runs = [io.read_halfspace(result_dir / f"m_q-{m_q:.0e}.hdf5") for m_q in mq_values]
bc = "ramp_rate"

# early runs: keep only the early part (late-time front detection is noisy)
keep_fraction = {0: 1 / 5, 1: 1 / 2, 2: 1 / 2}
results_front = [
    analysis.analyze_front(
        run,
        stress_front=True,
        slc=slice(int(len(run.t) * keep_fraction.get(i, 1.0))),
    )
    for i, run in enumerate(runs)
]

results_p_inj = [
    analysis.analyze_p_inj(run, slc=slice(int(len(run.t) * keep_fraction.get(i, 1.0))))
    for i, run in enumerate(runs)
]

result_p_inj_rigid = analysis.p_inj_rigid_ramp(
    runs[0], slc=slice(int(len(runs[0].t) * keep_fraction[0]))
)
result_p_inj_soft = analysis.p_inj_soft_ramp(runs[-1])

# pool every run onto dimensionless axes (t / t_c, x_f / x_c)
scales = [physics.ramp_rate_fixed_aperture_cubic(run.params) for run in runs]
tc = [scale.t_c for scale in scales]
xc = [scale.x_c for scale in scales]
pc = [run.params.k_n * scale.w_c for run, scale in zip(runs, scales)]

t_nd = np.concatenate([res.t / t for res, t in zip(results_front, tc)])
x_nd = np.concatenate([res.y / x for res, x in zip(results_front, xc)])
positive = t_nd > 0
t_nd, x_nd = t_nd[positive], x_nd[positive]
t_range_nd = np.geomspace(t_nd.min(), t_nd.max(), 200)
rigid_p_inj = result_p_inj_rigid.analytical(t_range_nd * tc[0]) / pc[0]
soft_p_inj = result_p_inj_soft.analytical(t_range_nd * tc[-1]) / pc[-1]

# %%

fig = plt.figure(
    figsize=(3.5 * 2, 2.625 * 1.15), dpi=120, layout="constrained", clear=True, num=1
)
ax1, ax2 = fig.subplots(1, 2)

sc_plots = []
for run, result_front, result_p_inj, scale in zip(
    runs, results_front, results_p_inj, scales
):
    label = (
        rf"$m_\mathrm{{q}}={plotting.sci_latex(run.params.m_q)}\,\mathrm{{m^2 / s}}$"
    )
    sc = ax1.scatter(
        result_p_inj.t / scale.t_c, result_p_inj.y / (scale.w_c * run.params.k_n), s=10
    )
    sc = ax2.scatter(
        result_front.t / scale.t_c, result_front.y / scale.x_c, s=10, label=label
    )
    sc_plots.append(sc)

fig.legend(handles=sc_plots, loc="outside upper center", fontsize="small", ncols=4)

ax1.plot(
    t_range_nd,
    rigid_p_inj,
    "k-"
)
ax1.plot(
    t_range_nd,
    soft_p_inj,
    "k-"
)

for ax in [ax1, ax2]:
    ax.set(
        xlabel=r"$t/t_{\mathrm{c}},\ t_{\mathrm{c}} = \left( \dfrac{a w_{\mathrm{i}}^{5}}{m_{\mathrm{q}}^{2}} \right)^{1/3} $",
        # title="Dimensionless pressure at the injector",
        xscale="log",
        yscale="log",
    )
ax1.set(
    ylabel=r"$p_{0} / p_{\mathrm{c}}, \ p_{c}= k_{\mathrm{n}} w_{\mathrm{i}}$",
)
ax2.set(
    ylabel=r"$x / x_{\mathrm{c}},\ x_{\mathrm{c}} = \left( \dfrac{a^{2} w_{\mathrm{i}^{7}}}{m_{\mathrm{q}}} \right)^{1/3} $"
)

fig.canvas.draw_idle()
plt.pause(0.01)
