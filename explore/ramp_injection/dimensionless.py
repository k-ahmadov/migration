import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

from fracinj import analysis, io, paths, physics, plotting

plt.style.use(["science", "bright", "latex-sans"])  # pyright: ignore[reportAttributeAccessIssue]

# %%

result_dir = paths.results_dir("halfspace", "linear")
mq_values = np.geomspace(1e-14, 1e-7, 8)
runs = [io.read_halfspace(result_dir / f"m_q-{m_q:.0e}.hdf5") for m_q in mq_values]
bc = "ramp_rate"

# early runs: keep only the early part (late-time front detection is noisy)
keep_fraction = {0: 1 / 5, 1: 1 / 2, 2: 1 / 2}
results = [
    analysis.analyze_front(
        run,
        stress_front=True,
        slc=slice(int(len(run.t) * keep_fraction.get(i, 1.0))),
    )
    for i, run in enumerate(runs)
]

# pool every run onto dimensionless axes (t / t_c, x_f / x_c)
tc = [physics.characteristic_time(run.params, bc=bc) for run in runs]
xc = [physics.characteristic_distance(run.params, bc=bc) for run in runs]
t_nd = np.concatenate([res.t / t for res, t in zip(results, tc)])
x_nd = np.concatenate([res.y / x for res, x in zip(results, xc)])
positive = t_nd > 0
t_nd, x_nd = t_nd[positive], x_nd[positive]
t_range_nd = np.geomspace(t_nd.min(), t_nd.max(), 200)

# %%

fig = plt.figure(
    figsize=(3.5 * 2, 2.625*1.15), dpi=200, layout="constrained", clear=True, num=1
)
ax1, ax2 = fig.subplots(1, 2)

sc_plots = []
for run, result, t_c, x_c in zip(runs, results, tc, xc):
    label = (
        rf"$m_\mathrm{{q}}={plotting.sci_latex(run.params.m_q)}\,\mathrm{{m^2 / s}}$"
    )
    sc = ax2.scatter(result.t / t_c, result.y / x_c, s=10, label=label)
    sc_plots.append(sc)

fig.legend(handles=sc_plots, loc="outside upper center", fontsize="small", ncols=4)

for ax in [ax1, ax2]:
    ax.set(
        xlabel=r"$t/t_{\mathrm{c}},\ t_{\mathrm{c}} = \left( \dfrac{a w_{\mathrm{i}}^{5}}{m_{\mathrm{q}}^{2}} \right)^{1/3} $",
        # title="Dimensionless pressure at the injector",
        xscale="log",
        yscale="log",
    )
ax2.set(ylabel=r"$x / x_{\mathrm{c}},\ x_{\mathrm{c}} = \left( \dfrac{a^{2} w_{\mathrm{i}^{7}}}{m_{\mathrm{q}}} \right)^{1/3} $")

fig.canvas.draw_idle()
plt.pause(0.01)
